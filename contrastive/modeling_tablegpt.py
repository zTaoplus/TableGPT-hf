'''
model that supports cross-table generation
'''
from typing import Optional, List, Tuple, Union, Dict,Literal


import torch
import torch.nn as nn
import transformers
import pandas as pd

from transformers import PreTrainedModel,PretrainedConfig,PreTrainedTokenizer
from transformers import Qwen2ForCausalLM,GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_tablegpt import TableGPTConfig
from .modeling_tablegpt_encoder import TablegptEncoderModel
from .utils import build_prompt

IGNORE_INDEX = -100 

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


# copied from vllm
def merge_multimodal_embeddings(input_ids: torch.Tensor,
                                inputs_embeds: torch.Tensor,
                                multimodal_embeddings: torch.Tensor,
                                placeholder_token_id: int) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    mask = (input_ids == placeholder_token_id)
    num_expected_tokens = mask.sum()

    if isinstance(multimodal_embeddings, torch.Tensor):
        batch_size, batch_tokens, embed_dim = multimodal_embeddings.shape
        total_tokens = batch_size * batch_tokens
        if num_expected_tokens != total_tokens:
            expr = f"{batch_size} x {batch_tokens}"
            raise ValueError(
                f"Attempted to assign {expr} = {total_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders")

        inputs_embeds[mask] = multimodal_embeddings.view(
            total_tokens, embed_dim)
    else:
        raise
    return inputs_embeds


# FIXME: unnecessary create the class for projector
# but the weight key has the model prefix, like projector.model.0.bias.
# currently cannot load the preojector weights and update weights.
class Projector(PreTrainedModel):
    def __init__(self,config:PretrainedConfig):
        super().__init__(config)

        self.config = config
        mlp_depth = self.config.mlp_depth
        encoder_hidden_size =  self.config.encoder_hidden_size
        decoder_hidden_size = self.config.decoder_hidden_size
        
        num_heads = self.config.num_heads
        
        if not self.config.multihead:
            num_heads = 1
        
        modules = [
            nn.Linear(encoder_hidden_size, decoder_hidden_size * num_heads)
        ]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(
                nn.Linear(decoder_hidden_size * num_heads,
                            encoder_hidden_size * num_heads))
        
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        ret = self.model(x)
        if self.config.multihead:
            ret = ret.view(*ret.shape[:-1], self.num_heads, -1)
        return ret
        

class TableGPTChatModel(PreTrainedModel):
    config_class = TableGPTConfig
    _supports_flash_attn_2 = True
    _no_split_modules = ['Qwen2DecoderLayer']

    def __init__(self,config: TableGPTConfig, use_flash_attn=True):
        
        super().__init__(config)


        # see https://huggingface.co/Qwen/Qwen2-7B#requirements
        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        self.config = config
        
        self.generation_config = GenerationConfig.from_model_config(self.config)
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        
        self.decoder = Qwen2ForCausalLM(config.llm_config)

        if not self.config.encoder_config:
            raise ValueError(
                "table encoder configs cannot found in hf config.")

        self.encoder = TablegptEncoderModel(config.name_or_path, self.config.encoder_config)

        self.projector = Projector(self.config.projector_config)


        self.insert_seq_token = self.config.encoder_config.insert_seq_token
        self.insert_embs_token = self.config.encoder_config.insert_embs_token
        self.insert_embs_token_id = self.config.encoder_config.insert_embs_token_id

        self.table_placeholder = self.insert_seq_token + self.insert_embs_token  + self.insert_seq_token
        
        
    def _table_tokenizer_insert(self, prompt: str, tokenizer: PreTrainedTokenizer,merge_embeds_type=Literal["vllm","original"]) -> List[int]:
        '''
        Tokenizes the input prompt by inserting a separator token 
        between each chunk of text.

        Args:
            prompt (str): The input prompt to be tokenized. 
                        It contains one or more instances of the 
                        INSERT_EMBS_TOKEN.
            tokenizer (transformers.PreTrainedTokenizer): 
                The tokenizer object used for tokenization.

        Returns:
        List[int]: The tokenized input prompt as a list of input IDs. 

        '''


        prompt_chunks = [
            tokenizer(e,
                    padding="longest",
                    max_length=tokenizer.model_max_length,
                    truncation=True).input_ids
            for e in prompt.split(self.insert_embs_token)
        ]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X))
                    for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[
                0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        token_cnt = 3 if merge_embeds_type =="vllm" else 1

        for x in insert_separator(prompt_chunks,
                                [self.insert_embs_token_id] *  token_cnt  * (offset + 1)):
            input_ids.extend(x[offset:])

        return torch.tensor([input_ids])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        path_emb: Optional[str] = None,
        path_csv: Optional[str] = None,
        insert_embs = None # 是否往中间插入embedding
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # get table embeddings
        bs = input_ids.shape[0]
        table_embeds = self.get_encoder_output(path_csv, path_emb)
        prepare_embs_func = self.projector.prepare_embeds if insert_embs == None or insert_embs[0] == False else self.projector.prepare_insert_embeds
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = prepare_embs_func(
            decoder = self.decoder,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            table_embeds=table_embeds,
        )

        return self.decoder.forward(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.to(dtype = self.decoder.dtype),
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # cache_position=cache_position,
            return_dict=return_dict
        )
    
    @torch.inference_mode()
    def generate(self,
                input_ids: Optional[torch.Tensor]= None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                table_embeds:Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                visual_features: Optional[torch.FloatTensor] = None,
                generation_config: Optional[GenerationConfig] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        generation_cfg = generation_config if generation_config is not None else self.generation_config
        print(f"generation cfg:{generation_config}")
        if inputs_embeds is not None:
            return self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                visual_features=visual_features,
                generation_config=generation_cfg,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        merge_embeds_type = kwargs.pop("merge_embeds_type","vllm")
        if merge_embeds_type == "vllm":
            cur_table_embeds = self.projector(table_embeds)

            cur_input_embeds = self.decoder.get_input_embeddings()(input_ids.clamp(min=0))

            inputs_embeds = merge_multimodal_embeddings(
                input_ids, cur_input_embeds,cur_table_embeds, self.insert_embs_token_id
            )

            
            del table_embeds, cur_table_embeds, cur_input_embeds

        else:
            _, _, attention_mask, _, inputs_embeds, _ = self.prepare_insert_embeds(
                input_ids=input_ids,
                table_embeds=table_embeds.unsqueeze(0)
            )

        
        input_ids = None
        

        return self.decoder.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_cfg,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # tables: ("var",df)
    def chat(self, 
            tokenizer:PreTrainedTokenizer, 
            query:str,
            tables:Optional[Dict[str,pd.DataFrame]], 
            max_rows:int=50,
            max_cols:int=100,
            history:Optional[List]= None,
            return_history:bool=False,
            verbose:bool=False,
            **generation_kwargs):
        
        history, full_prompt = build_prompt(query,tables,tokenizer,self.table_placeholder, history=history)

        if verbose:
            print(f"apply chat template full prompt is: {full_prompt}")

        merge_embeds_type = generation_kwargs.pop("merge_embeds_type","vllm")
        # 2. then use the custom tokenize
        input_ids = self._table_tokenizer_insert(full_prompt,tokenizer,merge_embeds_type)
        # get the table embeddings
        table_embeds = self.encoder.get_encoder_output(list(tables.values()),max_rows,max_cols)

        self.generation_config.update(**generation_kwargs)
    
        generation_output = self.generate(
            input_ids=input_ids.to(device=self.decoder.device),
            table_embeds=table_embeds.to(device=self.decoder.device),
            generation_config= self.generation_config,
            use_cache=True,
            merge_embeds_type=merge_embeds_type
        )

        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]

        history.append(("ai", response))

        return response, history if return_history else None
        
        
    def batch_chat(self):
        pass

    def prepare_insert_embeds(
        self, *, input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None, table_embeds, learnable_embeds = None
    ):
        assert learnable_embeds == None, "learnable embeddings is not yet supported"
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_insert_embs = (cur_input_ids == self.insert_embs_token_id).sum()
            if num_insert_embs == 0:
                raise ValueError("No insert embs token found in the input_ids")
            cur_table_embeds = table_embeds[batch_idx].clone()
            cur_table_embeds = self.projector(cur_table_embeds) # forward through the projector
            
            insert_emb_token_indices = [-1] + torch.where(cur_input_ids == self.insert_embs_token_id)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(insert_emb_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[insert_emb_token_indices[i]+1:insert_emb_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[insert_emb_token_indices[i]+1:insert_emb_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.decoder.get_input_embeddings()((torch.cat(cur_input_ids_noim)))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_insert_embs + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_insert_embs:
                    cur_insert_emb_features = cur_table_embeds[i] # num_heads * decode_hidden
                    if self.projector.config.multihead:
                        assert cur_insert_emb_features.shape == (self.num_heads, self.decoder_hidden_size), f"not match: {cur_insert_emb_features.shape}, f{(self.num_heads), self.decoder_hidden_size}"
                    cur_new_input_embeds.append(cur_insert_emb_features)
                    cur_new_labels.append(torch.full((cur_insert_emb_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
            device = next(self.decoder.parameters()).device
            cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.decoder.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            # new_labels = _labels

        if _attention_mask is None:
            pass # keep the newly created attention mask
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels