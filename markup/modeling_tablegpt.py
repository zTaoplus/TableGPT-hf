'''
model that supports cross-table generation
'''
from typing import Optional, List, Tuple, Union, Dict


import torch
import torch.nn as nn
import transformers
import pandas as pd

from transformers import PreTrainedModel,PreTrainedTokenizer
from transformers import Qwen2ForCausalLM,GenerationConfig,AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_tablegpt import TableGPTConfig
from .modeling_codet5p import CodeT5pModel
from .utils import build_markup_prompt

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



class TableGPTChatModel(PreTrainedModel):
    config_class = TableGPTConfig
    _supports_flash_attn_2 = True
    _no_split_modules = ['Qwen2DecoderLayer']

    def __init__(self,config: TableGPTConfig, use_flash_attn=True):
        
        super().__init__(config)


        # see https://huggingface.co/Qwen/Qwen2-7B#requirements
        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        self.config = config
        
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        
        self.decoder = Qwen2ForCausalLM(config.llm_config)

        if not self.config.encoder_config:
            raise ValueError(
                "table encoder configs cannot found in hf config.")

        self.encoder = CodeT5pModel(self.config.encoder_config)

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path,subfolder=self.config.encoder_config.subfolder)

        encoder_hidden_size=self.config.encoder_hidden_size
        decoder_hidden_size=self.config.decoder_hidden_size
        
        modules = [nn.Linear(encoder_hidden_size, decoder_hidden_size)]
        
        for _ in range(1, self.config.mlp_depth):
            
            modules.append(nn.GELU())
            
            modules.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
        
        self.projector = nn.Sequential(*modules)

        # generation config
        self.generation_config = GenerationConfig.from_model_config(self.config)

    def _table_tokenizer_insert(self, prompt: str, tokenizer: PreTrainedTokenizer) -> List[int]:
        placehoder_token = self.config.placeholder_token
        placehoder_token_id = self.config.placeholder_token_id

        prompt_chunks = [
            tokenizer(e,
                    padding="longest",
                    max_length=tokenizer.model_max_length,
                    truncation=True).input_ids
            for e in prompt.split(placehoder_token)
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

        for x in insert_separator(prompt_chunks,
                                [placehoder_token_id] * 1 * (offset + 1)):
            input_ids.extend(x[offset:])

        return torch.tensor(input_ids)

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
                attention_mask: Optional[torch.LongTensor] = None,
                visual_features: Optional[torch.FloatTensor] = None,
                generation_config: Optional[GenerationConfig] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **generation_kwargs,
                ):
        

        generation_cfg = generation_config if generation_config is not None else self.generation_config
        
        print(generation_config)

        if inputs_embeds is not None:
            return self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                visual_features=visual_features,
                generation_config=generation_cfg,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **generation_kwargs
            )

        return self.decoder.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_cfg,
            **generation_kwargs
        )
    
    def _input_proceesor(self,input_ids,tables:Dict[str,pd.DataFrame]):
        
        placeholder_token_id = self.config.placeholder_token_id
        max_length = self.config.encoder_max_length

        indices = torch.where(input_ids == -114)[0]

        new_prompt_token_ids = None 
        

        to_encoding = [df.to_markdown() for _,df in tables.items()]
   
        table_encoder_token_ids = self.encoder_tokenizer(to_encoding, return_tensors="pt", truncation=True, max_length=max_length).input_ids

        new_prompt_token_ids = input_ids.clone()
        for idx in indices:
            new_prompt_token_ids = torch.cat((new_prompt_token_ids[:idx], 
                                              torch.full((table_encoder_token_ids.shape[-1],), placeholder_token_id), 
                                              new_prompt_token_ids[idx+1:]))

        return new_prompt_token_ids,table_encoder_token_ids

    def chat(self, 
            tokenizer:PreTrainedTokenizer, 
            query:str,
            tables:Optional[Dict[str,pd.DataFrame]]=None,
            # table:Optional[pd.DataFrame] = None,
            history:Optional[List]= None,
            return_history:bool=False,
            verbose:bool=False,
            **generation_kwargs):
        
        history, full_prompt = build_markup_prompt(query,tokenizer,tables, self.config.placeholder_token, history=history)

        if verbose:
            print(f"apply chat template full prompt is: {full_prompt}")

        # 2. then use the custom tokenize
        input_ids = self._table_tokenizer_insert(full_prompt,tokenizer)
        inputs_embeds = None

        if tables is not None:
            input_ids,table_encoded_ids = self._input_proceesor(input_ids,tables)
            
            table_embeds = self.encoder(input_ids=table_encoded_ids.to(device=self.encoder.device)).last_hidden_state
            # _, _, _, _, inputs_embeds, _ = self._prepare_embeds(
            #     input_ids.unsqueeze(0),
            #     table_embeds.unsqueeze(0)
            # )
            self.projector.to(dtype=table_embeds.dtype)
            cur_table_embeds = self.projector(table_embeds)

            cur_input_embeds = self.decoder.get_input_embeddings()(input_ids.clamp(min=0).to(device=self.decoder.device))
            
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,cur_input_embeds,cur_table_embeds,self.config.placeholder_token_id
            )

            del table_embeds

            input_ids = None

        # generation_kwargs.setdefault("max_new_tokens",128)
        
        self.generation_config.update(**generation_kwargs)

        generation_output = self.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds.unsqueeze(0),
            use_cache=True,
            generation_config=self.generation_config
            # **generation_kwargs
        )

        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]

        history.append(("ai", response))

        return response, history if return_history else None
        
        
    def batch_chat(self):
        pass

    def _prepare_embeds(
        self, input_ids, table_imbeds, position_ids=None, attention_mask=None, past_key_values=None, labels=None,
    ):
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _attention_mask = attention_mask
        if attention_mask is None:
            #print(input_ids)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()

        input_ids = [cur_input_ids[cur_attention_mask].to(self.decoder.device) for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]

        new_input_embeds = []
        # new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_embeds = self.decoder.model.get_input_embeddings()(cur_input_ids)
            if table_imbeds is not None:
                self.projector.to(dtype=table_imbeds.dtype)
                cur_table_embeds = table_imbeds[batch_idx].clone()
                cur_table_embeds = self.projector(cur_table_embeds) # forward through the projector
                new_input_embeds.append(torch.cat([cur_table_embeds, cur_input_embeds], dim=0))
            else:
                new_input_embeds.append(cur_input_embeds)
            # new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))


        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.decoder.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _attention_mask is None:
            pass # keep the newly created attention mask
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, None