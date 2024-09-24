
from typing import List
from einops import rearrange
from transformers import PreTrainedModel ,AutoModel,AutoTokenizer
from torch.nn import functional as F
import torch
from torch import einsum
import torch.nn as nn

import numpy as np
import pandas as pd



from .configuration_tablegpt_enc import TableGPTEncoderConfig


def mask_fill_value(dtype=torch.float16):
    return torch.finfo(dtype).min if dtype == torch.float16 else float("-1e10")


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim * mult, dim))

    def forward(self, x, **kwargs):
        return self.net(x)


class RowColAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # s = batch size
        # b = number of rows
        # h = number of heads
        # n = number of columns
        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s b h n d', h=h),
                      (q, k, v))
        sim = einsum('s b h i d, s b h j d -> s b h i j', q, k) * self.scale

        # masking
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(
                1, sim.shape[1], sim.shape[2], 1, 1)
            sim = sim.masked_fill(mask == 0, mask_fill_value(sim.dtype))

        attn = sim.softmax(dim=-1)
        out = einsum('s b h i j, s b h j d -> s b h i d', attn, v)
        out = rearrange(out, 's b h n d -> s b n (h d)', h=h)
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # s = batch size
        # b = number of rows
        # h = number of heads
        # n = number of columns
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                      (q, k, v))

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # masking
        # torch.Size([12, 300, 300])
        if mask is not None:
            mask = (mask.unsqueeze(1).repeat(1, sim.shape[1], 1, 1))
            sim = sim.masked_fill(mask == 0, mask_fill_value(sim.dtype))

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class Qformer(nn.Module):

    def __init__(self, dim, dim_head, inner_dim, query_num):
        super().__init__()

        self.heads = inner_dim // dim_head
        self.query_num = query_num
        self.scale = dim_head**-0.5
        self.q = nn.Parameter(torch.randn(query_num, inner_dim))
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.ff = PreNorm(inner_dim, Residual(FeedForward(inner_dim)))

    def forward(self, x, mask=None):
        x = rearrange(x, 's b n d -> s n b d')

        h = self.heads
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q = self.q.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1,
                                                    1)
        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s b h n d', h=h),
                      (q, k, v))
        sim = einsum('s b h i d, s b h j d -> s b h i j', q, k) * self.scale

        # masking
        if mask is not None:
            mask = rearrange(mask, 's i j -> s j i')
            mask = mask[:, 0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
                1, sim.shape[1], sim.shape[2], sim.shape[3], 1)
            sim = sim.masked_fill(mask == 0, mask_fill_value(sim.dtype))

        attn = sim.softmax(dim=-1)
        out = einsum('s b h i j, s b h j d -> s b h i d', attn, v)
        out = rearrange(out, 's b h n d -> s b n (h d)', h=h)

        out = self.ff(out)
        return out


class RowColTransformer(nn.Module):
    # dim = dim of each token
    # nfeats = number of features (columns)
    # depth = number of attention layers
    # heads = number of heads in multihead attention
    # dim_head = dim of each head
    def __init__(self,
                 dim,
                 nfeats,
                 depth,
                 heads,
                 dim_head,
                 attn_dropout,
                 ff_dropout,
                 style='col',
                 mask=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.style = style

        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(
                    nn.ModuleList([
                        PreNorm(
                            dim,
                            Residual(
                                RowColAttention(dim,
                                                heads=heads,
                                                dim_head=dim_head,
                                                dropout=attn_dropout))),
                        PreNorm(dim,
                                Residual(FeedForward(dim,
                                                     dropout=ff_dropout))),
                        PreNorm(
                            dim,
                            Residual(
                                RowColAttention(dim,
                                                heads=heads,
                                                dim_head=dim_head,
                                                dropout=attn_dropout))),
                        PreNorm(dim,
                                Residual(FeedForward(dim,
                                                     dropout=ff_dropout))),
                    ]))
            else:
                self.layers.append(
                    nn.ModuleList([
                        PreNorm(
                            dim * nfeats,
                            Residual(
                                Attention(dim * nfeats,
                                          heads=heads,
                                          dim_head=64,
                                          dropout=attn_dropout))),
                        PreNorm(
                            dim * nfeats,
                            Residual(
                                FeedForward(dim * nfeats,
                                            dropout=ff_dropout))),
                    ]))

    def forward(self, x, mask=None):
        _, _, n, _ = x.shape  # [bs, n_rows, n_cols, dim]
        row_mask = None
        col_mask = None
        if mask is not None:
            col_mask = einsum('b i j, b i k -> b j k', mask, mask)
            row_mask = einsum('b i j, b k j -> b i k', mask, mask)
        # print(col_mask.shape, row_mask.shape)
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x, mask=col_mask)
                x = ff1(x)
                x = rearrange(x, 's b n d -> s n b d')
                x = attn2(x, mask=row_mask)
                x = ff2(x)
                x = rearrange(x, 's n b d -> s b n d', n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 's b n d -> s 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 's 1 b (n d) -> s b n d', n=n)
        return x


# transformer
class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Residual(
                            Attention(dim,
                                      heads=heads,
                                      dim_head=dim_head,
                                      dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim,
                                                      dropout=ff_dropout))),
                ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


# mlp
class SimpleMLP(nn.Module):

    def __init__(self, dims):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(),
                                    nn.Linear(dims[1], dims[2]))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def get_flatten_table_emb(table_emb, mask):
    flatten_table_emb = torch.zeros(table_emb.size(0), table_emb.size(2),
                                    table_emb.size(3)).to(table_emb.device)
    row_num = torch.sum(mask, dim=1).int()
    for i in range(len(table_emb)):
        flatten_table_emb[i] = torch.mean(table_emb[i, :row_num[i, 0], :, :],
                                          dim=0)
    return flatten_table_emb


class TablegptEncoderModel(PreTrainedModel):
    main_input_name = "inputs_embeds"
    def __init__(self,pretrained_name_or_path: str, config:TableGPTEncoderConfig,*args, **kwargs):
        super().__init__(config)


        self.config = config

        self.num_cols = config.num_cols
        self.attentiontype = config.attentiontype
        self.pred_type = config.pred_type
        self.cont_dim = config.cont_dim
        self.pooling = config.pooling
        self.numeric_mlp = config.numeric_mlp
        self.ff_dropout = config.ff_dropout
        self.attn_dropout = config.attn_dropout
        self.dim_head = config.dim_head
        self.depth = config.depth
        self.heads = config.heads

        self.st = AutoModel.from_config(config.st_config)
        self.dim = self.st.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path,subfolder=config.subfolder)

        self.st.pooler = None

        # transformer
        self.transformer = RowColTransformer(dim=self.dim,
                                             nfeats=self.num_cols,
                                             depth=self.depth,
                                             heads=self.heads,
                                             dim_head=self.dim_head,
                                             attn_dropout=self.attn_dropout,
                                             ff_dropout=self.ff_dropout,
                                             style=self.attentiontype)
        
        self.col_specific_projection_head = SimpleMLP(
            [self.dim, self.dim, self.cont_dim])

        self.qformer = Qformer(dim=self.dim,
                               dim_head=128,
                               inner_dim=3584,
                               query_num=3)

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0]  #First element of model_output contains all token embeddings

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).to(dtype=token_embeddings.dtype)

        return torch.sum(token_embeddings * input_mask_expanded,
                         1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, input_ids, attention_mask, token_type_ids):
        bs, num_rows, num_cols, seq_len = input_ids.shape[0], input_ids.shape[
            1], input_ids.shape[2], input_ids.shape[3]
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(-1, seq_len)

        last_hidden_state = self.st(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        embeddings = embeddings.reshape(bs, num_rows, num_cols, -1)

        return embeddings

    @torch.inference_mode()
    def forward(
        self,
        input_ids:torch.Tensor,
        attention_mask:torch.Tensor,
        token_type_ids:torch.Tensor,
        table_mask:torch.Tensor,
    ):

        tab_emb = self.get_embeddings(input_ids, attention_mask,
                                      token_type_ids)

        if self.pooling == 'cls':
            # roll the table on dim 1 (row dim)
            tab_emb = torch.roll(tab_emb, 1, 1)
            # insert [cls] token at the first row
            tab_emb[:, 0, :, :] = self.cls

        cell_emb = self.transformer(tab_emb, mask=table_mask)

        col_emb = self.attn_pooling(cell_emb, table_mask)

        return col_emb

    def attn_pooling(self, cell_emb, table_mask):
        output = self.qformer(cell_emb, mask=table_mask)
        return output
    
    def get_embedded_table(self, table:pd.DataFrame , max_rows,max_cols):
        
        def process_table_df(table_df):
            numeric_columns = table_df.select_dtypes(include=["number"]).columns

            # fill missing values with mean
            table_df[numeric_columns] = table_df[numeric_columns].apply(
                lambda col: col.fillna(col.mean() if not col.isna().all() else 0)
            )
            if len(table_df) > max_rows:
                table_df = table_df.sample(n=max_rows)
                
            
            table_np = table_df.to_numpy().astype(str)
            
            return table_np

        def load_tokenized_table(anchor_table):
            anchor_table = process_table_df(anchor_table)
            anchor_row_num, num_cols = anchor_table.shape[0], anchor_table.shape[1]
            # anchor_row_num = anchor_table.shape[0]
            anchor_table = anchor_table.reshape(-1)
            max_length = 64
            
            tokenized_anchor_table = self.tokenizer(anchor_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')                

            tokenized_anchor_table = {k: v.reshape(anchor_row_num, num_cols, -1) for k, v in tokenized_anchor_table.items()}
            return tokenized_anchor_table


        # table_df

        df_col_count = table.shape[1]

        anchor_table = load_tokenized_table(table)
        num_cols = anchor_table['input_ids'].shape[1]
        anchor_table_row_num = anchor_table['input_ids'].shape[0]
        

        anchor_table_padded = {k: F.pad(v, (0, 0, 0, max_cols - v.shape[1], 0, max_rows - v.shape[0]), "constant", 1) for k, v in anchor_table.items()}
        anchor_table_mask = np.zeros((max_rows, max_cols))
        
        anchor_table_mask[:anchor_table_row_num, : num_cols] = 1
        ret = (
            anchor_table_padded['input_ids'],
            anchor_table_padded['attention_mask'],
            anchor_table_padded['token_type_ids'],
            torch.tensor(anchor_table_mask),
            df_col_count
        )
        return ret

    def get_encoder_output(self, tables:List[pd.DataFrame], max_rows:int,max_cols:int):
        
        # return table coount * seq length * embedding dim tensor?
        # and then merge to the input ids?
        
        table_embeds = []

        anchor_table_input_ids = []
        anchor_table_attention_mask = []
        anchor_table_token_type_ids = []
        anchor_table_mask = []
        column_count = []

        for table in tables:
            p, q, r, s, cnt = self.get_embedded_table(table,max_cols,max_rows)
            column_count.append(cnt)
            anchor_table_input_ids.append(p)
            anchor_table_attention_mask.append(q)
            anchor_table_token_type_ids.append(r)
            anchor_table_mask.append(s)
            
        anchor_table_input_ids = torch.stack(anchor_table_input_ids, dim=0).to(self.st.device)
        anchor_table_attention_mask = torch.stack(anchor_table_attention_mask, dim=0).to(self.st.device)
        anchor_table_token_type_ids = torch.stack(anchor_table_token_type_ids, dim=0).to(self.st.device)
        anchor_table_mask = torch.stack(anchor_table_mask, dim=0).to(self.st.device)

        
        table_embeds = self(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask)
        del anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask
      
        cat_table_embeds = []

        for j in range(len(column_count)):
            cat_table_embeds.append(table_embeds[j, :column_count[j]])
        
        return torch.cat(cat_table_embeds,dim=0)