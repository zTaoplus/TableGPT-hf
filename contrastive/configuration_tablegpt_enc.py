from transformers import BertConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class TableGPTEncoderConfig(PretrainedConfig):
    model_type = 'tablegpt'
    is_composition = True

    def __init__(
        self,
        st_config=None,
        num_cols=None,
        depth=None,
        heads=None,
        attn_dropout=0.1,
        ff_dropout =0.1,
        attentiontype="colrow",
        pred_type="contrastive",
        dim_head=64,
        pooling="mean",
        col_name=False,
        numeric_mlp=False,
        max_rows=50,
        max_cols=100,
        insert_embs_token="<insert_embs>",
        insert_embs_token_id=-114,
        insert_seq_token="<insert_sep>",
        **kwargs
    ):
        
        # TODO: should add some validators for config load
        super().__init__(**kwargs)

        if st_config is None:
            st_config = {}
            logger.info('st_config is None. Initializing the BertConfig config with default values (`BertConfig`).')


        self.st_config = BertConfig(**st_config)

        self.num_cols=num_cols
        self.depth=depth
        self.heads=heads
        self.attn_dropout=attn_dropout
        self.ff_dropout =ff_dropout
        self.attentiontype=attentiontype
        self.pred_type=pred_type
        self.dim_head=dim_head
        self.pooling=pooling
        self.col_name=col_name
        self.numeric_mlp=numeric_mlp
        self.max_rows=max_rows
        self.max_cols=max_cols
        self.insert_embs_token=insert_embs_token
        self.insert_embs_token_id=insert_embs_token_id
        self.insert_seq_token=insert_seq_token
