from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_codet5p import CodeT5pModuleConfig

logger = logging.get_logger(__name__)

class TableGPTConfig(PretrainedConfig):
    model_type = 'tablegpt'
    is_composition = True

    def __init__(self,
                 encoder_config=None,
                 llm_config=None,
                 encoder_hidden_size=1024,
                 decoder_hidden_size=3584,
                 mlp_depth=1,
                 encoder_max_length=64,
                 placeholder_token="<TABLE_CONTNET>",
                 placeholder_token_id=-114,
                 **kwargs):

        super().__init__(**kwargs)

        self.encoder_config = CodeT5pModuleConfig(**encoder_config)
        self.llm_config = PretrainedConfig(**llm_config)

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.mlp_depth = mlp_depth
        self.encoder_max_length = encoder_max_length
        self.placeholder_token= placeholder_token
        self.placeholder_token_id = placeholder_token_id
