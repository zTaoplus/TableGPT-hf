from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_tablegpt_enc import TableGPTEncoderConfig

logger = logging.get_logger(__name__)


class TableGPTConfig(PretrainedConfig):
    model_type = 'tablegpt'
    is_composition = True

    def __init__(
            self,
            encoder_config=None,
            llm_config=None,
            projector_config=None,
            **kwargs):
        
        super().__init__(**kwargs)
        # TODO: should add some validators for config load
        if llm_config is None:
            encoder_config = {}
            logger.info('llm_config is None. Initializing the Qwen2Config config with default values (`Qwen2Config`).')
        
        if encoder_config is None:
            encoder_config = {}
            logger.info('encoder_config is None. Initializing the TableGPTEncoderConfig with default values.')

        # INIT encoder config, llm config
        self.encoder_config = TableGPTEncoderConfig(**encoder_config)
        self.llm_config = Qwen2Config(**llm_config)
        self.projector_config = PretrainedConfig(**projector_config)