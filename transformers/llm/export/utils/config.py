from transformers import PretrainedConfig, AutoConfig
from utils.model_mapper import ModelMapper
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict

# model config

class LlmConfig(PretrainedConfig):
    model_type = "llm_config"

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.pop("hidden_size", 0)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 0)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 0)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", self.num_attention_heads)
        self.head_dim = kwargs.pop("head_dim", self.hidden_size // self.num_attention_heads if self.num_attention_heads > 0 else 0)
        self.rope_theta = kwargs.pop("rope_theta", 10000.0)
        self.rope_ratio = kwargs.pop("rope_ratio", 1.0)
        self.sliding_window = kwargs.pop("sliding_window", 0)
        self.sliding_window = self.sliding_window if self.sliding_window is not None else 0
        self.layer_types = kwargs.pop("layer_types", [])
        self.attention_type = kwargs.pop("attention_type", 'full')
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        self.model_map = kwargs.pop("model_map", {})
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, **kwargs)

        model_type, model_map = ModelMapper().get_map(config)
        llm_config_kwargs = {
            'origin_config': config,
            'model_type': model_type,
            'model_map': model_map
        }
        llm_config = cls(**llm_config_kwargs)
        # rename attribute for different models
        ModelMapper.do_map(llm_config, config, model_map['config'])

        # Post-processing and setting defaults
        if llm_config.num_key_value_heads is None:
            llm_config.num_key_value_heads = llm_config.num_attention_heads

        if llm_config.rope_theta is None:
            llm_config.rope_theta = 10000.0

        if llm_config.rope_ratio is None:
            llm_config.rope_ratio = 1.0

        if llm_config.head_dim is None and llm_config.hidden_size > 0 and llm_config.num_attention_heads > 0:
            if isinstance(llm_config.num_attention_heads, list):
                llm_config.head_dim = [llm_config.hidden_size // atten_head for atten_head in llm_config.num_attention_heads]
            else:
                llm_config.head_dim = llm_config.hidden_size // llm_config.num_attention_heads

        # Determine attention type
        sliding_attn_layers = []
        if hasattr(llm_config, 'layer_types') and llm_config.layer_types:
            for i in range(len(llm_config.layer_types)):
                if llm_config.layer_types[i] == 'sliding_attention':
                    sliding_attn_layers.append(i)

        if llm_config.num_hidden_layers > 0 and len(sliding_attn_layers) >= llm_config.num_hidden_layers:
            llm_config.attention_type = 'sliding'
        elif len(sliding_attn_layers) > 0:
            llm_config.attention_type = 'mix'
            llm_config.sliding_attn_layers = sliding_attn_layers
        else:
            llm_config.attention_type = 'full'

        return llm_config

# export config

@dataclass
class VisionExportConfig:
    """Configuration for vision-related capabilities."""
    image_mean: Optional[List[float]] = field(default_factory=list)
    image_norm: Optional[List[float]] = field(default_factory=list)
    image_size: Optional[Union[int, List[int]]] = None
    image_size_unit: Optional[int] = None
    vision_start: Optional[int] = None
    vision_end: Optional[int] = None
    image_pad: Optional[int] = None
    num_grid_per_side: Optional[int] = None
    has_deepstack: bool = False
    image_max_size: Optional[int] = None
    global_image: Optional[int] = None
    vision_id_start_id: Optional[int] = None
    vision_id_end_id: Optional[int] = None
    vision_slice_start_id: Optional[int] = None
    vision_slice_end_id: Optional[int] = None

@dataclass
class LLMExportConfig:
    """Top-level container for all export configurations."""
    is_audio: bool = False
    is_visual: bool = False
    has_talker: bool = False
    attention_mask: str = 'float'
    attention_type: str = 'full'
    sliding_window: int = 0
    tie_embeddings: Optional[List[Union[int]]] = field(default_factory=list)
    jinja: Dict[str, Any] = field(default_factory=dict)
    vision: Optional[VisionExportConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration to a dictionary for JSON serialization."""
        nested_dict = asdict(self)
        vision_data = nested_dict.pop('vision', None)

        if vision_data:
            nested_dict.update(vision_data)

        final_dict = {key: value for key, value in nested_dict.items() if value is not None}
        return final_dict