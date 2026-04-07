import torch
import importlib
from packaging.version import Version
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoModelForCausalLM
from typing import Optional, List

from utils.config import LlmConfig
from utils.tokenizer import LlmTokenizer
from utils.model_mapper import ModelMapper
from utils.transformers import Embedding, Rotary, Decoder, Lm

class LlmModel(PreTrainedModel):
    config_class = LlmConfig

    def __init__(self, config, args=None):
        super().__init__(config)
        self.config = config
        self.args = args
        self.tokenizer = None
        self.model = None
        self.visual = None
        self.audio = None
        self.talker = None
        self.mtp = None
        self.scale_emb = None

    def _init_weights(self, module):
        pass

    def get_config(self):
        llm_config = {}
        models = ['visual', 'audio', 'talker']
        for m in models:
            if hasattr(self, m) and getattr(self, m) is not None:
                m_config = getattr(self, m).get_config()
                llm_config.update(m_config)
        return llm_config

    @staticmethod
    def get_model_class(model_type: str):
        # Same as in LlmExporter
        MODEL_CLASS_MAPPING = {
            'qwen3_5': 'Qwen3_5ForConditionalGeneration',
            'qwen3_5_moe': 'Qwen3_5MoeForConditionalGeneration',
            'qwen3_vl': 'Qwen3VLForConditionalGeneration',
            'qwen3_vl_moe': 'Qwen3VLMoeForConditionalGeneration',
            'qwen2_5_omni': 'Qwen2_5OmniForConditionalGeneration',
            'qwen2_5_vl': 'Qwen2_5_VLForConditionalGeneration',
            'qwen2_vl': 'Qwen2VLForConditionalGeneration',
            'qwen2_audio': 'Qwen2AudioForConditionalGeneration',
            'smolvlm': 'AutoModelForImageTextToText',
            'idefics3': 'AutoModelForVision2Seq',
            'funaudiochat': 'AutoModelForSeq2SeqLM',
            'glm_ocr': 'GlmOcrForConditionalGeneration',
            'lfm2_vl': 'Lfm2VlForConditionalGeneration',
            'gemma4': 'Gemma4ForConditionalGeneration',
        }
        if model_type is None or model_type not in MODEL_CLASS_MAPPING:
            return AutoModelForCausalLM
        class_name = MODEL_CLASS_MAPPING[model_type]
        try:
            module = importlib.import_module('transformers')
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            return AutoModelForCausalLM

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, args=None, **kwargs):
        config = LlmConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model_type = config.model_type
        model_class = cls.get_model_class(model_type)

        load_kwargs = {'trust_remote_code': True}
        if Version(importlib.metadata.version("transformers")) >= Version("4.56.0"):
            load_kwargs['dtype'] = 'auto'
        else:
            load_kwargs['torch_dtype'] = 'auto'

        if model_type == 'internvl_chat':
            load_kwargs['use_flash_attn'] = False

        # Check if skip_weight mode is enabled (load structure only, no weights)
        skip_weight = args is not None and hasattr(args, 'skip_weight') and args.skip_weight

        if skip_weight:
            # Load model skeleton without weights using accelerate
            from accelerate import init_empty_weights
            with init_empty_weights():
                original_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
                # Try different methods to create model from config (some models don't have from_config)
                if hasattr(model_class, 'from_config'):
                    original_model = model_class.from_config(original_config, trust_remote_code=True)
                elif hasattr(model_class, '_from_config'):
                    original_model = model_class._from_config(original_config)
                else:
                    original_model = AutoModelForCausalLM.from_config(original_config, trust_remote_code=True)
                original_model.to_empty(device="cpu")
        elif model_type == 'lfm2_audio':
            # LFM2-Audio uses liquid_audio package, not standard HF class
            from pathlib import Path
            from liquid_audio import LFM2AudioModel
            original_model = LFM2AudioModel.from_pretrained(
                Path(pretrained_model_name_or_path), device='cpu', dtype=torch.bfloat16
            )
            # Force sdpa attention on CPU (flash_attention_2 requires GPU)
            original_model.lfm.set_attn_implementation('sdpa')
        else:
            # Normal loading with weights
            try:
                original_model = model_class.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
            except Exception:
                original_model = AutoModel.from_pretrained(pretrained_model_name_or_path, **load_kwargs)

        # print(f"Loading model type: {model_type}\n{original_model}")

        # LoRA
        if args.lora_path is not None and not args.lora_split:
            from peft import PeftModel
            adapter = PeftModel.from_pretrained(original_model, model_id=args.lora_path)
            original_model = adapter.merge_and_unload(progressbar=True)

        original_model = original_model.eval()
        model = cls(config, args)

        ModelMapper.do_map(model, original_model, config.model_map['model'])

        model.tokenizer = LlmTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            model_type=model_type
        )

        # Rebuild modules
        if model.lm is None:
            out_features, in_features = model.embed.weight.shape
            model.lm = torch.nn.Linear(in_features, out_features, bias=False)
            model.lm.weight = model.embed.weight
        elif not isinstance(model.lm, torch.nn.Linear):
            weight = model.lm.weight
            out_features, in_features = weight.shape
            model.lm = torch.nn.Linear(in_features, out_features, bias=False)
            model.lm.weight = weight

        model.embed = Embedding(model.embed, config)

        # gemma4: dual rotary for sliding vs full attention layers
        if model_type == 'gemma4' and hasattr(config, 'rope_parameters') and config.rope_parameters is not None:
            rp = config.rope_parameters
            origin_config = config.origin_config
            text_config = origin_config.text_config if hasattr(origin_config, 'text_config') else origin_config
            # Sliding attention rotary
            sliding_rp = rp.get('sliding_attention', {})
            sliding_config = type('Config', (), {
                'rope_theta': sliding_rp.get('rope_theta', 10000.0),
                'rope_ratio': None,
                'head_dim': text_config.head_dim,
                'model_type': 'gemma4',
                'rope_parameters': None,
                'rope_scaling': None,
                'max_position_embeddings': config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 131072,
            })()
            model.rotary_sliding = Rotary(sliding_config)
            # Full attention rotary
            full_rp = rp.get('full_attention', {})
            global_head_dim = getattr(text_config, 'global_head_dim', text_config.head_dim)
            partial_factor = full_rp.get('partial_rotary_factor', 1.0)
            full_config = type('Config', (), {
                'rope_theta': full_rp.get('rope_theta', 1000000.0),
                'rope_ratio': None,
                'head_dim': global_head_dim,
                'model_type': 'gemma4',
                'rope_parameters': None,
                'rope_scaling': None,
                'max_position_embeddings': config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 131072,
            })()
            model.rotary_full = Rotary(full_config)
            # Adjust rotary_dim for partial rotary factor
            if partial_factor < 1.0:
                rotary_dim = int(global_head_dim * partial_factor)
                model.rotary_full.rotary_dim = rotary_dim
                model.rotary_full.theta = 1.0 / (full_rp.get('rope_theta', 1000000.0) ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / global_head_dim))
            model.rotary = model.rotary_sliding  # default rotary for config reference
        else:
            model.rotary = Rotary(config)
            model.rotary_sliding = None
            model.rotary_full = None

        model.blocks = torch.nn.ModuleList([
            Decoder(block, i, config, model.rotary, config.model_map) for i, block in enumerate(model.blocks.children())
        ])
        # Check for final_logit_softcapping (gemma4, gemma2)
        origin_config = config.origin_config
        text_config = origin_config.text_config if hasattr(origin_config, 'text_config') else origin_config
        final_logit_softcapping = getattr(text_config, 'final_logit_softcapping', None)
        model.lm = Lm(model.lm, final_logit_softcapping=final_logit_softcapping)

        if 'gemma' in model_type and hasattr(model.embed, 'embed_scale'):
            model.scale_emb = model.embed.embed_scale

        # Multi-modal parts
        if model.visual is not None:
            from utils.vision import Vision
            vision_cls = Vision.get_vision(model_type)
            if vision_cls is not None:
                model.visual = vision_cls(model.visual.float(), model).float()
            else:
                model.visual = None
        if hasattr(model, 'audio') and model.audio is not None:
            from utils.audio import Audio
            audio_type = model.audio.config.model_type if hasattr(model.audio, 'config') else model_type
            audio_cls = Audio.get_audio(audio_type)
            if audio_cls is not None:
                model.audio = audio_cls(model.audio, model)
            else:
                model.audio = None
        if hasattr(model, 'talker') and model.talker is not None:
            from utils.talker import Talker
            model.talker = Talker.get_talker(model_type)(model.talker, model.token2wav, model)
        if model_type == 'poi_qwen2_mtp':
            model.mtp = [model.mtp1, model.mtp2]
        if model.mtp is not None:
            from utils.mtp import Mtp
            model.mtp = Mtp.get_mtp(model_type)(model.mtp, model)

        return model

    def embedding(self, input_ids):
        # Store original input_ids for PLE (gemma4)
        self._last_input_ids = input_ids
        if self.visual is not None and input_ids.numel() > 1:
            result = self.visual.embed(input_ids)
            # Also apply audio embeddings if audio module has pending embeddings
            if self.audio is not None and self.audio.audio_embeds is not None:
                audio_pad_id = self.audio.config.audio_token_id
                audio_mask = (input_ids == audio_pad_id).squeeze()
                if audio_mask.any():
                    embed_scale = self.config.hidden_size ** 0.5
                    result[audio_mask] = self.audio.audio_embeds.to(result.dtype) / embed_scale
                    self.audio.audio_embeds = None
            return result
        if self.audio is not None and input_ids.numel() > 1:
            return self.audio.embed(input_ids)
        return self.embed(input_ids)

    def compute_ple_from_embeddings(self, hidden_states, ple_embeddings):
        """Compute PLE from pre-looked-up embeddings (for export/C++ mode).
        ple_embeddings: [1, seq_len, num_layers * ple_dim] — already scaled by embed_scale.
        """
        num_layers = self.config.num_hidden_layers
        ple_dim = ple_embeddings.shape[-1] // num_layers
        per_layer_inputs = ple_embeddings.reshape(*ple_embeddings.shape[:2], num_layers, ple_dim)
        # Project from main embeddings
        hs_for_proj = hidden_states.view(1, -1, self.config.hidden_size)
        per_layer_proj = self.per_layer_model_projection(hs_for_proj) * (self.config.hidden_size ** -0.5)
        per_layer_proj = per_layer_proj.reshape(*hs_for_proj.shape[:-1], num_layers, ple_dim)
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)
        return (per_layer_proj + per_layer_inputs) * (2.0 ** -0.5)

    def compute_ple(self, hidden_states, input_ids=None):
        """Compute Per-Layer Embeddings for gemma4."""
        if not (hasattr(self, 'embed_tokens_per_layer') and self.embed_tokens_per_layer is not None):
            return None
        if input_ids is None:
            input_ids = getattr(self, '_last_input_ids', None)
        if input_ids is None:
            return None
        # Replace multimodal token IDs with pad_token_id for PLE lookup
        # (matches HF behavior: llm_input_ids[multimodal_mask] = pad_token_id)
        ple_ids = input_ids.clone()
        oc = getattr(self.config, 'origin_config', self.config)
        tc = getattr(oc, 'text_config', oc)
        pad_token_id = getattr(tc, 'pad_token_id', 0) or 0
        for attr in ['image_token_id', 'audio_token_id', 'video_token_id']:
            token_id = getattr(oc, attr, None)
            if isinstance(token_id, int):
                ple_ids[ple_ids == token_id] = pad_token_id
        num_layers = self.config.num_hidden_layers
        ple_dim = self.embed_tokens_per_layer.embedding_dim // num_layers
        # 1. Lookup per-layer embeddings (ScaledWordEmbedding applies scale internally)
        per_layer_inputs = self.embed_tokens_per_layer(ple_ids)
        per_layer_inputs = per_layer_inputs.reshape(*input_ids.shape, num_layers, ple_dim)
        # 2. Project from main embeddings
        hs_for_proj = hidden_states.view(1, -1, self.config.hidden_size)
        per_layer_proj = self.per_layer_model_projection(hs_for_proj) * (self.config.hidden_size ** -0.5)
        per_layer_proj = per_layer_proj.reshape(*hs_for_proj.shape[:-1], num_layers, ple_dim)
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)
        # 3. Combine
        return (per_layer_proj + per_layer_inputs) * (2.0 ** -0.5)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                logits_index: torch.Tensor = torch.tensor([-1], dtype=torch.int32),
                deepstack_embeds: torch.Tensor = None,
                ple_embeddings: torch.Tensor = None
                ):
        hidden_states = input_ids # llm forward without embedding

        # gemma4: compute PLE
        # For ONNX export: scale_emb is NOT in forward(), it's applied externally.
        # For Python test: scale_emb is applied in forward() above (selective scaling).
        # Both cases: hidden_states at this point has text positions scaled.
        per_layer_inputs = None
        if hasattr(self, 'per_layer_model_projection') and self.per_layer_model_projection is not None:
            ple_proj_input = hidden_states
            # For multimodal inputs, PLE projection uses pad embeddings at multimodal positions
            # (matches HF: llm_inputs_embeds = where(multimodal_mask, pad_embedding, inputs_embeds))
            ids = getattr(self, '_last_input_ids', None)
            if ids is not None:
                oc = getattr(self.config, 'origin_config', self.config)
                mm_mask = torch.zeros_like(ids, dtype=torch.bool)
                for attr in ['image_token_id', 'audio_token_id', 'video_token_id']:
                    token_id = getattr(oc, attr, None)
                    if isinstance(token_id, int):
                        mm_mask = mm_mask | (ids == token_id)
                if mm_mask.any():
                    tc = getattr(oc, 'text_config', oc)
                    pad_id = getattr(tc, 'pad_token_id', 0) or 0
                    pad_emb = self.embed(torch.tensor([[pad_id]]))  # [1, 1, hidden_size]
                    ple_proj_input = hidden_states.clone()
                    mm_flat = mm_mask.squeeze()
                    ple_proj_input[mm_flat] = pad_emb.squeeze()
            if ple_embeddings is None and hasattr(self, 'embed_tokens_per_layer') and self.embed_tokens_per_layer is not None:
                if ids is not None:
                    per_layer_inputs = self.compute_ple(ple_proj_input, ids)
            elif ple_embeddings is not None:
                per_layer_inputs = self.compute_ple_from_embeddings(ple_proj_input, ple_embeddings)

        # scale_emb: multiply ALL positions uniformly (text + vision).
        # Vision positions are pre-divided by scale_emb, so after this multiply they restore.
        if self.scale_emb is not None:
            hidden_states = hidden_states * self.scale_emb

        eagle_hidden_states = []
        rotary_pos_emb = self.rotary(position_ids)
        if self.args and self.args.test and rotary_pos_emb.dtype != hidden_states.dtype:
            rotary_pos_emb = rotary_pos_emb.type(hidden_states.dtype)

        # gemma4: compute separate rotary for full attention layers
        rotary_pos_emb_full = None
        if self.rotary_full is not None:
            rotary_pos_emb_full = self.rotary_full(position_ids)
            if self.args and self.args.test and rotary_pos_emb_full.dtype != hidden_states.dtype:
                rotary_pos_emb_full = rotary_pos_emb_full.type(hidden_states.dtype)

        # KV sharing cache (gemma4: layers 15-34 share KV with layers 13/14)
        shared_kv_cache = {}
        for i in range(len(self.blocks)):
            # Set shared KV cache reference on attention
            if hasattr(self.blocks[i].self_attn, 'is_kv_shared_layer'):
                self.blocks[i].self_attn._shared_kv_cache = shared_kv_cache

            # eagle3 hidden states
            if self.args and self.args.eagle_path and (i == len(self.blocks)-3 or i == len(self.blocks)//2 or i==2):
                eagle_hidden_states.append(hidden_states)

            # sliding or full attn mask
            if self.config.attention_type == 'mix':
                is_sliding = i in self.config.sliding_attn_layers
                layer_attention_mask = attention_mask[int(is_sliding)]
            else:
                layer_attention_mask = attention_mask

            # gemma4: use different rotary for full vs sliding layers
            if rotary_pos_emb_full is not None and not (hasattr(self.config, 'sliding_attn_layers') and i in self.config.sliding_attn_layers):
                layer_rotary = rotary_pos_emb_full
            else:
                layer_rotary = rotary_pos_emb

            # Set per-layer input for PLE
            if per_layer_inputs is not None:
                self.blocks[i]._per_layer_input = per_layer_inputs[:, :, i, :]
            hidden_states = self.blocks[i](hidden_states, layer_rotary, layer_attention_mask)
            if deepstack_embeds is not None and i in range(deepstack_embeds.shape[0]):
                hidden_states += deepstack_embeds[i]

        talker_embeds = None
        if hasattr(self, 'talker') and self.talker is not None:
            talker_embeds = self.final_layernorm(hidden_states) + input_ids.permute([1, 0, 2])
            self.talker.add_talker_embeds(talker_embeds)

        final_layernorm = hidden_states
        logits_index_long = logits_index.to(torch.int64)
        if self.mtp is None:
            hidden_states = hidden_states[:, logits_index_long:, :]
            hidden_states = self.final_layernorm(hidden_states)
            # default: set hidden_state before lm_head as output node
            final_layernorm = hidden_states
        else:
            # final_layernorm need compute all logists
            if self.config.model_type == 'mimo':
                final_layernorm = hidden_states # mimo
            hidden_states = self.final_layernorm(hidden_states)
            if self.config.model_type == 'poi_qwen2_mtp':
                final_layernorm = hidden_states # poi
            hidden_states = hidden_states[:, logits_index_long:, :]
        logits = self.lm(hidden_states)

        if self.args and self.args.eagle_path is not None:
            final_layernorm = torch.cat(eagle_hidden_states, dim=-1)

        return logits, final_layernorm, talker_embeds

    def get_attention_mask(self, seq_len: int, new_tokens: int = 0):
        if self.config.model_type == 'chatglm':
            return self.chatglm_attention_mask()
        if self.config.attention_type == 'full':
            return self.full_attention_mask(seq_len, new_tokens)
        elif self.config.attention_type == 'sliding':
            return self.sliding_attention_mask(self.config.sliding_window, seq_len, new_tokens)
        elif self.config.attention_type == 'mix':
            full_mask = self.full_attention_mask(seq_len, new_tokens)
            sliding_mask = self.sliding_attention_mask(self.config.sliding_window, seq_len, new_tokens)
            return torch.stack([full_mask, sliding_mask], dim=0)
        return None

    def full_attention_mask(self, seq_len, new_tokens):
        if new_tokens:
            return torch.zeros([1, 1, 1, seq_len], dtype=torch.float32)
        return (1 - torch.tril(torch.ones([1, 1, seq_len, seq_len]))) * torch.finfo(torch.float32).min

    def sliding_attention_mask(self, sliding_window, seq_len, new_tokens):
        if new_tokens:
            sliding_mask = torch.zeros([1, 1, 1, seq_len], dtype=torch.float32)
            num_tokens_to_mask = seq_len - sliding_window
            if num_tokens_to_mask > 0:
                sliding_mask[..., :num_tokens_to_mask] = torch.finfo(torch.float32).min
            return sliding_mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        query_indices = torch.arange(seq_len).view(-1, 1)
        key_indices = torch.arange(seq_len).view(1, -1)
        window_mask = (key_indices > query_indices - sliding_window)
        final_mask_bool = causal_mask & window_mask
        sliding_mask = torch.where(final_mask_bool, 0.0, torch.finfo(torch.float32).min)
        return sliding_mask.view(1, 1, seq_len, seq_len)

    def get_position_ids(self, seq_len, new_tokens=0, input_ids=None):
        if self.visual is not None and hasattr(self.visual, 'get_position_ids'):
            return self.visual.get_position_ids(input_ids, seq_len, new_tokens)
        if self.config.model_type == 'chatglm':
            return self.chatglm_position_ids(seq_len, new_tokens)
        if new_tokens:
            position_ids = torch.tensor([seq_len - 1], dtype=torch.int)
        else:
            position_ids = torch.arange(seq_len, dtype=torch.int)

        if self.rotary.is_mrope:
            position_ids = torch.stack([position_ids] * 3)
        else:
            position_ids = position_ids.unsqueeze(0)
        return position_ids

    def chatglm_attention_mask(self, seq_len, is_decode):
        if is_decode:
            return torch.zeros([1]).bool().reshape([1, 1, 1, 1])
        attention_mask = torch.zeros([seq_len, seq_len], dtype=torch.bool)
        for i in range(seq_len - 1):
            attention_mask[i][-1] = True
        return attention_mask.reshape([1, 1, seq_len, seq_len])

    def chatglm_position_ids(self, seq_len, new_tokens):
        if new_tokens:
            return torch.tensor([seq_len - 2, new_tokens + 1]).reshape([1, 2, 1])
        position_ids_0 = torch.arange(seq_len, dtype=torch.int)
        position_ids_1 = torch.zeros(seq_len, dtype=torch.int)
        position_ids_0[-1] = position_ids_0[-2]
        position_ids_1[-1] = 1
        return torch.stack([position_ids_0, position_ids_1]).view(1, 2, -1)

class EmbeddingModel(LlmModel):
    def __init__(self, config, args=None):
        super().__init__(config, args)
        self.is_reranker = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, args=None, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        model_type = config.model_type
        if model_type == 'qwen3':
            model = super(EmbeddingModel, cls).from_pretrained(pretrained_model_name_or_path, args=args).float().eval()
            return model
        # gte, bge
        config._attn_implementation = 'eager'
        model = cls(config, args)
        if model_type == 'new' and 'NewForSequenceClassification' in config.architectures:
            model.is_reranker = True
            from transformers import AutoModelForSequenceClassification
            origin_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config, trust_remote_code=True).float().eval()
            model.classifier = origin_model.classifier
            origin_model = origin_model.new
        else:
            origin_model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config, trust_remote_code=True).float().eval()

        transformer = origin_model.encoder
        model.lm = origin_model.pooler
        model.embed = origin_model.embeddings
        model.word_embeddings = model.embed.word_embeddings
        model.token_type_embeddings = model.embed.token_type_embeddings.weight.data[0]
        model.embedding_layernorm = model.embed.LayerNorm
        if hasattr(model.embed, 'position_embeddings'):
            model.position_embeddings = model.embed.position_embeddings
        model.hidden_size = model.word_embeddings.weight.shape[-1]
        model.blocks = transformer.layer
        # some wrapper
        model.num_hidden_layers = len(model.blocks)
        return model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        if self.config.model_type == 'bert':
            return self.bge_forward(inputs_embeds, attention_mask, position_ids)
        if self.config.model_type == 'new':
            return self.gte_forward(inputs_embeds, attention_mask, position_ids)
        if self.config.model_type == 'qwen3':
            return self.qwen3_forward(inputs_embeds, attention_mask, position_ids)
        raise RuntimeError(f'Not support embedding model: {self.config.model_type}!')

    def word_embed(self, input_ids):
        if hasattr(self, 'word_embeddings'):
            return self.word_embeddings(input_ids.view(1, -1))
        return self.embed(input_ids.view(1, -1))

    def bge_forward(self, inputs_embeds, attention_mask, position_ids):
        inputs_embeds = inputs_embeds.reshape(1, -1, self.config.hidden_size)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings + self.token_type_embeddings
        hidden_states = self.embedding_layernorm(embeddings)
        for i in range(self.config.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states, attention_mask)[0]
        sentence_embeddings = hidden_states[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def gte_reranker_forward(self, inputs_embeds, attention_mask, position_ids):
        freqs = position_ids.float().reshape(-1, 1) * self.embed.rotary_emb.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        rope_embeds = torch.stack([emb.cos(), emb.sin()]).unsqueeze(-2).unsqueeze(1)
        hidden_states = self.embedding_layernorm(inputs_embeds + self.token_type_embeddings)
        for i in range(self.config.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states, attention_mask, rope_embeds)[0]
        pooled_output = self.lm(hidden_states)
        logits = self.classifier(pooled_output)
        return logits

    def gte_embedding_forward(self, inputs_embeds, attention_mask, position_ids):
        inputs_embeds = inputs_embeds.reshape(1, -1, self.config.hidden_size)
        freqs = position_ids.float().reshape(-1, 1) * self.embed.rotary_emb.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        rope_embeds = torch.stack([emb.cos(), emb.sin()]).unsqueeze(-2).unsqueeze(1)
        attention_bias = 1 - attention_mask.float()
        hidden_states = self.embedding_layernorm(inputs_embeds + self.token_type_embeddings)
        for i in range(self.config.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states, attention_bias, rope_embeds)[0]
        sentence_embeddings = hidden_states[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def gte_forward(self, inputs_embeds, attention_mask, position_ids):
        if self.is_reranker:
            return self.gte_reranker_forward(inputs_embeds, attention_mask, position_ids)
        return self.gte_embedding_forward(inputs_embeds, attention_mask, position_ids)

    def qwen3_forward(self, inputs_embeds, attention_mask, position_ids):
        hidden_states = inputs_embeds
        rotary_pos_emb = self.rotary(position_ids)
        for i in range(len(self.blocks)):
            hidden_states = self.blocks[i](hidden_states, rotary_pos_emb, attention_mask)
        last_hidden_states = hidden_states[:, -1, :]
        last_hidden_states = self.final_layernorm(last_hidden_states)
        return last_hidden_states

    def get_position_ids(self, seq_len) -> torch.Tensor:
        return torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    def get_attention_mask(self, seq_len) -> torch.Tensor:
        if self.config.model_type == 'qwen3':
            return super().get_attention_mask(seq_len, 0)
        return torch.ones([1, 1, seq_len, seq_len], dtype=torch.float)