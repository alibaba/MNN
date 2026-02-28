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
        model.rotary = Rotary(config)
        model.blocks = torch.nn.ModuleList([
            Decoder(block, i, config, model.rotary, config.model_map) for i, block in enumerate(model.blocks.children())
        ])
        model.lm = Lm(model.lm)

        if 'gemma' in model_type and hasattr(model.embed, 'embed_scale'):
            model.scale_emb = model.embed.embed_scale

        # Multi-modal parts
        if model.visual is not None:
            from utils.vision import Vision
            # model.visual = Vision.get_vision(model_type)(model.visual, model)
            model.visual = Vision.get_vision(model_type)(model.visual.float(), model).float()
        if hasattr(model, 'audio') and model.audio is not None:
            from utils.audio import Audio
            model.audio = Audio.get_audio(model.audio.config.model_type)(model.audio, model)
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
        if self.visual is not None and input_ids.numel() > 1:
            return self.visual.embed(input_ids)
        if self.audio is not None and input_ids.numel() > 1:
            return self.audio.embed(input_ids)
        return self.embed(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                logits_index: torch.Tensor = torch.tensor([-1], dtype=torch.int32),
                deepstack_embeds: torch.Tensor = None
                ):
        hidden_states = input_ids # llm forward without embedding

        if self.scale_emb is not None:
            hidden_states = hidden_states * self.scale_emb

        eagle_hidden_states = []
        rotary_pos_emb = self.rotary(position_ids)
        if self.args and self.args.test and rotary_pos_emb.dtype != hidden_states.dtype:
            rotary_pos_emb = rotary_pos_emb.type(hidden_states.dtype)

        for i in range(len(self.blocks)):
            # eagle3 hidden states
            if self.args and self.args.eagle_path and (i == len(self.blocks)-3 or i == len(self.blocks)//2 or i==2):
                eagle_hidden_states.append(hidden_states)

            # sliding or full attn mask
            if self.config.attention_type == 'mix':
                is_sliding = i in self.config.sliding_attn_layers
                layer_attention_mask = attention_mask[int(is_sliding)]
            else:
                layer_attention_mask = attention_mask

            hidden_states = self.blocks[i](hidden_states, rotary_pos_emb, layer_attention_mask)
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