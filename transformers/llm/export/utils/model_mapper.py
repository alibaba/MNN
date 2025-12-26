import copy
from transformers import __version__ as TRANSFORMERS_VERSION

class ModelMapper:
    def __init__(self):
        self.attrs = []
        self.mapper = dict()
        self.init_models()

    def get_map(self, config):
        model_type = config.model_type
        if model_type == 'chatglm':
            if hasattr(config, 'vocab_size') and config.vocab_size == 130528:
                model_type = 'chatglm'
            else:
                model_type = 'chatglm2'
        if model_type in self.mapper:
            return model_type, self.mapper[model_type]
        return model_type, self.default_map

    def regist(self, model_type, model_map):
        assert('config' in model_map and
               'decoder' in model_map and
               'attention' in model_map)
        self.mapper[model_type] = model_map

    def init_models(self):
        self.init_default_map()
        for method_name in dir(self):
            if callable(getattr(self, method_name)) and method_name.startswith("regist_"):
                method = getattr(self, method_name)
                method()

    def regist_llama(self):
        llama_map = self.default_map
        self.regist('llama', llama_map)
        self.regist('qwen2', llama_map)
        self.regist('internlm', llama_map)
        self.regist('mobilellm', llama_map)
        # baichuan
        baichuan_map = copy.deepcopy(self.default_map)
        baichuan_map[self.attention_key] = {
            'qkv_proj': 'W_pack',
            'o_proj': 'o_proj'
        }
        self.regist('baichuan', baichuan_map)
    def regist_deepseek_vl(self):
        deepseek_vlmap = {
            'config': {
                'hidden_size': 'language_config.hidden_size',
                'num_attention_heads': 'language_config.num_attention_heads',
                'num_hidden_layers': 'language_config.num_hidden_layers',
                'rope_theta': 'language_config.rope_theta',
                'head_dim': 'language_config.head_dim',
                'num_key_value_heads': 'language_config.num_key_value_heads',
            },
            'model': {
                'lm': 'language_model.lm_head',
                'embed': 'language_model.model.embed_tokens',
                'blocks': 'language_model.model.layers',
                'final_layernorm': 'language_model.model.norm',
                'visual': 'vision_model'
            },
            'decoder': {
                'self_attn': 'self_attn',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'o_proj'
            }
        }
        self.regist('deepseek-vl', deepseek_vlmap)

    def regist_qwen_omni(self):
        omni_map = {
            'config': {
                'hidden_size': 'thinker_config.text_config.hidden_size',
                'head_dim': 'thinker_config.text_config.head_dim',
                'num_attention_heads': 'thinker_config.text_config.num_attention_heads',
                'num_hidden_layers': 'thinker_config.text_config.num_hidden_layers',
                'num_key_value_heads': 'thinker_config.text_config.num_key_value_heads',
                'rope_theta': 'thinker_config.text_config.rope_theta',
                'rope_scaling': 'thinker_config.text_config.rope_scaling'
            },
            'model': {
                'lm': 'thinker.lm_head',
                'embed': 'thinker.model.embed_tokens',
                'blocks': 'thinker.model.layers',
                'final_layernorm': 'thinker.model.norm',
                'visual': 'thinker.visual',
                'audio': 'thinker.audio_tower',
                'talker': 'talker',
                'token2wav': 'token2wav'
            },
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('qwen2_5_omni', omni_map)

    def regist_qwen(self):
        qwen_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'rope_theta': 'rotary_emb_base',
            },
            'model': {
                'lm': 'lm_head',
                'embed': 'transformer.wte',
                'blocks': 'transformer.h',
                'final_layernorm': 'transformer.ln_f',
                'visual': 'transformer.visual'
            },
            'decoder': {
                'self_attn': 'attn',
                'mlp': 'mlp',
                'input_layernorm': 'ln_1',
                'post_attention_layernorm': 'ln_2'
            },
            'attention': {
                'qkv_proj': 'c_attn',
                'o_proj': 'c_proj'
            }
        }
        self.regist('qwen', qwen_map)

    def regist_qwen3(self):
        qwen3_attention = {
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj',
            'q_norm': 'q_norm',
            'k_norm': 'k_norm'
        }
        qwen3_map = {
            'config': self.default_config,
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': qwen3_attention
        }
        self.regist('qwen3', qwen3_map)

    def regist_llama4_text(self):
        llama4_text_attention = {
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj',
            'qk_norm': 'qk_norm'
        }
        llama4_text_decoder = copy.deepcopy(self.default_decoder)
        llama4_text_decoder['mlp'] = 'feed_forward'
        llama4_text_map = {
            'config': self.default_config,
            'model': self.default_model,
            'decoder': llama4_text_decoder,
            'attention': llama4_text_attention
        }
        self.regist('llama4_text', llama4_text_map)

    def regist_qwen3_moe(self):
        qwen3_attention = {
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj',
            'q_norm': 'q_norm',
            'k_norm': 'k_norm'
        }
        qwen3_mlp = {
            'num_experts': 'num_experts',
            'top_k': 'top_k',
            'norm_topk_prob': 'norm_topk_prob',
            'gate': 'gate',
            'experts': 'experts'
        }
        qwen3_moe_map = {
            'config': self.default_config,
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': qwen3_attention,
            'mlp': qwen3_mlp,
        }
        self.regist('qwen3_moe', qwen3_moe_map)

    def regist_mimo(self):
        mimo_model = copy.deepcopy(self.default_model)
        mimo_model['mtp'] = 'model.mtp_layers'
        mimo_map = {
            'config': self.default_config,
            'model': mimo_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('mimo', mimo_map)

    def regist_poi_qwen2_mtp(self):
        poi_qwen2_mtp_model = copy.deepcopy(self.default_model)
        poi_qwen2_mtp_model['mtp1'] = 'MTP1'
        poi_qwen2_mtp_model['mtp2'] = 'MTP2'
        poi_qwen2_mtp_map = {
            'config': self.default_config,
            'model': poi_qwen2_mtp_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('poi_qwen2_mtp', poi_qwen2_mtp_map)

    def regist_glm(self):
        glm_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_layers'
            },
            'model': {
                'lm': 'lm_head',
                'embed': 'transformer.word_embeddings',
                'blocks': 'transformer.layers',
                'final_layernorm': 'transformer.final_layernorm',
            },
            'decoder': {
                'self_attn': 'attention',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'qkv_proj': 'query_key_value',
                'o_proj': 'dense'
            }
        }
        self.regist('chatglm', glm_map)

    def regist_glm2(self):
        glm2_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_key_value_heads': 'multi_query_group_num',
                'num_hidden_layers': 'num_layers',
                'rope_ratio': 'rope_ratio'
            },
            'model': {
                'lm': 'transformer.output_layer',
                'embed': 'transformer.embedding.word_embeddings',
                'blocks': 'transformer.encoder.layers',
                'final_layernorm': 'transformer.encoder.final_layernorm',
            },
            'decoder': {
                'self_attn': 'self_attention',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'qkv_proj': 'query_key_value',
                'o_proj': 'dense'
            }
        }
        self.regist('chatglm2', glm2_map)

    def regist_phi(self):
        phi_map = {
            'config': {
                'hidden_size': 'n_embd',
                'num_attention_heads': 'n_head',
                'num_hidden_layers': 'n_layer',
                'rotary_dim': 'rotary_dim'
            },
            'model': {
                'lm': 'lm_head.linear',
                'embed': 'transformer.embd.wte',
                'blocks': 'transformer.h',
                'final_layernorm': 'lm_head.ln',
            },
            'decoder': {
                'self_attn': 'mixer',
                'mlp': 'mlp',
                'input_layernorm': 'ln',
            },
            'attention': {
                'qkv_proj': 'Wqkv',
                'o_proj': 'out_proj'
            }
        }
        self.regist('phi-msft', phi_map)

    def regist_phi3(self):
        phi3_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'rope_theta': 'rope_theta',
                'head_dim': 'head_dim',
                'num_key_value_heads': 'num_key_value_heads',
            },
            'model': {
                'lm': 'lm_head',
                'embed': 'model.embed_tokens',
                'blocks': 'model.layers',
                'final_layernorm': 'model.norm'
            },
            'decoder': {
                'self_attn': 'self_attn',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'qkv_proj': 'qkv_proj',
            }
        }
        # self.regist('phi3', phi3_map)

    def regist_intervl(self):
        intervl_map = {
            'config': {
                'hidden_size': 'llm_config.hidden_size',
                'num_attention_heads': 'llm_config.num_attention_heads',
                'num_hidden_layers': 'llm_config.num_hidden_layers',
                'rope_theta': 'llm_config.rope_theta',
                'head_dim': 'llm_config.head_dim',
                'num_key_value_heads': 'llm_config.num_key_value_heads',
            },
            'model': {
                'lm': 'language_model.lm_head',
                'embed': 'language_model.model.embed_tokens',
                'blocks': 'language_model.model.layers',
                'final_layernorm': 'language_model.model.norm',
                'visual': 'vision_model'
            },
            'decoder': {
                'self_attn': 'self_attn',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'o_proj'
            }
        }
        self.regist('internvl_chat', intervl_map)

    def regist_gemma2(self):
        gemma2_decoder = copy.deepcopy(self.default_decoder)
        gemma2_decoder['pre_feedforward_layernorm'] = 'pre_feedforward_layernorm'
        gemma2_decoder['post_feedforward_layernorm'] = 'post_feedforward_layernorm'
        gemma2_map = {
            'config': self.default_config,
            'model': self.default_model,
            'decoder': gemma2_decoder,
            'attention': self.default_attention
        }
        self.regist('gemma2', gemma2_map)

    def regist_gemma3(self):
        gemma3_map = {
            'config': {
                'hidden_size': 'text_config.hidden_size',
                'head_dim': 'text_config.head_dim',
                'num_attention_heads': 'text_config.num_attention_heads',
                'num_hidden_layers': 'text_config.num_hidden_layers',
                'num_key_value_heads': 'text_config.num_key_value_heads',
                'rope_theta': 'text_config.rope_theta',

                'image_size': 'vision_config.image_size',
                'num_channels': 'vision_config.num_channels',

                'model_type': 'model_type',
                'image_token_index': 'image_token_index', #'<image_soft_token>'
                'boi_token_index': 'boi_token_index', #'<start_of_image>'
                'eoi_token_index': 'eoi_token_index', #'<end_of_image>'
            },
            'model': {
                'lm': 'language_model.lm_head',
                'embed': 'language_model.model.embed_tokens',
                'blocks': 'language_model.model.layers',
                'final_layernorm': 'language_model.model.norm',
                'vision_tower': 'vision_tower',
                'visual': 'vision_tower.vision_model',
                'multi_modal_projector': 'multi_modal_projector'
            },
            'decoder': {
                'self_attn': 'self_attn',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm',
                'pre_feedforward_layernorm': 'pre_feedforward_layernorm',
                'post_feedforward_layernorm': 'post_feedforward_layernorm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'o_proj',
                'q_norm': 'q_norm',
                'k_norm': 'k_norm'
            }
        }
        self.regist('gemma3', gemma3_map)

    def regist_gemma3_text(self):
        gemma3_text_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'head_dim': 'head_dim',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'num_key_value_heads': 'num_key_value_heads',
                'rope_theta': 'rope_theta',
                'max_position_embeddings': 'max_position_embeddings',
                'model_type': 'model_type',
                'vocab_size': 'vocab_size',
                'bos_token_id': 'bos_token_id',
                'eos_token_id': 'eos_token_id',
                'max_position_embeddings': 'max_position_embeddings',
                'pad_token_id': 'pad_token_id',
                'layer_types': 'layer_types',
                'sliding_window': 'sliding_window'
            },
            'model': {
                'lm': 'lm_head',
                'embed': 'model.embed_tokens',
                'blocks': 'model.layers',
                'final_layernorm': 'model.norm',
                'rotary_emb': 'model.rotary_emb',
                'rotary_emb_local': 'model.rotary_emb_local'
            },
            'decoder': {
                'self_attn': 'self_attn',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm',
                'pre_feedforward_layernorm': 'pre_feedforward_layernorm',
                'post_feedforward_layernorm': 'post_feedforward_layernorm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'o_proj',
                'q_norm': 'q_norm',
                'k_norm': 'k_norm'
            }
        }
        self.regist('gemma3_text', gemma3_text_map)

    def register_openelm(self):
        openelm_config = {
            'hidden_size': 'model_dim',
            'head_dim': 'head_dim',
            'num_attention_heads': 'num_query_heads',
            'num_hidden_layers': 'num_transformer_layers',
            'num_key_value_heads': 'num_kv_heads',
            'rope_theta': 'rope_freq_constant'
        }
        openelm_model = {
            'lm': 'lm_head',
            'embed': 'transformer.token_embeddings',
            'blocks': 'transformer.layers',
            'final_layernorm': 'transformer.norm'
        }
        openelm_decoder = {
            'self_attn': 'attn',
            'mlp': 'ffn',
            'input_layernorm': 'attn_norm',
            'post_attention_layernorm': 'ffn_norm'
        }
        openelm_attention = {
            'qkv_proj': 'qkv_proj',
            'o_proj': 'out_proj',
            'q_norm': 'q_norm',
            'k_norm': 'k_norm'
        }
        openelm_map = {
            'config': openelm_config,
            'model': openelm_model,
            'decoder': openelm_decoder,
            'attention': openelm_attention
        }
        self.regist('openelm', openelm_map)

    def regist_idefics3(self):
        idefics3_config = {
            'hidden_size': 'text_config.hidden_size',
            'head_dim': 'text_config.head_dim',
            'num_attention_heads': 'text_config.num_attention_heads',
            'num_hidden_layers': 'text_config.num_hidden_layers',
            'num_key_value_heads': 'text_config.num_key_value_heads',
            'rope_theta': 'text_config.rope_theta',
            'rope_scaling': 'text_config.rope_scaling'
        }
        idefics3_model = {
            'lm': 'lm_head',
            'embed': 'model.text_model.embed_tokens',
            'blocks': 'model.text_model.layers',
            'final_layernorm': 'model.text_model.norm',
            'visual': 'model.vision_model',
            'visual.connector': 'model.connector'
        }
        idefics3_map = {
            'config': idefics3_config,
            'model': idefics3_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('idefics3', idefics3_map)
        self.regist('smolvlm', idefics3_map)

    def regist_fastvlm(self):
        fastvlm_model = copy.deepcopy(self.default_model)
        fastvlm_model['visual'] = 'model.vision_tower'
        fastvlm_map = {
            'config': self.default_config,
            'model': fastvlm_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('llava_qwen2', fastvlm_map)

    def regist_qwenvl(self):
        if TRANSFORMERS_VERSION <= '4.52.1':
            return
        qwen2vl_model = {
            'lm': 'lm_head',
            'embed': 'model.language_model.embed_tokens',
            'blocks': 'model.language_model.layers',
            'final_layernorm': 'model.language_model.norm',
            'visual': 'model.visual'
        }
        qwen2vl_map = {
            'config': self.default_config,
            'model': qwen2vl_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('qwen2_vl', qwen2vl_map)
        self.regist('qwen2_5_vl', qwen2vl_map)
        qwen3vl_config = {
            'hidden_size': 'text_config.hidden_size',
            'head_dim': 'text_config.head_dim',
            'num_attention_heads': 'text_config.num_attention_heads',
            'num_hidden_layers': 'text_config.num_hidden_layers',
            'num_key_value_heads': 'text_config.num_key_value_heads',
            'rope_theta': 'text_config.rope_theta',
            'rope_scaling': 'text_config.rope_scaling',
            'max_position_embeddings': 'text_config.max_position_embeddings'
        }
        qwen3_attention = {
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj',
            'q_norm': 'q_norm',
            'k_norm': 'k_norm'
        }
        qwen3vl_map = {
            'config': qwen3vl_config,
            'model': qwen2vl_model,
            'decoder': self.default_decoder,
            'attention': qwen3_attention

        }
        qwen3vlmoe_mlp = {
            'num_experts': 'num_experts',
            'top_k': 'top_k',
            'gate': 'gate',
            'experts': 'experts'
        }
        qwen3vlmoe_map = {
            'config': qwen3vl_config,
            'model': qwen2vl_model,
            'decoder': self.default_decoder,
            'attention': qwen3_attention,
            'mlp': qwen3vlmoe_mlp
        }
        self.regist('qwen3_vl', qwen3vl_map)
        self.regist('qwen3_vl_moe', qwen3vlmoe_map)

    def regist_hunyuan_v1_dense(self):
        hunyuan_attention = {
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj',
            'q_norm': 'query_layernorm',
            'k_norm': 'key_layernorm'
        }
        hunyuan_map = {
            'config': self.default_config,
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': hunyuan_attention
        }
        self.regist('hunyuan_v1_dense', hunyuan_map)

    def regist_gpt_oss(self):
        gpt_oss_config = {
            'hidden_size': 'hidden_size',
            'head_dim': 'head_dim',
            'num_attention_heads': 'num_attention_heads',
            'num_hidden_layers': 'num_hidden_layers',
            'num_key_value_heads': 'num_key_value_heads',
            'rope_theta': 'rope_theta',
            'rope_scaling': 'rope_scaling',
            'max_position_embeddings': 'max_position_embeddings',
            'sliding_window': 'sliding_window',
            'layer_types': 'layer_types'
        }
        gpt_oss_attention = copy.deepcopy(self.default_attention)
        gpt_oss_attention['sinks'] = 'sinks'
        gpt_oss_mlp = {
            'num_experts': 'router.num_experts',
            'top_k': 'router.top_k',
            'router': 'router',
            'experts': 'experts'
        }
        gpt_osss_map = {
            'config': gpt_oss_config,
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': gpt_oss_attention,
            'mlp': gpt_oss_mlp
        }
        self.regist('gpt_oss', gpt_osss_map)

    def regist_minicpm(self):
        minicpm_config = copy.deepcopy(self.default_config)
        minicpm_config['scale_emb'] = 'scale_emb'
        minicpm_decoder = copy.deepcopy(self.default_decoder)
        minicpm_decoder['scale_depth'] = 'scale_depth'
        minicpm_map = {
            'config': minicpm_config,
            'model': self.default_model,
            'decoder': minicpm_decoder,
            'attention': self.default_attention
        }
        self.regist('minicpm', minicpm_map)

    def regist_minicpmv(self):
        minicpmv_config = copy.deepcopy(self.default_config)
        minicpmv_config['scale_emb'] = 'scale_emb'
        minicpmv_model = {
            'lm': 'llm.lm_head',
            'embed': 'llm.model.embed_tokens',
            'blocks': 'llm.model.layers',
            'final_layernorm': 'llm.model.norm',
            'visual': 'vpm',
            'resampler': 'resampler'
        }
        minicpmv_map = {
            'config': minicpmv_config,
            'model': minicpmv_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('minicpmv', minicpmv_map)

    def init_default_map(self):
        # default map is `LlamaForCausalLM`
        self.config_key = 'config'
        self.model_key = 'model'
        self.decoder_key = 'decoder'
        self.attention_key = 'attention'
        self.default_config = {
            'hidden_size': 'hidden_size',
            'head_dim': 'head_dim',
            'num_attention_heads': 'num_attention_heads',
            'num_hidden_layers': 'num_hidden_layers',
            'num_key_value_heads': 'num_key_value_heads',
            'rope_theta': 'rope_theta',
            'rope_scaling': 'rope_scaling',
            'max_position_embeddings': 'max_position_embeddings'
        }
        self.default_model = {
            'lm': 'lm_head',
            'embed': 'model.embed_tokens',
            'blocks': 'model.layers',
            'final_layernorm': 'model.norm',
            'visual': 'visual'
        }
        self.default_decoder = {
            'self_attn': 'self_attn',
            'mlp': 'mlp',
            'input_layernorm': 'input_layernorm',
            'post_attention_layernorm': 'post_attention_layernorm'
        }
        self.default_attention = {
            'qkv_proj': 'qkv_proj',
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj'
        }
        self.default_map = {
            'config': self.default_config,
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }

    @staticmethod
    def do_map(dst, src, mapping):
        # Sort mapping by key to ensure parents are set before children
        # e.g., 'visual' is processed before 'visual.connector' for SmolVLM
        for dst_path, src_path in sorted(mapping.items(), key=lambda x: x[0]):
            # --- 1. Retrieve value from source ---
            val = src
            for attr in src_path.split('.'):
                if hasattr(val, attr):
                    val = getattr(val, attr)
                else:
                    val = None
                    break

            # --- 2. Navigate to destination parent node ---
            dst_parts = dst_path.split('.')
            target = dst

            # Traverse to the second-to-last object
            path_valid = True
            for attr in dst_parts[:-1]:
                if hasattr(target, attr):
                    target = getattr(target, attr)
                    if target is None:
                        path_valid = False
                        break
                else:
                    path_valid = False
                    break

            # --- 3. Set value ---
            if path_valid and target:
                setattr(target, dst_parts[-1], val)