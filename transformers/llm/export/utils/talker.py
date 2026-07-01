import os
import torch
torch.set_printoptions(precision=4, sci_mode=False)
from .model_mapper import ModelMapper
from .transformers import Rotary, Decoder
from .token2wav import Qwen2_5OmniToken2Wav, Qwen3TTSToken2Wav
from .spinner import spinner_run
from .torch_utils import onnx_export
from .custom_op import FakeLinear

class Talker(torch.nn.Module):
    def __init__(self, talker, token2wav, base):
        super().__init__()
        self.model_type = base.config.model_type
        self.thinker_embed = base.embed
        self.args = base.args
        self.talker = talker.float()
        self.token2wav = Qwen2_5OmniToken2Wav(token2wav, base) if token2wav is not None else None
        self.config = base.config
        self.hidden_size = base.config.hidden_size
        self.llm_config = { 'has_talker': True }
        self.rope_ratio = 1.0
        self.quant_bit = 4
        if self.hidden_size <= 2048:
            # Qwen2.5-Omni-3B using 8 bit quantization
            self.quant_bit = 8
        self.init_config()
        self.load()

    def get_config(self):
        return self.llm_config

    @staticmethod
    def get_talker(model_type):
        audio_models = {
            'qwen2_5_omni': Qwen2_5OmniTalker,
            'qwen3_tts': Qwen3TTSTalker,
        }
        if model_type in audio_models:
            return audio_models[model_type]
        return None

    def init_config(self):
        pass

    def load(self):
        raise NotImplementedError

    def add_token_embeds(self, thinker_embeds):
        raise NotImplementedError

    def add_hidden_states(self, thinker_hidden_states):
        raise NotImplementedError

    def add_generate_ids(self, token_id):
        raise NotImplementedError

    def forward(self, inputs_embeds, attention_mask, position_ids):
        raise NotImplementedError

    def export(self, onnx_path):
        raise NotImplementedError

    def export_embed(self):
        import ctypes
        tensor_data = self.embed.weight.data.bfloat16()
        data_ptr = tensor_data.untyped_storage().data_ptr()
        buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
        embedding_file = f'{self.args.dst_path}/talker_embeddings_bf16.bin'
        with open(embedding_file, 'wb') as f:
            f.write(buffer)
        return embedding_file

class OmniRotary(Rotary):
    def __init__(self, model):
        super().__init__(model)
        self.mrope_section = model.mrope_section
        self.theta_sections = self.theta.unsqueeze(0).split(self.mrope_section, dim=-1)

    def forward(self, position_ids):
        position_ids = position_ids.float().unsqueeze(-1)
        idx_theta = torch.concat([
            position_ids[0] * self.theta_sections[0],
            position_ids[1] * self.theta_sections[1],
            position_ids[2] * self.theta_sections[2]
        ], dim=-1)
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(3)
        return rotary_pos_emb

_QWEN3_TTS_DECODER_MAP = {
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
        'o_proj': 'o_proj',
        'q_norm': 'q_norm',
        'k_norm': 'k_norm',
    }
}


def _unload_linear_children(module, prefix, unloaded_ops):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            linear_name = f'{prefix}/{name}/Linear'
            unloaded_ops[linear_name] = child.cpu()
            setattr(module, name, FakeLinear(child.in_features, child.out_features, child.bias is not None, linear_name))


def _unload_decoder_blocks(blocks, unloaded_ops):
    for i, block in enumerate(blocks):
        block.self_attn.export_fused_attn = True
        _unload_linear_children(block.self_attn, f'/layers.{i}/self_attn', unloaded_ops)
        _unload_linear_children(block.mlp, f'/layers.{i}/mlp', unloaded_ops)


class Qwen2_5OmniTalker(Talker):
    def __init__(self, talker, token2wav, base):
        super().__init__(talker, token2wav, base)
        self.input_hidden_size = base.config.hidden_size
        self.seq_len = 0
        self.token_len = 0
        self.talker_embeds = []

    def load(self):
        # load talker model
        self.model_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'head_dim': 'head_dim',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'num_key_value_heads': 'num_key_value_heads',
                'rope_parameters': 'rope_parameters',
                'rope_theta': 'rope_theta',
                'rope_scaling': 'rope_scaling'
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
                'o_proj': 'o_proj',
            }
        }
        ModelMapper.do_map(self, self.talker.config, self.model_map['config'])
        self.mrope_section = self.rope_scaling['mrope_section']
        if self.rope_theta is None and 'rope_theta' in self.rope_parameters:
            self.rope_theta = self.rope_parameters['rope_theta']
        self.embed = self.talker.model.embed_tokens
        self.rotary = OmniRotary(self)
        # self.rotary = Rotary(self)
        self.blocks = []
        for block in self.talker.model.layers:
            layer_id = len(self.blocks)
            decoder = Decoder(block, layer_id, self)
            decoder.self_attn.export_fused_attn = True
            self.blocks.append(decoder)


    def forward(self, inputs_embeds, attention_mask, position_ids):
        hidden_states = self.talker.thinker_to_talker_proj(inputs_embeds)
        rotary_pos_emb = self.rotary(position_ids)

        for i in range(self.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states, rotary_pos_emb, attention_mask)

        hidden_states = hidden_states[:, -1, :]
        hidden_states = self.talker.model.norm(hidden_states)
        logits = self.talker.codec_head(hidden_states)
        return logits

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            position_ids = torch.tensor([[self.seq_len - 1]], dtype=torch.int)
        else:
            position_ids = torch.arange(self.seq_len, dtype=torch.int).unsqueeze(0)
        position_ids = torch.stack([position_ids] * 3)
        return position_ids

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, 1, 1, self.seq_len], dtype=torch.float32)
        return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min

    def generate(self):
        talker_text_bos_token = 151872
        talker_inputs_embeds = torch.cat(
            [
                self.talker_embeds[0],
                self.thinker_embed(torch.tensor([[talker_text_bos_token]], dtype=torch.long)) + \
                self.embed(torch.LongTensor([self.talker.codec_pad_token])),
                self.talker_embeds[1] + self.embed(torch.LongTensor([self.talker.codec_bos_token])),
            ],
            dim=1,
        )
        thinker_reply_part = torch.cat(self.talker_embeds[2:], dim=1)
        thinker_reply_part = torch.cat(
            [
                thinker_reply_part,
                self.thinker_embed(
                    torch.tensor([[self.talker.text_eos_token]], dtype=torch.long)
                ),
                self.thinker_embed(
                    torch.tensor([[self.talker.text_pad_token]], dtype=torch.long)
                ),
            ],
            dim=1,
        )

        _, self.seq_len, _ = talker_inputs_embeds.shape
        _, reply_len, _ = thinker_reply_part.shape

        inputs_embeds = talker_inputs_embeds.float()
        self.token_len = 0
        self.stop_ids = [8292, 8294]
        token_id = None
        tokens = []
        while self.token_len < 256:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            if self.token_len > 0:
                inputs_embeds = self.embed(token_id)
                if self.token_len <= reply_len:
                    inputs_embeds = inputs_embeds + thinker_reply_part[:, self.token_len - 1, :]
                else:
                    inputs_embeds = inputs_embeds + thinker_reply_part[:, -1, :]
            logits = self.forward(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                position_ids=position_ids)
            token_id = torch.argmax(logits)
            self.token_len += 1
            self.seq_len += 1
            tokens.append(int(token_id))
            if int(token_id) in self.stop_ids:
                break
        talker_generate_codes = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        # 3. Generate wavs from code
        wav = self.token2wav.generate(talker_generate_codes,)
        import soundfile as sf
        sf.write(
            "output.wav",
            wav.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )

    def add_talker_embeds(self, talker_embed):
        self.talker_embeds.append(talker_embed)

    @spinner_run(f'export talker to ')
    def export(self, onnx_path):
        self.export_embed()
        self.seq_len = 3
        self.token_len = 0
        inputs_embeds = torch.randn([1, self.seq_len, self.input_hidden_size])
        posision_ids = self.get_position_ids()
        attention_mask = self.get_attention_mask()
        talker_onnx = f'{onnx_path}/talker.onnx'
        onnx_export(self, (inputs_embeds, attention_mask, posision_ids),
                    talker_onnx,
                    input_names=['inputs_embeds', 'attention_mask', 'position_ids'],
                    output_names=['logits'],
                    dynamic_axes={
                        "inputs_embeds": { 1: "size" },
                        "attention_mask": { 2: "size", 3: "size" },
                        "position_ids": { 2: "size" }
                    })
        return talker_onnx


class Qwen3TTSCodePredictor(torch.nn.Module):
    def __init__(self, predictor, base):
        super().__init__()
        self.predictor = predictor.float()
        self.args = base.args
        self.config = predictor.config
        self.model_type = base.config.model_type
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.config.head_dim
        self.num_attention_heads = self.config.num_attention_heads
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_key_value_heads = self.config.num_key_value_heads
        self.rope_theta = self.config.rope_theta
        self.rope_ratio = 1.0
        self.rope_scaling = getattr(self.config, 'rope_scaling', None)
        if hasattr(self.rope_scaling, '__dict__'):
            self.rope_scaling = vars(self.rope_scaling)
        self.model_map = _QWEN3_TTS_DECODER_MAP
        self.rotary = Rotary(self)
        self.blocks = []
        for block in self.predictor.model.layers:
            layer_id = len(self.blocks)
            decoder = Decoder(block, layer_id, self)
            decoder.self_attn.export_fused_attn = True
            self.blocks.append(decoder)

    def export_embed(self):
        import ctypes
        tensor_data = torch.stack([
            embed.weight.data.bfloat16()
            for embed in self.predictor.model.codec_embedding
        ], dim=0).contiguous()
        data_ptr = tensor_data.untyped_storage().data_ptr()
        buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
        embedding_file = f'{self.args.dst_path}/code_predictor_embeddings_bf16.bin'
        with open(embedding_file, 'wb') as f:
            f.write(buffer)
        return embedding_file

    def unload_param(self):
        self.unloaded_ops = {}
        with torch.no_grad():
            if isinstance(self.predictor.small_to_mtp_projection, torch.nn.Linear):
                linear = self.predictor.small_to_mtp_projection
                name = '/small_to_mtp_projection/Linear'
                self.unloaded_ops[name] = linear.cpu()
                self.predictor.small_to_mtp_projection = FakeLinear(
                    linear.in_features, linear.out_features, linear.bias is not None, name)
            _unload_decoder_blocks(self.blocks, self.unloaded_ops)
            for i, head in enumerate(self.predictor.lm_head):
                if isinstance(head, torch.nn.Linear):
                    name = f'/code_heads.{i}/Linear'
                    self.unloaded_ops[name] = head.cpu()
                    self.predictor.lm_head[i] = FakeLinear(
                        head.in_features, head.out_features, head.bias is not None, name)

    def forward(self, talker_hidden_states, codec_embeds, attention_mask, position_ids):
        hidden_states = torch.cat([talker_hidden_states.unsqueeze(1), codec_embeds], dim=1)
        hidden_states = self.predictor.small_to_mtp_projection(hidden_states)
        rotary_pos_emb = self.rotary(position_ids)
        for i in range(self.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states, rotary_pos_emb, attention_mask)
        hidden_states = self.predictor.model.norm(hidden_states)
        logits = []
        for i in range(self.config.num_code_groups - 1):
            logits.append(self.predictor.lm_head[i](hidden_states[:, i + 1:i + 2, :]))
        return torch.cat(logits, dim=1)

    @spinner_run(f'export qwen3_tts code predictor to ')
    def export(self, onnx_path):
        self.export_embed()
        self.unload_param()
        seq_len = self.config.num_code_groups
        talker_hidden_states = torch.randn([1, self.hidden_size])
        codec_embeds = torch.randn([1, self.config.num_code_groups - 1, self.hidden_size])
        attention_mask = (1 - torch.tril(torch.ones([1, 1, seq_len, seq_len]))) * torch.finfo(torch.float32).min
        position_ids = torch.arange(seq_len, dtype=torch.int).unsqueeze(0)
        code_predictor_onnx = f'{onnx_path}/code_predictor.onnx'
        onnx_export(self, (talker_hidden_states, codec_embeds, attention_mask, position_ids),
                    code_predictor_onnx,
                    input_names=['talker_hidden_states', 'codec_embeds', 'attention_mask', 'position_ids'],
                    output_names=['logits'],
                    dynamic_axes={
                        'attention_mask': { 2: 'size', 3: 'size' },
                        'position_ids': { 1: 'size' }
                    })
        return code_predictor_onnx


class Qwen3TTSPromptEmbedder(torch.nn.Module):
    def __init__(self, talker, config):
        super().__init__()
        self.talker = talker.float()
        self.config = config

    def forward(self, codec_embeds, text_raw_embeds, tts_raw_embeds):
        raw_embeds = torch.cat((tts_raw_embeds, text_raw_embeds), dim=1)
        projected_embeds = self.talker.text_projection(raw_embeds)
        tts_embeds = projected_embeds[:, :3]
        text_embeds = projected_embeds[:, 3:]
        tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_embeds.chunk(3, dim=1)
        role_embed = text_embeds[:, :3]
        prefill_embed = torch.cat((tts_pad_embed.expand(-1, codec_embeds.shape[1] - 2, -1), tts_bos_embed), dim=1)
        prefill_embed = prefill_embed + codec_embeds[:, :-1]
        first_text_embed = text_embeds[:, 3:4]
        talker_input_embed = torch.cat((role_embed, prefill_embed, first_text_embed + codec_embeds[:, -1:]), dim=1)
        trailing_text_hidden = torch.cat((text_embeds[:, 4:-5], tts_eos_embed), dim=1)
        return talker_input_embed, trailing_text_hidden, tts_pad_embed

    @spinner_run(f'export qwen3_tts prompt embedder to ')
    def export(self, onnx_path):
        codec_embeds = torch.randn([1, 6, self.config.talker_config.hidden_size], dtype=torch.float32)
        text_raw_embeds = torch.randn([1, 11, self.config.talker_config.text_hidden_size], dtype=torch.float32)
        tts_raw_embeds = torch.randn([1, 3, self.config.talker_config.text_hidden_size], dtype=torch.float32)
        prompt_embedder_onnx = f'{onnx_path}/prompt_embedder.onnx'
        onnx_export(self, (codec_embeds, text_raw_embeds, tts_raw_embeds), prompt_embedder_onnx,
                    input_names=['codec_embeds', 'text_raw_embeds', 'tts_raw_embeds'],
                    output_names=['inputs_embeds', 'trailing_text_hidden', 'tts_pad_embed'],
                    dynamic_axes={
                        'codec_embeds': {1: 'codec_prefix_len'},
                        'text_raw_embeds': {1: 'text_len'},
                        'inputs_embeds': {1: 'prompt_len'},
                        'trailing_text_hidden': {1: 'trailing_len'},
                    })
        return prompt_embedder_onnx


class Qwen3TTSCodecEmbedder(torch.nn.Module):
    def __init__(self, talker, code_predictor):
        super().__init__()
        self.talker = talker.float()

    def forward(self, codec_embeds, text_hidden):
        codec_embed = codec_embeds.sum(1, keepdim=True)
        return codec_embed + text_hidden.unsqueeze(1)

    @spinner_run(f'export qwen3_tts codec embedder to ')
    def export(self, onnx_path):
        codec_embeds = torch.randn([1, self.talker.config.code_predictor_config.num_code_groups,
                                    self.talker.config.hidden_size], dtype=torch.float32)
        text_hidden = torch.randn([1, self.talker.config.hidden_size], dtype=torch.float32)
        codec_embedder_onnx = f'{onnx_path}/codec_embedder.onnx'
        onnx_export(self, (codec_embeds, text_hidden), codec_embedder_onnx,
                    input_names=['codec_embeds', 'text_hidden'],
                    output_names=['inputs_embeds'],
                    dynamic_axes={
                        'codec_embeds': {1: 'code_groups'},
                    })
        return codec_embedder_onnx

class Qwen3TTSTalker(Talker):
    def __init__(self, talker, token2wav, base):
        super().__init__(talker, token2wav, base)

    def init_config(self):
        self.quant_bit = self.args.quant_bit
        self.llm_config = {
            'has_talker': True,
            'talker_model': 'talker.mnn',
            'talker_weight': 'talker.mnn.weight',
            'talker_embedding_file': 'talker_embeddings_bf16.bin',
            'talker_text_embedding_file': 'talker_text_embeddings_bf16.bin',
            'talker_text_hidden_size': self.talker.config.text_hidden_size,
            'tts_bos_token_id': self.config.origin_config.tts_bos_token_id,
            'tts_eos_token_id': self.config.origin_config.tts_eos_token_id,
            'tts_pad_token_id': self.config.origin_config.tts_pad_token_id,
            'talker_type': 'qwen3_tts',
            'code_predictor_model': 'code_predictor.mnn',
            'code_predictor_weight': 'code_predictor.mnn.weight',
            'code_predictor_embedding_file': 'code_predictor_embeddings_bf16.bin',
            'code_predictor_vocab_size': self.talker.config.code_predictor_config.vocab_size,
            'code_predictor_groups': self.talker.config.code_predictor_config.num_code_groups,
            'speech_decoder_model': 'speech_decoder.mnn',
            'speech_decoder_weight': 'speech_decoder.mnn.weight',
            'speech_decoder_upsample_rate': 1920,
            'speaker_encoder_model': 'speaker_encoder.mnn',
            'speaker_encoder_weight': 'speaker_encoder.mnn.weight',
            'speaker_encoder_sample_rate': 24000,
            'codec_embedder_model': 'codec_embedder.mnn',
            'codec_embedder_weight': 'codec_embedder.mnn.weight',
            'jinja': {
                'chat_template': '{% if qwen3_tts_language == "auto" %} differentiable Me beg begCH{% elif qwen3_tts_language == "chinese" %}spiable So Me begCH{% elif qwen3_tts_language == "english" %}spiableakes Me begCH{% elif qwen3_tts_language == "german" %}spiable ref Me begCH{% elif qwen3_tts_language == "italian" %}spiable field Me begCH{% elif qwen3_tts_language == "portuguese" %}spiableiven Me begCH{% elif qwen3_tts_language == "spanish" %}spiableever Me begCH{% elif qwen3_tts_language == "japanese" %}spiable still Me begCH{% elif qwen3_tts_language == "korean" %}spiable rep Me begCH{% elif qwen3_tts_language == "french" %}spiablemed Me begCH{% elif qwen3_tts_language == "russian" %}spiableOD Me begCH{% else %}spiableakes Me begCH{% endif %}{% for message in messages %}{% if message.role == "assistant" %}<|im_start|>assistant\n{{ message.content }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}',
                'context': {
                    'qwen3_tts_language': 'auto'
                }
            },
        }

    def load(self):
        self.model_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'head_dim': 'head_dim',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'num_key_value_heads': 'num_key_value_heads',
                'rope_theta': 'rope_theta',
                'rope_scaling': 'rope_scaling'
            }
        }
        self.model_map.update(_QWEN3_TTS_DECODER_MAP)
        ModelMapper.do_map(self, self.talker.config, self.model_map['config'])
        if hasattr(self.rope_scaling, '__dict__'):
            self.rope_scaling = vars(self.rope_scaling)
        self.embed = self.talker.model.codec_embedding
        self.rotary = Rotary(self)
        self.blocks = []
        for block in self.talker.model.layers:
            layer_id = len(self.blocks)
            decoder = Decoder(block, layer_id, self)
            decoder.self_attn.export_fused_attn = True
            self.blocks.append(decoder)
        self.code_predictor = Qwen3TTSCodePredictor(self.talker.code_predictor, self)
        self.prompt_embedder = Qwen3TTSPromptEmbedder(self.talker, self.config.origin_config)
        self.codec_embedder = Qwen3TTSCodecEmbedder(self.talker, self.talker.code_predictor)
        self.token2wav = Qwen3TTSToken2Wav(self)
        self.llm_config['speech_decoder_upsample_rate'] = self.token2wav.decode_upsample_rate

    def export_text_embed(self):
        import ctypes
        tensor_data = self.talker.model.text_embedding.weight.data.bfloat16()
        data_ptr = tensor_data.untyped_storage().data_ptr()
        buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
        embedding_file = f'{self.args.dst_path}/talker_text_embeddings_bf16.bin'
        with open(embedding_file, 'wb') as f:
            f.write(buffer)
        return embedding_file

    def unload_param(self):
        self.unloaded_ops = {}
        with torch.no_grad():
            _unload_decoder_blocks(self.blocks, self.unloaded_ops)
            _unload_linear_children(self.talker.text_projection, '/text_projection', self.unloaded_ops)
            if isinstance(self.talker.codec_head, torch.nn.Linear):
                linear = self.talker.codec_head
                name = '/codec_head/Linear'
                self.unloaded_ops[name] = linear.cpu()
                self.talker.codec_head = FakeLinear(
                    linear.in_features, linear.out_features, linear.bias is not None, name)

    def add_talker_embeds(self, talker_embed):
        pass

    def add_token_embeds(self, thinker_embeds):
        pass

    def add_hidden_states(self, thinker_hidden_states):
        pass

    def add_generate_ids(self, token_id):
        pass

    def forward(self, inputs_embeds, attention_mask, position_ids, codec_embeds, text_raw_embeds, tts_raw_embeds):
        rotary_pos_emb = self.rotary(position_ids)
        hidden_states = inputs_embeds
        for i in range(self.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states, rotary_pos_emb, attention_mask)
        hidden_states = self.talker.model.norm(hidden_states)
        hidden_states = hidden_states[:, -1, :]
        logits = self.talker.codec_head(hidden_states)
        prompt_inputs, trailing_text_hidden, tts_pad_embed = self.prompt_embedder(
            codec_embeds, text_raw_embeds, tts_raw_embeds)
        return logits, hidden_states, prompt_inputs, trailing_text_hidden, tts_pad_embed

    @spinner_run(f'export qwen3_tts talker to ')
    def export(self, onnx_path):
        self.export_embed()
        self.export_text_embed()
        self.unload_param()
        seq_len = 3
        inputs_embeds = torch.randn([1, seq_len, self.hidden_size])
        position_ids = torch.stack([torch.arange(seq_len, dtype=torch.int)] * 3)
        attention_mask = (1 - torch.tril(torch.ones([1, 1, seq_len, seq_len]))) * torch.finfo(torch.float32).min
        codec_embeds = torch.randn([1, 6, self.hidden_size], dtype=torch.float32)
        text_raw_embeds = torch.randn([1, 11, self.talker.config.text_hidden_size], dtype=torch.float32)
        tts_raw_embeds = torch.randn([1, 3, self.talker.config.text_hidden_size], dtype=torch.float32)
        talker_onnx = f'{onnx_path}/talker.onnx'
        onnx_export(self, (inputs_embeds, attention_mask, position_ids, codec_embeds, text_raw_embeds, tts_raw_embeds),
                    talker_onnx,
                    input_names=['inputs_embeds', 'attention_mask', 'position_ids',
                                 'codec_embeds', 'text_raw_embeds', 'tts_raw_embeds'],
                    output_names=['logits', 'hidden_states', 'prompt_inputs_embeds',
                                  'trailing_text_hidden', 'tts_pad_embed'],
                    dynamic_axes={
                        "inputs_embeds": { 1: "size" },
                        "attention_mask": { 2: "size", 3: "size" },
                        "position_ids": { 1: "size" },
                        "codec_embeds": { 1: "codec_prefix_len" },
                        "text_raw_embeds": { 1: "text_len" },
                        "prompt_inputs_embeds": { 1: "prompt_len" },
                        "trailing_text_hidden": { 1: "trailing_len" },
                    })
        return [talker_onnx,
                self.code_predictor.export(onnx_path),
                self.codec_embedder.export(onnx_path)]
