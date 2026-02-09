import os
import torch
torch.set_printoptions(precision=4, sci_mode=False)
from .model_mapper import ModelMapper
from .transformers import Rotary, Embedding, Decoder
from .token2wav import Qwen2_5OmniToken2Wav
from .spinner import spinner_run
from .torch_utils import onnx_export

class Talker(torch.nn.Module):
    def __init__(self, talker, token2wav, base):
        super().__init__()
        self.model_type = base.config.model_type
        self.thinker_embed = base.embed
        self.args = base.args
        self.talker = talker.float()
        self.token2wav = Qwen2_5OmniToken2Wav(token2wav, base)
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