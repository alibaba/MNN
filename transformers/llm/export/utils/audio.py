import torch
from .transformers import Decoder
from .spinner import spinner_run
from .torch_utils import onnx_export

class Audio(torch.nn.Module):
    def __init__(self, audio, base):
        super().__init__()
        self.model_type = base.config.model_type
        self.audio = audio
        self.embed_ = base.embed
        self.tokenizer = base.tokenizer
        self.config = base.config.origin_config
        self.hidden_size = base.config.hidden_size
        self.llm_config = { 'is_audio': True }
        self.rope_ratio = 1.0
        self.quant_bit = 16
        self.init_config()
        self.load()

    def get_config(self):
        return self.llm_config

    @staticmethod
    def get_audio(model_type):
        audio_models = {
            'qwen2_audio_encoder': Qwen2Audio,
            'qwen2_5_omni_audio_encoder': Qwen2_5OmniAudio,
            'funaudiochat_audio_encoder': FunAudioChatAudio,
            'lfm2_audio': Lfm2Audio,
            'gemma4_audio': Gemma4Audio,
        }
        if model_type in audio_models:
            return audio_models[model_type]
        return None

    def init_config(self):
        pass

    def load(self):
        raise NotImplementedError

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        return input_ids

    def forward(self, images):
        raise NotImplementedError

    def embed(self, input_ids, images = None, videos = None):
        raise NotImplementedError

    def export(self, onnx_path):
        raise NotImplementedError

class Qwen2Audio(Audio):
    def __init__(self, audio, base):
        super().__init__(audio, base)
        self.audio_embeds = None
        self.audio_pad_id = 151646
        self.n_fft = 400
        self.sampling_rate = 16000
        self.hop_length = 160
        self.chunk_length = 30
        self.feature_size = 128
        self.n_samples = self.chunk_length * self.sampling_rate
        self.max_length = self.n_samples // self.hop_length
        from transformers.audio_utils import mel_filter_bank
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def load(self):
        # model
        self.audio_tower = self.audio
        self.multi_modal_projector = self.audio.multi_modal_projector
        # config
        self.llm_config['is_audio'] = True

    def str_to_ids(self, prompt):
        if '<audio>' in prompt and '</audio>' in prompt:
            import re
            from io import BytesIO
            from urllib.request import urlopen
            import librosa
            pattern = r'(<audio>.*?</audio>)'
            parts = re.split(pattern, prompt)
            txt_prompt = ''
            for part in parts:
                if re.match(pattern, part):
                    audio_content = re.search(r'<audio>(.*?)</audio>', part).group(1)
                    if audio_content.startswith('http://') or audio_content.startswith('https://'):
                        audio_obj = librosa.load(BytesIO(urlopen(audio_content).read()), sr=self.sampling_rate)[0]
                    else:
                        # local file
                        audio_obj = librosa.load(audio_content, sr=self.sampling_rate)[0]
                    audio_embed_len = self.audio_process(audio_obj)
                    audio_pad_str = '<|AUDIO|>' * audio_embed_len
                    txt_prompt += audio_pad_str
                else:
                    txt_prompt += part
        else:
            txt_prompt = prompt
        input_ids = self.tokenizer(txt_prompt, return_tensors="pt")['input_ids']
        return input_ids

    def forward(self, input_features):
        input_features = input_features.to(dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device)
        inputs_embeds = torch.nn.functional.gelu(self.audio_tower.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.audio_tower.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        _, seq_len, _ = inputs_embeds.shape
        embed_pos = self.audio_tower.embed_positions.weight[:seq_len, :]
        hidden_states = inputs_embeds + embed_pos
        for encoder_layer in self.audio_tower.layers:
            hidden_states = encoder_layer(hidden_states, None, None)[0]
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.audio_tower.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.audio_tower.layer_norm(hidden_states)
        audio_features = self.multi_modal_projector(hidden_states)
        return audio_features

    def _torch_extract_fbank_features(self, waveform):
        window = torch.hann_window(self.n_fft)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def audio_process(self, audio_obj):
        # audio_obj = np.pad(audio_obj, (0, self.n_samples - audio_obj.shape[0]))
        waveform = torch.from_numpy(audio_obj).type(torch.float32)
        input_features = self._torch_extract_fbank_features(waveform).unsqueeze(0)
        audio_embeds = self.forward(input_features)
        self.audio_embeds = audio_embeds.permute([1, 0, 2])
        return self.audio_embeds.shape[0]

    def embed(self, input_ids, images = None, videos = None):
        input_embeds = self.embed_(input_ids)
        if self.audio_embeds is not None:
            audio_mask = (input_ids == self.audio_pad_id).squeeze()
            input_embeds[audio_mask] = self.audio_embeds.type(input_embeds.dtype)
        return input_embeds

    @spinner_run(f'export audio to ')
    def export(self, onnx_path):
        input_features = torch.randn((1, self.feature_size, self.max_length))

        model = self.float()
        onnx_model = f'{onnx_path}/audio.onnx'
        onnx_export(model, (input_features),
                    onnx_model,
                    input_names=['input_features'],
                    output_names=['audio_embeds'],
                    dynamic_axes={"input_features": {
                        2: "size"
                    }})
        return onnx_model

class AudioMlp(torch.nn.Module):
    def __init__(self, fc1, fc2, act):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.act = act

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Qwen2_5OmniAudio(Qwen2Audio):
    def __init__(self, audio, base):
        super().__init__(audio, base)
        self.quant_bit = 4

    def load(self):
        # config
        config = self.audio.config
        self.n_window = config.n_window
        self.llm_config['is_audio'] = True
        self.llm_config['n_window'] = self.n_window
        self.hidden_size = config.d_model
        self.num_attention_heads = config.encoder_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rotary = None
        self.model_map = {
            'decoder': {
                'self_attn': 'self_attn',
                'input_layernorm': 'self_attn_layer_norm',
                'post_attention_layernorm': 'final_layer_norm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'out_proj'
            }
        }
        self.blocks = []
        for layer in self.audio.layers:
            layer_id = len(self.blocks)
            block = Decoder(layer, layer_id, self)
            block.mlp = AudioMlp(layer.fc1, layer.fc2, layer.activation_fn)
            self.blocks.append(block)

    def forward(self, input_features, attention_mask = None):
        input_features = input_features.to(dtype=self.audio.conv1.weight.dtype, device=self.audio.conv1.weight.device)
        inputs_embeds = torch.nn.functional.gelu(self.audio.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.audio.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        _, seq_len, _ = inputs_embeds.shape
        embed_pos = self.audio.positional_embedding.positional_embedding[:seq_len, :]
        hidden_states = inputs_embeds + embed_pos
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.audio.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.audio.ln_post(hidden_states)
        audio_features = self.audio.proj(hidden_states)
        return audio_features

    def audio_process(self, audio_obj):
        # audio_obj = np.pad(audio_obj, (0, self.n_samples - audio_obj.shape[0]))
        waveform = torch.from_numpy(audio_obj).type(torch.float32)
        input_features = self._torch_extract_fbank_features(waveform).unsqueeze(0)
        _, _, seq_len = input_features.shape
        seq_len = int(seq_len // 2)
        cu_seqlens = [i for i in range(0, seq_len, self.n_window)]
        if seq_len % self.n_window != 0:
            cu_seqlens.append(seq_len)
        cu_seqlens = torch.tensor(cu_seqlens)
        attention_mask = torch.full(
            [1, seq_len, seq_len], torch.finfo(torch.float32).min
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        audio_embeds = self.forward(input_features, attention_mask)
        self.audio_embeds = audio_embeds.permute([1, 0, 2])
        return self.audio_embeds.shape[0]

    @spinner_run(f'export audio to ')
    def export(self, onnx_path):
        input_features = torch.randn((1, self.feature_size, self.max_length))
        seq_len = self.max_length // 2
        attention_mask = torch.randn([1, seq_len, seq_len])
        model = self.float()
        onnx_model = f'{onnx_path}/audio.onnx'
        onnx_export(model, (input_features, attention_mask),
                    onnx_model,
                    input_names=['input_features', 'attention_mask'],
                    output_names=['audio_embeds'],
                    dynamic_axes={"input_features": {
                        0: "size"
                    }, "attention_mask": {
                        1: "size", 2: "size"
                    }})
        return onnx_model

class FunAudioChatAudio(Qwen2_5OmniAudio):
    def __init__(self, audio, base):
        super().__init__(audio, base)
        self.audio_pad_id = 151669

    def load(self):
        # model
        self.audio = self.audio.float()
        self.audio_tower = self.audio.audio_tower.float()
        # config
        self.group_size = self.audio.config.group_size
        # call parent load
        super().load()

    def forward(self, input_features, attention_mask = None):
        # call parent forward to get audio_features before group pooling
        audio_features = super().forward(input_features, attention_mask)
        # group pooling and continual_output_matching
        batch, seqlen, hidden_size = audio_features.shape
        padding_feature = torch.zeros(
            (batch, (self.group_size - seqlen % self.group_size) % self.group_size, hidden_size),
            dtype=torch.long,
            device=audio_features.device,
        )
        audio_features = torch.cat([audio_features, padding_feature], dim=1)
        audio_features = audio_features.reshape(batch, -1, self.group_size, hidden_size)
        audio_features = audio_features.mean(dim=2)
        audio_features = self.audio_tower.continual_output_matching(audio_features)
        return audio_features


class Lfm2Audio(Audio):
    """Audio encoder for LFM2-Audio (FastConformer + MLP adapter).

    Supports audio understanding: audio → conformer → adapter → inject into LFM → text.
    """

    def __init__(self, audio, base):
        # Store adapter and constants before super().__init__() using __dict__
        # to bypass Module.__setattr__ (which requires __init__ to be called first)
        self.__dict__['_audio_adapter_ref'] = base.audio_adapter
        self.__dict__['audio_pad_id'] = 16  # <|reserved_6|> as audio placeholder
        self.__dict__['sampling_rate'] = 16000
        super().__init__(audio, base)
        self.audio_embeds = None
        self.quant_bit = 4

    def load(self):
        self.conformer = self.audio.float()
        self.audio_adapter_module = self._audio_adapter_ref.float()
        self.llm_config['is_audio'] = True
        self.llm_config['audio_type'] = 'conformer'
        self.llm_config['audio_pad'] = self.audio_pad_id

        # Initialize mel spectrogram preprocessor (matching config.json preprocessor settings)
        from liquid_audio.model.conformer.processor import AudioToMelSpectrogramPreprocessor
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=16000, window_size=0.025, window_stride=0.01,
            window='hann', normalize='per_feature', n_fft=512,
            features=128, log=True, dither=1e-5, pad_to=0, pad_value=0.0,
        ).eval().float()

    def forward(self, input_features, input_lengths):
        """Run conformer encoder + adapter on mel features.

        Args:
            input_features: [B, 128, T] mel spectrogram
            input_lengths: [B] actual mel lengths
        Returns:
            audio_features: [T_valid, hidden_size] (valid tokens only, padding removed)
            enc_lens: [B] valid token counts
        """
        audio_enc, enc_lens = self.conformer(input_features, input_lengths)
        # audio_enc: [B, d_model=512, T_enc]
        # Extract valid (non-padded) tokens using boolean mask
        len_mask = torch.arange(audio_enc.shape[-1], device=audio_enc.device).unsqueeze(0) < enc_lens.unsqueeze(1)
        audio_enc_valid = audio_enc.transpose(1, 2)[len_mask]  # [T_valid, 512]
        # Project to LFM hidden size
        audio_features = self.audio_adapter_module(audio_enc_valid)  # [T_valid, 2048]
        return audio_features, enc_lens

    def audio_process(self, audio_obj):
        """Process raw audio waveform to get audio embeddings.

        Args:
            audio_obj: numpy array of audio samples (16kHz)
        Returns:
            num_tokens: number of audio embedding tokens
        """
        waveform = torch.from_numpy(audio_obj).float().unsqueeze(0)  # [1, T]
        length = torch.tensor([waveform.shape[1]], dtype=torch.long)
        mel, mel_len = self.preprocessor(waveform, length)  # [1, 128, T_mel]
        audio_features, enc_lens = self.forward(mel, mel_len)  # [T_valid, 2048]
        self.audio_embeds = audio_features.unsqueeze(1)  # [T_valid, 1, 2048]
        return self.audio_embeds.shape[0]

    def str_to_ids(self, prompt):
        if '<audio>' not in prompt:
            return self.tokenizer(prompt, return_tensors="pt")['input_ids']

        import re
        import librosa
        pattern = r'(<audio>.*?</audio>)'
        parts = re.split(pattern, prompt)
        # No manual BOS — the chat template already includes <|startoftext|>
        all_ids = []
        for part in parts:
            if re.match(pattern, part):
                audio_path = re.search(r'<audio>(.*?)</audio>', part).group(1)
                audio_obj = librosa.load(audio_path, sr=self.sampling_rate)[0]
                num_tokens = self.audio_process(audio_obj)
                all_ids.extend([self.audio_pad_id] * num_tokens)
            else:
                if part:
                    ids = self.tokenizer.encode(part, add_special_tokens=False)
                    all_ids.extend(ids)
        return torch.tensor([all_ids])

    def embed(self, input_ids, images=None, videos=None):
        input_embeds = self.embed_(input_ids)
        if self.audio_embeds is not None:
            audio_mask = (input_ids == self.audio_pad_id).squeeze()
            input_embeds[audio_mask] = self.audio_embeds.type(input_embeds.dtype)
        return input_embeds

    @spinner_run(f'export audio to ')
    def export(self, onnx_path):
        class AudioExport(torch.nn.Module):
            def __init__(self, conformer, adapter):
                super().__init__()
                self.conformer = conformer
                self.adapter = adapter

            def forward(self, input_features):
                # input_features: [1, 128, T_mel]
                input_length = torch.tensor(
                    [input_features.shape[2]], dtype=torch.long, device=input_features.device
                )
                audio_enc, enc_lens = self.conformer(input_features, input_length)
                # audio_enc: [1, 512, T_enc]
                audio_enc = audio_enc.transpose(1, 2)  # [1, T_enc, 512]
                audio_features = self.adapter(audio_enc)  # [1, T_enc, 2048]
                return audio_features

        model = AudioExport(self.conformer, self.audio_adapter_module).float().eval()
        # Pre-allocate positional embeddings for max sequence length
        model.conformer.set_max_audio_length(5000)

        input_features = torch.randn((1, 128, 1000))
        onnx_model = f'{onnx_path}/audio.onnx'
        onnx_export(model, (input_features,),
                    onnx_model,
                    input_names=['input_features'],
                    output_names=['audio_embeds'],
                    dynamic_axes={"input_features": {
                        2: "size"
                    }})
        return onnx_model


class Gemma4AudioExportModel(torch.nn.Module):
    """ONNX-exportable wrapper for gemma4 audio encoder.

    Replaces unfold-based chunked attention with index-gather approach,
    and skips HF's create_bidirectional_mask (not needed for non-padded audio).
    """

    def __init__(self, audio_tower, embed_audio):
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio
        cfg = audio_tower.config
        self.chunk_size = cfg.attention_chunk_size
        self.max_past = cfg.attention_context_left - 1
        self.max_future = cfg.attention_context_right
        self.context_size = self.chunk_size + self.max_past + self.max_future
        self.gradient_clipping = cfg.gradient_clipping

    def _clippable_linear(self, module, x):
        if module.use_clipped_linears:
            x = torch.clamp(x, module.input_min, module.input_max)
        x = module.linear(x)
        if module.use_clipped_linears:
            x = torch.clamp(x, module.output_min, module.output_max)
        return x

    def _convert_to_block(self, hidden_states):
        B, S, H, D = hidden_states.shape
        num_blocks = (S + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - S
        if pad > 0:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, 0, pad), value=0.0)
        return hidden_states.reshape(B, num_blocks, self.chunk_size, H, D)

    def _extract_block_context(self, hidden_states):
        B, S, H, D = hidden_states.shape
        num_blocks = (S + self.chunk_size - 1) // self.chunk_size
        padded = torch.nn.functional.pad(
            hidden_states, (0, 0, 0, 0, self.max_past, self.max_future + self.chunk_size - 1), value=0.0
        )
        offsets = torch.arange(self.context_size, device=hidden_states.device)
        block_starts = torch.arange(num_blocks, device=hidden_states.device) * self.chunk_size
        indices = (block_starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
        result = padded[:, indices].reshape(B, num_blocks, self.context_size, H, D)
        return result

    def _rel_shift(self, x):
        B, H, NB, BS, PL = x.shape
        CS = self.context_size
        x = torch.nn.functional.pad(x, (0, CS + 1 - PL), value=0.0)
        x = x.reshape(B, H, NB, BS * (CS + 1))
        x = x[..., :BS * CS]
        return x.reshape(B, H, NB, BS, CS)

    def _build_blocked_mask(self, S, device):
        """Build 5D blocked sliding window attention mask."""
        num_blocks = (S + self.chunk_size - 1) // self.chunk_size
        q_idx = torch.arange(self.chunk_size, device=device)
        c_idx = torch.arange(self.context_size, device=device)
        b_idx = torch.arange(num_blocks, device=device)
        abs_query = b_idx.unsqueeze(1) * self.chunk_size + q_idx.unsqueeze(0)
        abs_key = b_idx.unsqueeze(1) * self.chunk_size - self.max_past + c_idx.unsqueeze(0)
        query_valid = abs_query < S
        key_valid = (abs_key >= 0) & (abs_key < S)
        # Sliding window: c > q AND c <= q + max_past (strict left, closed right)
        slide = (c_idx.unsqueeze(0) > q_idx.unsqueeze(1)) & \
                (c_idx.unsqueeze(0) <= q_idx.unsqueeze(1) + self.max_past)
        mask = query_valid.unsqueeze(2) & key_valid.unsqueeze(1) & slide.unsqueeze(0)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_blocks, chunk_size, context_size]

    def _conformer_attention(self, attn, hidden_states, position_embeddings, attention_mask):
        B, S, _ = hidden_states.shape
        num_heads = attn.num_heads
        head_dim = attn.head_dim

        query_states = self._clippable_linear(attn.q_proj, hidden_states).float()
        key_states = self._clippable_linear(attn.k_proj, hidden_states).float()
        value_states = self._clippable_linear(attn.v_proj, hidden_states).float()

        query_states = query_states.view(B, S, num_heads, head_dim)
        key_states = key_states.view(B, S, num_heads, head_dim)
        value_states = value_states.view(B, S, num_heads, head_dim)

        query_states = query_states * attn.q_scale * torch.nn.functional.softplus(attn.per_dim_scale)
        key_states = key_states * attn.k_scale

        query_blocked = self._convert_to_block(query_states)
        key_context = self._extract_block_context(key_states)
        value_context = self._extract_block_context(value_states)
        num_blocks = query_blocked.shape[1]

        relative_key_states = attn.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, num_heads, head_dim).to(query_states.dtype)

        queries = query_blocked.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_context.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(B, num_heads, -1, head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(B, num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / attn.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * attn.softcap

        # Apply sliding window mask
        if attention_mask is not None:
            invalid_value = torch.tensor(-1e9, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(attention_mask, attn_weights, invalid_value)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(value_context.dtype)
        attn_output = attn_weights @ value_context.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(B, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :S].contiguous()
        attn_output = self._clippable_linear(attn.post, attn_output.to(attn.post.linear.weight.dtype))
        return attn_output

    def _feed_forward(self, ff, hidden_states):
        gc = min(self.gradient_clipping, torch.finfo(ff.ffw_layer_1.linear.weight.dtype).max)
        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = ff.pre_layer_norm(hidden_states)
        hidden_states = self._clippable_linear(ff.ffw_layer_1, hidden_states)
        hidden_states = ff.act_fn(hidden_states)
        hidden_states = self._clippable_linear(ff.ffw_layer_2, hidden_states)
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = ff.post_layer_norm(hidden_states)
        hidden_states = hidden_states * ff.post_layer_scale
        hidden_states = hidden_states + residual
        return hidden_states

    def _causal_conv1d(self, conv, x):
        """Causal Conv1d with explicit pad value for ONNX compatibility."""
        left_pad = (conv.kernel_size[0] - 1) * conv.dilation[0] + 1 - conv.stride[0]
        x = torch.nn.functional.pad(x, (left_pad, 0), value=0.0)
        return torch.nn.functional.conv1d(x, conv.weight, conv.bias,
                                          stride=conv.stride, dilation=conv.dilation, groups=conv.groups)

    def _light_conv(self, lconv, hidden_states):
        residual = hidden_states
        hidden_states = lconv.pre_layer_norm(hidden_states)
        hidden_states = self._clippable_linear(lconv.linear_start, hidden_states)
        hidden_states = torch.nn.functional.glu(hidden_states, dim=-1)
        hidden_states = self._causal_conv1d(lconv.depthwise_conv1d, hidden_states.transpose(1, 2)).transpose(1, 2)
        gc = min(self.gradient_clipping, torch.finfo(lconv.linear_start.linear.weight.dtype).max)
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = lconv.conv_norm(hidden_states)
        hidden_states = lconv.act_fn(hidden_states)
        hidden_states = self._clippable_linear(lconv.linear_end, hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states

    def _encoder_layer(self, layer, hidden_states, position_embeddings, attention_mask):
        gc = min(self.gradient_clipping, torch.finfo(layer.norm_pre_attn.weight.dtype).max)
        hidden_states = self._feed_forward(layer.feed_forward1, hidden_states)
        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = layer.norm_pre_attn(hidden_states)
        hidden_states = self._conformer_attention(layer.self_attn, hidden_states, position_embeddings, attention_mask)
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = layer.norm_post_attn(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self._light_conv(layer.lconv1d, hidden_states)
        hidden_states = self._feed_forward(layer.feed_forward2, hidden_states)
        hidden_states = torch.clamp(hidden_states, -gc, gc)
        hidden_states = layer.norm_out(hidden_states)
        return hidden_states

    def forward(self, input_features):
        at = self.audio_tower
        # 1. Subsample conv projection (no mask needed for export)
        hidden_states = input_features.unsqueeze(1)  # [B, 1, T, F]
        hidden_states = at.subsample_conv_projection.layer0.conv(hidden_states.to(at.subsample_conv_projection.layer0.conv.weight.dtype))
        hidden_states = at.subsample_conv_projection.layer0.act(
            at.subsample_conv_projection.layer0.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        )
        hidden_states = at.subsample_conv_projection.layer1.conv(hidden_states)
        hidden_states = at.subsample_conv_projection.layer1.act(
            at.subsample_conv_projection.layer1.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        )
        B, C, T, F = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous().reshape(B, T, -1)
        hidden_states = at.subsample_conv_projection.input_proj_linear(hidden_states)
        # 2. Relative positional encoding
        position_embeddings = at.rel_pos_enc(hidden_states)
        # 3. Build sliding window attention mask
        S = hidden_states.shape[1]
        attention_mask = self._build_blocked_mask(S, hidden_states.device)
        # 4. Encoder layers
        for layer in at.layers:
            hidden_states = self._encoder_layer(layer, hidden_states, position_embeddings, attention_mask)
        # 4. Output projection
        hidden_states = at.output_proj(hidden_states)
        # 5. Multimodal embedder
        audio_features = self.embed_audio(hidden_states)
        return audio_features


class Gemma4Audio(Audio):
    def __init__(self, audio, base):
        object.__setattr__(self, 'embed_audio_ref', base.embed_audio)
        super().__init__(audio, base)
        self.sampling_rate = 16000
        self.feature_size = 128
        self.audio_embeds = None

    def load(self):
        self.audio_tower = self.audio.float()
        self.embed_audio = self.embed_audio_ref.float()
        self.llm_config['is_audio'] = True
        self.llm_config['audio_pad'] = self.config.audio_token_id
        self.llm_config['audio_start'] = self.config.boa_token_id
        self.llm_config['audio_end'] = self.config.eoa_token_id
        self.llm_config['audio_type'] = 'usm'
        self.export_model = Gemma4AudioExportModel(self.audio_tower, self.embed_audio)

    def init_config(self):
        self.llm_config['is_audio'] = True

    def _extract_mel_features(self, audio_obj):
        """USM-style mel spectrogram extraction matching Gemma4AudioFeatureExtractor."""
        import numpy as np
        from transformers.audio_utils import mel_filter_bank, window_function
        waveform = audio_obj if isinstance(audio_obj, np.ndarray) else audio_obj.numpy()
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        frame_length = 320  # 20ms * 16000
        hop_length = 160    # 10ms * 16000
        fft_length = 512
        mel_floor = 0.001
        # Semicausal padding
        pad_left = frame_length // 2
        waveform = np.pad(waveform, ((0, 0), (pad_left, 0)), mode='constant')
        # Frame extraction (unfold)
        frame_size = frame_length + 1  # 321
        B, L = waveform.shape
        num_frames = (L - frame_size) // hop_length + 1
        strides = (waveform.strides[0], waveform.strides[1] * hop_length, waveform.strides[1])
        frames = np.lib.stride_tricks.as_strided(waveform, (B, num_frames, frame_size), strides)
        # No preemphasis (preemphasis=0), take first frame_length samples
        frames = frames[..., :-1]
        # Window
        window = window_function(frame_length).astype(np.float32)
        frames = frames * window
        # RFFT
        stft = np.fft.rfft(frames, n=fft_length, axis=-1)
        magnitude = np.abs(stft)
        # Mel filterbank
        mel_filters = mel_filter_bank(
            num_frequency_bins=fft_length // 2 + 1,
            num_mel_filters=128,
            min_frequency=0.0, max_frequency=8000.0,
            sampling_rate=16000, norm=None, mel_scale='htk',
        )
        mel_spec = np.matmul(magnitude, mel_filters)
        log_mel = np.log(mel_spec + mel_floor)
        return torch.from_numpy(log_mel.astype(np.float32))

    def forward(self, input_features):
        return self.export_model(input_features)

    def audio_process(self, audio_obj):
        input_features = self._extract_mel_features(audio_obj)  # [1, T, 128]
        with torch.no_grad():
            audio_embeds = self.forward(input_features)  # [1, T/4, hidden_size]
        self.audio_embeds = audio_embeds.permute(1, 0, 2)  # [T/4, 1, hidden_size]
        return self.audio_embeds.shape[0]

    def str_to_ids(self, prompt):
        if '<audio>' not in prompt:
            return self.tokenizer(prompt, return_tensors="pt")['input_ids']
        import re
        import librosa
        audio_pad_id = self.config.audio_token_id
        boa_token = self.tokenizer.decode([self.config.boa_token_id])
        eoa_token = self.tokenizer.decode([self.config.eoa_token_id])
        pad_token = self.tokenizer.decode([audio_pad_id])
        # Parse <audio> tags, process audio, and replace with placeholder tokens
        pattern = r'(<audio>.*?</audio>)'
        parts = re.split(pattern, prompt)
        txt_prompt = ''
        for part in parts:
            if re.match(pattern, part):
                audio_path = re.search(r'<audio>(.*?)</audio>', part).group(1)
                audio_obj = librosa.load(audio_path, sr=self.sampling_rate)[0]
                num_tokens = self.audio_process(audio_obj)
                txt_prompt += boa_token + pad_token * num_tokens + eoa_token
            else:
                txt_prompt += part
        input_ids = self.tokenizer(txt_prompt, return_tensors="pt", add_special_tokens=False)['input_ids']
        return input_ids

    def embed(self, input_ids, images=None, videos=None):
        input_embeds = self.embed_(input_ids)
        if self.audio_embeds is not None:
            audio_pad_id = self.config.audio_token_id
            audio_mask = (input_ids == audio_pad_id).squeeze()
            input_embeds[audio_mask] = self.audio_embeds.to(input_embeds.dtype)
        return input_embeds

    @spinner_run(f'export audio to ')
    def export(self, onnx_path):
        input_features = torch.randn((1, 600, self.feature_size))
        model = self.export_model.float().eval()
        onnx_model = f'{onnx_path}/audio.onnx'
        onnx_export(model, (input_features,),
                    onnx_model,
                    input_names=['input_features'],
                    output_names=['audio_embeds'],
                    dynamic_axes={"input_features": {1: "seq_len"}})
        return onnx_model