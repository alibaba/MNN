import torch

class Audio(torch.nn.Module):
    def __init__(self, audio, base):
        super().__init__()
        self.audio = audio
        self.embed_ = base.embed
        self.tokenizer = base.tokenizer
        self.config = base.config
        self.hidden_size = base.hidden_size
        self.llm_config = base.llm_config
        self.quant_bit = 16
        self.init_config()
        self.load()

    @staticmethod
    def get_audio(model_type):
        audio_models = {
            'qwen2_audio': Qwen2Audio,
        }
        if model_type in audio_models:
            return audio_models[model_type]
        return None

    def init_config(self):
        self.llm_config['is_audio'] = True

    def load(self):
        raise NotImplementedError

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        return input_ids

    def forward(self, images):
        raise NotImplementedError

    def embed(self, input_ids, images = None, videos = None):
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
        self.audio_tower = self.audio.audio_tower
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