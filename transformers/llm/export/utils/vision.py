import math
import torch
import numpy as np

from .transformers import VisionRotary, Decoder

class Vision(torch.nn.Module):
    def __init__(self, visual, base):
        super().__init__()
        self.model_type = base.model_type
        self.visual = visual.eval()
        self.embed_ = base.embed
        self.tokenizer = base.tokenizer
        self.config = base.config
        self.hidden_size = base.hidden_size
        self.llm_config = base.llm_config
        # mllama
        self.cross_attention_states = None
        self.cross_attention_mask = None
        self.init_config()
        self.load()

    @staticmethod
    def get_vision(model_type):
        visual_models = {
            'qwen': QwenVision,
            'qwen2_vl': Qwen2Vision,
            'mllama': MllamaVision
        }
        if model_type in visual_models:
            return visual_models[model_type]
        return None

    def init_config(self):
        from transformers.image_utils import (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        self.llm_config['is_visual'] = True
        image_mean = np.array(OPENAI_CLIP_MEAN) * 255.0
        image_norm = 1 / (np.array(OPENAI_CLIP_STD) * 255.0)
        self.llm_config['image_mean'] = image_mean.tolist()
        self.llm_config['image_norm'] = image_norm.tolist()

    def export(self, onnx_path):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        return input_ids

    def forward(self, images):
        raise NotImplementedError

    def embed(self, input_ids, images = None, videos = None):
        raise NotImplementedError

class QwenVision(Vision):
    def __init__(self, visual, base):
        self.quant_bit = 16
        super().__init__(visual, base)

    def load(self):
        self.image_start_id = self.config.visual['image_start_id']
        self.image_size = self.config.visual['image_size']
        self.llm_config['is_visual'] = True
        self.llm_config['image_size'] = self.image_size
        self.llm_config['vision_start'] = self.tokenizer.img_start_id
        self.llm_config['vision_end'] = self.tokenizer.img_end_id
        self.llm_config['image_pad'] = self.tokenizer.img_pad_id

    def export(self, onnx_path):
        input_images = torch.randn((1, 3, self.image_size, self.image_size))
        onnx_model = f'{onnx_path}/visual.onnx'
        torch.onnx.export(self, (input_images),
                        onnx_model,
                        input_names=['input_images'],
                        output_names=['image_embeds'],
                        dynamic_axes={
                            "input_images": { 0: "size" },
                        },
                        do_constant_folding=True,
                        verbose=False,
                        opset_version=15)
        return onnx_model

    def forward(self, images):
        return self.visual(images).transpose(1, 0)

    def embed(self, input_ids, images = None, videos = None):
        if not torch.any(input_ids == self.image_start_id):
            return self.embed_(input_ids)
        bos_pos = torch.where(input_ids == self.image_start_id)
        eos_pos = torch.where(input_ids == self.image_start_id + 1)
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        images = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1 : b - 1].tolist()
            image = image[ : image.index(self.image_start_id + 2)]
            images.append(bytes(image).decode('utf-8'))
        images = self.visual.encode(images).transpose(1, 0)
        hidden_states = self.embed_(input_ids)
        for idx, (i, a, b) in enumerate(img_pos):
            hidden_states[a + 1 : b, i] = images[:, idx]
        return hidden_states

class Qwen2Vision(Vision):
    def __init__(self, visual, base):
        self.quant_bit = 4
        self.temporal_patch_size = 2
        self.patch_size = 14
        self.merge_size = 2
        self.image_height = 420
        self.image_width = 420
        self.image_embeds = None
        super().__init__(visual, base)

    def load(self):
        self.vision_start_id = self.config.vision_start_token_id
        self.vision_end_id = self.config.vision_end_token_id
        self.image_pad_id = self.config.image_token_id
        self.llm_config['image_size'] = self.image_height
        self.llm_config['vision_start'] = self.vision_start_id
        self.llm_config['vision_end'] = self.vision_end_id
        self.llm_config['image_pad'] = self.image_pad_id
        # load model
        config = self.visual.config
        self.hidden_size = config.embed_dim
        self.num_attention_heads = config.num_heads
        self.num_key_value_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rope_theta = 10000.0
        self.rotary_dim = self.head_dim // 2
        self.rotary = VisionRotary(self)
        self.model_map = {
            'decoder': {
                'self_attn': 'attn',
                'mlp': 'mlp',
                'input_layernorm': 'norm1',
                'post_attention_layernorm': 'norm2'
            },
            'attention': {
                'qkv_proj': 'qkv',
                'o_proj': 'proj'
            }
        }
        self.patch_embed = self.visual.patch_embed
        self.blocks = []
        for block in self.visual.blocks.children():
            layer_id = len(self.blocks)
            self.blocks.append(Decoder(block, layer_id, self))
        self.merger = self.visual.merger

    def str_to_ids(self, prompt):
        if '<img>' in prompt and '</img>' in prompt:
            import re
            import requests
            from PIL import Image
            pattern = r'(<img>.*?</img>)'
            parts = re.split(pattern, prompt)
            txt_prompt = ''
            for part in parts:
                if re.match(pattern, part):
                    img_content = re.search(r'<img>(.*?)</img>', part).group(1)
                    # find <hw></hw> in image_content
                    match = re.search(r'<hw>(.*?)</hw>', img_content)
                    img_content = img_content[:match.start()] + img_content[match.end():]
                    hw = match.group(1).split(',')
                    self.image_height, self.image_width = int(hw[0]), int(hw[1])
                    if img_content.startswith('http://') or img_content.startswith('https://'):
                        image_obj = Image.open(requests.get(img_content, stream=True).raw)
                    img_pad_len = self.img_process(image_obj)
                    img_pad_str = '<|image_pad|>' * img_pad_len
                    img_str = f'<|vision_start|>{img_pad_str}<|vision_end|>'
                    txt_prompt += img_str
                else:
                    txt_prompt += part
        else:
            txt_prompt = prompt
        input_ids = self.tokenizer(txt_prompt, return_tensors="pt")['input_ids']
        return input_ids

    def forward(self, flatten_patches, position_ids, attention_mask):
        rotary_pos_emb = self.rotary(position_ids)
        hidden_states = self.patch_embed(flatten_patches)
        if rotary_pos_emb.dtype != hidden_states.dtype:
            rotary_pos_emb = rotary_pos_emb.to(hidden_states.dtype)
        for blk in self.blocks:
            hidden_states, _ = blk(hidden_states, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        image_embeds = self.merger(hidden_states)
        image_embeds = image_embeds.unsqueeze(1)
        return image_embeds

    def images_forward(self, images):
        images = [images] * self.temporal_patch_size
        patches = torch.concat(images, axis=0)
        _, channel, height, width = patches.shape
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])
        pos_ids = []
        for t, h, w in image_grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.merge_size,
                self.merge_size,
                w // self.merge_size,
                self.merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.merge_size,
                self.merge_size,
                w // self.merge_size,
                self.merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids]))
        position_ids = torch.cat(pos_ids, dim=0)
        seq_len = grid_t * grid_h * grid_w
        attention_mask = torch.zeros([1, seq_len, seq_len], dtype=torch.float)
        return self.forward(flatten_patches, position_ids, attention_mask)

    def smart_resize(self, height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def img_process(self, image):
        from transformers.image_transforms import (
            convert_to_rgb,
            resize,
            rescale,
            normalize
        )
        from transformers.image_utils import (
            OPENAI_CLIP_MEAN,
            OPENAI_CLIP_STD,
            PILImageResampling,
            infer_channel_dimension_format,
            to_numpy_array
        )
        image = convert_to_rgb(image)
        image = to_numpy_array(image)
        resized_height, resized_width = self.smart_resize(self.image_height, self.image_width)
        format = infer_channel_dimension_format(image)
        resample = PILImageResampling.BICUBIC
        image = resize(image, size=(resized_height, resized_width), resample=resample, input_data_format=format)
        image = rescale(image, scale=1 / 255.0, input_data_format=format)
        image = normalize(image=image, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, input_data_format=format)
        image = np.expand_dims(image, [0])
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        self.image_embeds = self.images_forward(image)
        return self.image_embeds.shape[0]

    def embed(self, input_ids, images = None, videos = None):
        input_embeds = self.embed_(input_ids)
        if self.image_embeds is not None:
            image_mask = (input_ids == self.image_pad_id).squeeze()
            input_embeds[image_mask] = self.image_embeds
        return input_embeds

    def export(self, onnx_path):
        patch = torch.randn([900, 1176])
        posision_ids = torch.zeros([2, 900], dtype=torch.int32)
        attention_mask = torch.zeros([1, 900, 900], dtype=torch.float)
        onnx_model = f'{onnx_path}/visual.onnx'
        torch.onnx.export(self, (patch, posision_ids, attention_mask),
                        onnx_model,
                        input_names=['patches', 'position_ids', 'attention_mask'],
                        output_names=['image_embeds'],
                        dynamic_axes={
                            "patches": { 0: "size" },
                            "position_ids": { 1: "size" },
                            "attention_mask": { 1: "size", 2: "size" }
                        },
                        do_constant_folding=True,
                        verbose=False,
                        opset_version=15)
        return onnx_model

class MllamaVision(Vision):
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.multi_modal_projector = base.multi_modal_projector
        self.image_objs = []

    def load(self):
        self.llm_config['is_visual'] = True
        self.llm_config['image_size'] = self.config.vision_config.image_size
        self.image_size = self.config.vision_config.image_size

    def str_to_ids(self, prompt):
        if '<img>' in prompt and '</img>' in prompt:
            import re
            import requests
            from PIL import Image
            pattern = r'(<img>.*?</img>)'
            parts = re.split(pattern, prompt)
            txt_prompt = ''
            for part in parts:
                if re.match(pattern, part):
                    img_content = re.search(r'<img>(.*?)</img>', part).group(1)
                    if img_content.startswith('http://') or img_content.startswith('https://'):
                        self.image_objs.append(Image.open(requests.get(img_content, stream=True).raw))
                    txt_prompt += '<|image|>'
                else:
                    txt_prompt += part
        else:
            txt_prompt = prompt
        input_ids = self.tokenizer(txt_prompt, return_tensors="pt")['input_ids']
        # image process
        for img in self.image_objs:
            self.img_process(img)
        return input_ids

    def img_process(self, image):
        self.image_size = 560
        resized_height = self.image_size
        resized_width = self.image_size
        from transformers.image_transforms import (
            convert_to_rgb,
            resize,
            rescale,
            normalize
        )
        from transformers.image_utils import (
            OPENAI_CLIP_MEAN,
            OPENAI_CLIP_STD,
            PILImageResampling,
            infer_channel_dimension_format,
            to_numpy_array
        )
        image = convert_to_rgb(image)
        image = to_numpy_array(image)
        format = infer_channel_dimension_format(image)
        resample = PILImageResampling.BICUBIC
        image = resize(image, size=(resized_height, resized_width), resample=resample, input_data_format=format)
        image = rescale(image, scale=1 / 255.0, input_data_format=format)
        image = normalize(image=image, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, input_data_format=format)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, [0, 1, 2])
        pad_val = np.zeros_like(image)
        image = np.concatenate([image, pad_val, pad_val, pad_val], axis=2)
        image = torch.from_numpy(image)
        self.cross_attention_states = self.forward(image)

    def forward(self, images):
        aspect_ratio_ids = torch.tensor([[1]])
        aspect_ratio_mask = torch.tensor([[[1, 0, 0, 0]]])
        vision_outputs = self.visual(images, aspect_ratio_ids, aspect_ratio_mask)
        cross_attention_states = vision_outputs[0]
        cross_attention_states = cross_attention_states.type(self.multi_modal_projector.weight.dtype)
        cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size)
        return cross_attention_states

    def embed(self, input_ids, images = None, videos = None):
        txt_embeds = self.embed_(input_ids)
        return txt_embeds