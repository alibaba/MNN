import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from typing import Tuple, Optional, Dict, Any

from .transformers import VisionRotary, Decoder
from .spinner import spinner_run
from .torch_utils import onnx_export

class Vision(torch.nn.Module):
    def __init__(self, visual, base):
        super().__init__()
        self.quant_bit = 8
        self.quant_block = 128
        self.transformer_fuse = True
        self.group_conv_native = False
        self.model_type = base.config.model_type
        self.visual = visual.eval()
        self.embed_ = base.embed
        self.tokenizer = base.tokenizer
        self.config = base.config.origin_config
        self.hidden_size = base.config.hidden_size
        self.llm_config = { "is_visual": True }
        self.rope_ratio = 1.0
        self.init_config()
        self.load()

    def get_config(self):
        return self.llm_config

    @staticmethod
    def get_vision(model_type):
        visual_models = {
            'deepseek-vl': DeepSeekVL,
            'internvl_chat': InternVLVision,
            'qwen': QwenVision,
            'qwen2_vl': Qwen2Vision,
            'qwen2_5_vl':Qwen2_5Vision,
            'qwen2_5_omni': Qwen2_5OmniVision,
            'qwen3_vl': Qwen3Vision,
            'qwen3_vl_moe': Qwen3Vision,
            'qwen3_5': Qwen3_5Vision,
            'qwen3_5_moe': Qwen3_5Vision,
            'gemma3': Gemma3Vision,
            'idefics3': Idefics3Vision,
            'smolvlm': Idefics3Vision,
            'llava_qwen2': MobileCLIPVision,
            'minicpmv': MiniCPMVision,
        }
        if model_type in visual_models:
            return visual_models[model_type]
        return None

    def init_config(self):
        from transformers.image_utils import (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        self.norm_mean = OPENAI_CLIP_MEAN
        self.norm_std = OPENAI_CLIP_STD
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
        return self.embed_(input_ids)

    def deepstacks(self):
        return None

class DeepSeekVL(Vision):
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.quant_bit = 8
        self.aligner = base.model.aligner
        self.vision_model = visual

    def load(self):
        self.image_size = 1024
        self.llm_config['is_visual'] = True
        self.llm_config['image_size'] = self.image_size
        # self.llm_config['vision_start'] = self.tokenizer.img_start_id
        # self.llm_config['vision_end'] = self.tokenizer.img_end_id
        # self.llm_config['image_pad'] = self.tokenizer.img_pad_id
    def init_config(self):
        self.llm_config['is_visual'] = True
        IMAGENET_MEAN = [0.0, 0.0, 0.0]
        IMAGENET_STD = [1.0, 1.0, 1.0]
        for i in range(3):
            IMAGENET_MEAN[i] = IMAGENET_MEAN[i] * 255.0
            IMAGENET_STD[i] = 1.0 / IMAGENET_STD[i] / 255.0
        self.llm_config['image_mean'] = IMAGENET_MEAN
        self.llm_config['image_norm'] = IMAGENET_STD
        self.llm_config['image_size_unit'] = 14
    def export(self, onnx_path):
        input_images = torch.randn((1, 3, self.image_size, self.image_size), dtype=torch.float32)
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (input_images),
                    onnx_model,
                    input_names=['input_images'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "input_images": { 0: "size", 2: "height", 3: "width"},
                    })
        return onnx_model
    def forward(self, images):
        vit_embeds = self.aligner(self.vision_model(images))
        # For mnn's embedding, the order is (seq, batch, hidden)
        vit_embeds = vit_embeds.permute(1, 0, 2)
        return vit_embeds


class InternVLVision(Vision):
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.quant_bit = 8
        self.vision_model = visual
        self.mlp1 = visual.mlp1
        self.select_layer = visual.select_layer

    def load(self):
        self.image_size = self.config.force_image_size
        self.downsample_ratio = self.config.downsample_ratio
        self.llm_config['is_visual'] = True
        self.llm_config['image_size'] = self.image_size
        # self.llm_config['vision_start'] = self.tokenizer.img_start_id
        # self.llm_config['vision_end'] = self.tokenizer.img_end_id
        # self.llm_config['image_pad'] = self.tokenizer.img_pad_id

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, (h * scale_factor).int(), (c / scale_factor).int())
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, (h * scale_factor).int(), (w * scale_factor).int(),
                   (c / (scale_factor * scale_factor)).int())
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = (vit_embeds.shape[1] ** 0.5).int()
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)

        # For mnn's embedding, the order is (seq, batch, hidden)
        vit_embeds = vit_embeds.permute(1, 0, 2)
        return vit_embeds

    def init_config(self):
        self.llm_config['is_visual'] = True
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        for i in range(3):
            IMAGENET_MEAN[i] = IMAGENET_MEAN[i] * 255.0
            IMAGENET_STD[i] = 1.0 / IMAGENET_STD[i] / 255.0
        self.llm_config['image_mean'] = IMAGENET_MEAN
        self.llm_config['image_norm'] = IMAGENET_STD
        self.llm_config['image_size_unit'] = 14

    def export(self, onnx_path):
        input_images = torch.randn((1, 3, self.image_size, self.image_size), dtype=torch.float32)
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (input_images),
                    onnx_model,
                    input_names=['input_images'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "input_images": { 0: "size", 2: "height", 3: "width"},
                    })
        return onnx_model

    def forward(self, images):
        return self.extract_feature(images)

class QwenVision(Vision):
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.quant_bit = 16

    def load(self):
        self.image_start_id = self.config.visual['image_start_id']
        self.image_size = self.config.visual['image_size']
        self.llm_config['is_visual'] = True
        self.llm_config['image_size'] = self.image_size
        self.llm_config['vision_start'] = self.tokenizer.img_start_id
        self.llm_config['vision_end'] = self.tokenizer.img_end_id
        self.llm_config['image_pad'] = self.tokenizer.img_pad_id

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        input_images = torch.randn((1, 3, self.image_size, self.image_size))
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (input_images),
                    onnx_model,
                    input_names=['input_images'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "input_images": { 0: "size" },
                    })
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
        self.temporal_patch_size = 2
        self.patch_size = 14
        self.merge_size = 2
        self.image_height = 420
        self.image_width = 420
        self.min_pixels = 3136
        self.max_pixels = 12845056
        self.image_embeds = []
        self.image_grid_thw = []
        super().__init__(visual, base)
        self.quant_bit = 4

    def load(self):
        self.vision_start_id = self.config.vision_start_token_id
        self.vision_end_id = self.config.vision_end_token_id
        self.image_pad_id = self.config.image_token_id
        self.llm_config['image_size'] = self.image_height
        self.llm_config['vision_start'] = self.vision_start_id
        self.llm_config['vision_end'] = self.vision_end_id
        self.llm_config['image_pad'] = self.image_pad_id
        self.vision_start_token = '<|vision_start|>'
        self.vision_end_token = '<|vision_end|>'
        self.image_pad_token = '<|image_pad|>'
        # load model
        config = self.visual.config
        if hasattr(config, "embed_dim"):
            self.hidden_size = config.embed_dim
        else:
            self.hidden_size = config.hidden_size
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
                    if match:
                        img_content = img_content[:match.start()] + img_content[match.end():]
                        hw = match.group(1).split(',')
                        self.image_height, self.image_width = int(hw[0]), int(hw[1])
                    if img_content.startswith('http://') or img_content.startswith('https://'):
                        image_obj = Image.open(requests.get(img_content, stream=True).raw)
                    else:
                        image_obj = Image.open(img_content)
                    img_pad_len = self.img_process(image_obj)
                    img_pad_str = self.image_pad_token * img_pad_len
                    img_str = f'{self.vision_start_token}{img_pad_str}{self.vision_end_token}'
                    txt_prompt += img_str
                else:
                    txt_prompt += part
        else:
            txt_prompt = prompt
        input_ids = self.tokenizer(txt_prompt, return_tensors="pt")['input_ids']
        return input_ids

    def get_position_ids(self, input_ids, seq_len, token_len):
        if token_len:
            position_ids = torch.tensor([[seq_len - 1]] * 3, dtype=torch.int)
            return position_ids
        input_ids = input_ids.flatten()
        txt_len, vision_idx, cur_idx = 0, 0, 0
        position_ids_list = []
        for i, token in enumerate(input_ids):
            if token != self.image_pad_id:
                txt_len += 1
            if token == self.vision_start_id:
                text_index = torch.arange(cur_idx, cur_idx + txt_len, dtype=torch.int)
                cur_idx += txt_len
                txt_len = 0
                position_ids_list.append(torch.stack([text_index, text_index, text_index]))
            elif token == self.vision_end_id:
                t, h, w = self.image_grid_thw[vision_idx]
                h = h // self.merge_size
                w = w // self.merge_size
                t_index = torch.arange(t).view(-1, 1).expand(-1, h * w).flatten()
                h_index = torch.arange(h).view(1, -1, 1).expand(t, -1, w).flatten()
                w_index = torch.arange(w).view(1, 1, -1).expand(t, h, -1).flatten()
                position_ids_list.append(torch.stack([t_index, h_index, w_index]) + cur_idx)
                cur_idx += w
                vision_idx += 1
        if txt_len > 0:
            text_index = torch.arange(cur_idx, cur_idx + txt_len, dtype=torch.int)
            position_ids_list.append(torch.stack([text_index, text_index, text_index]))
        position_ids = torch.cat(position_ids_list, dim=1)
        return position_ids

    def vision_position_ids(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            llm_h, llm_w = h // self.merge_size, w // self.merge_size
            # compute pos_ids
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(llm_h, self.merge_size, llm_w, self.merge_size)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(llm_h, self.merge_size, llm_w, self.merge_size)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids]))
        position_ids = torch.cat(pos_ids, dim=0)
        return position_ids

    def vision_attention_mask(self, grid_thw, cu_window_seqlens = None):
        seq_len = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        if cu_window_seqlens is None:
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        else:
            cu_seqlens = cu_window_seqlens
        attention_mask = torch.full([1, seq_len, seq_len], torch.finfo(torch.float32).min)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    def vision_reshape(self, images):
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
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])
        self.image_grid_thw.append([grid_t, grid_h, grid_w])
        return flatten_patches, grid_thw

    def images_forward(self, images):
        flatten_patches, grid_thw = self.vision_reshape(images)
        position_ids = self.vision_position_ids(grid_thw)
        attention_mask = self.vision_attention_mask(grid_thw)
        return self.forward(flatten_patches, position_ids, attention_mask)

    def forward(self, flatten_patches, position_ids, attention_mask):
        rotary_pos_emb = self.rotary(position_ids)
        hidden_states = self.patch_embed(flatten_patches)
        if rotary_pos_emb.dtype != hidden_states.dtype:
            rotary_pos_emb = rotary_pos_emb.to(hidden_states.dtype)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        image_embeds = self.merger(hidden_states)
        image_embeds = image_embeds.unsqueeze(1)
        return image_embeds

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
            PILImageResampling,
            infer_channel_dimension_format,
            to_numpy_array
        )
        image = convert_to_rgb(image)
        image = to_numpy_array(image)
        resized_height, resized_width = self.smart_resize(self.image_height, self.image_width, self.patch_size * self.merge_size, self.min_pixels, self.max_pixels)
        format = infer_channel_dimension_format(image)
        resample = PILImageResampling.BICUBIC
        image = resize(image, size=(resized_height, resized_width), resample=resample, input_data_format=format)
        image = rescale(image, scale=1 / 255.0, input_data_format=format)
        image = normalize(image=image, mean=self.norm_mean, std=self.norm_std, input_data_format=format)
        image = np.expand_dims(image, [0])
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image_embed = self.images_forward(image)
        self.image_embeds.append(image_embed)
        return image_embed.shape[0]

    def embed(self, input_ids, images = None, videos = None):
        input_embeds = self.embed_(input_ids)
        if self.image_embeds is not None and len(self.image_embeds) > 0:
            image_mask = (input_ids == self.image_pad_id).squeeze()
            input_embeds[image_mask] = torch.concat(self.image_embeds, dim=0).to(input_embeds.dtype)
        return input_embeds

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        patch = torch.randn([900, 1176])
        posision_ids = torch.zeros([2, 900], dtype=torch.int32)
        attention_mask = torch.zeros([1, 900, 900], dtype=torch.float)
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (patch, posision_ids, attention_mask),
                    onnx_model,
                    input_names=['patches', 'position_ids', 'attention_mask'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "patches": { 0: "size" },
                        "position_ids": { 1: "size" },
                        "attention_mask": { 1: "size", 2: "size" }
                    })
        return onnx_model

class Gemma3Vision(Vision):
    def __init__(self, visual, base):
        # read from gemma3_map
        self.image_size = base.image_size
        # embedding functions
        super().__init__(visual, base)
        self.quant_bit = 8
        self.vision_tower = base.vision_tower
        self.multi_modal_projector = base.multi_modal_projector.float()

    def init_config(self):
        self.image_mean_from_preprcessor_config = [0.5, 0.5, 0.5]
        self.image_std_from_preprcessor_config = [0.5, 0.5, 0.5]
        for i in range(3):
            self.image_mean_from_preprcessor_config[i] = self.image_mean_from_preprcessor_config[i] * 255.0
            self.image_std_from_preprcessor_config[i] = 1.0 / self.image_std_from_preprcessor_config[i] / 255.0
        self.llm_config['is_visual'] = True
        self.llm_config['image_mean'] = self.image_mean_from_preprcessor_config
        self.llm_config['image_norm'] = self.image_std_from_preprcessor_config
        self.llm_config['vision_start'] = self.config.boi_token_index
        self.llm_config['vision_end'] = self.config.eoi_token_index
        self.llm_config['image_pad'] = self.config.image_token_index

    def load(self):
        self.llm_config['image_size'] = self.image_size

    def forward(self, pixel_values):
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        image_features_transpose = image_features.permute(1, 0, 2)
        return image_features_transpose

    def export(self, onnx_path):
        input_images = torch.randn((1, 3, self.image_size, self.image_size))
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (input_images),
                    onnx_model,
                    input_names=['input_images'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "input_images": { 0: "size", 2: "height", 3: "width"},
                    })
        return onnx_model

    def embed(self, input_ids):
        txt_embeds = self.embed_(input_ids)
        return txt_embeds

class Qwen2_5Vision(Qwen2Vision):
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.merge_unit = self.merge_size * self.merge_size
        self.window_size = visual.window_size
        self.fullatt_block_indexes = visual.fullatt_block_indexes

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.merge_size,
                grid_w // self.merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def images_forward(self, images):
        flatten_patches, grid_thw = self.vision_reshape(images)
        position_ids = self.vision_position_ids(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        normal_attention_mask = self.vision_attention_mask(grid_thw)
        fullatt_attention_mask = self.vision_attention_mask(grid_thw, cu_window_seqlens)
        attention_mask = torch.stack([normal_attention_mask, fullatt_attention_mask], dim=0)
        return self.forward(flatten_patches, position_ids, attention_mask, window_index)

    def forward(self, flatten_patches, position_ids, attention_mask, window_index):
        hidden_states = self.patch_embed(flatten_patches)
        seq_len, _ = hidden_states.size()
        position_ids = position_ids.reshape(2, seq_len // self.merge_unit, self.merge_unit)
        position_ids = position_ids[:, window_index, :]
        position_ids = position_ids.reshape(2, seq_len)
        rotary_pos_emb = self.rotary(position_ids)
        if rotary_pos_emb.dtype != hidden_states.dtype:
            rotary_pos_emb = rotary_pos_emb.to(hidden_states.dtype)
        hidden_states = hidden_states.reshape(seq_len // self.merge_unit, self.merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = attention_mask[0]
            else:
                attention_mask_now = attention_mask[1]
            hidden_states = blk(hidden_states, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask_now)
        image_embeds = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        image_embeds = image_embeds[reverse_indices, :]
        image_embeds = image_embeds.unsqueeze(1)
        return image_embeds

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        patch = torch.randn([400, 1176])
        posision_ids = torch.zeros([2, 400], dtype=torch.int32)
        attention_mask = torch.zeros([2, 1, 400, 400], dtype=torch.float)
        window_index = torch.arange(100, dtype=torch.int32)
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (patch, posision_ids, attention_mask, window_index),
                    onnx_model,
                    input_names=['patches', 'position_ids', 'attention_mask', 'window_index'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "patches": { 0: "size" },
                        "position_ids": { 1: "size" },
                        "attention_mask": { 2: "size", 3: "size" },
                        "window_index": { 0: "size" }
                    })
        return onnx_model

class Qwen2_5OmniVision(Qwen2_5Vision):
    def __init__(self, visual, base):
        self.temporal_patch_size = 2
        self.patch_size = 14
        self.merge_size = 2
        self.image_height = 420
        self.image_width = 420
        self.image_embeds = None
        super().__init__(visual, base)
        self.quant_bit = 8

    def load(self):
        self.config = self.config.thinker_config
        self.vision_start_id = self.config.vision_start_token_id
        self.vision_end_id = self.config.vision_end_token_id
        self.image_pad_id = self.config.image_token_index
        self.llm_config['image_size'] = self.image_height
        self.llm_config['vision_start'] = self.vision_start_id
        self.llm_config['vision_end'] = self.vision_end_id
        self.llm_config['image_pad'] = self.image_pad_id
        self.vision_start_token = '<|vision_bos|>'
        self.vision_end_token = '<|vision_eos|>'
        self.image_pad_token = '<|IMAGE|>'
        # load model
        config = self.visual.config
        if hasattr(config, "embed_dim"):
            self.hidden_size = config.embed_dim
        else:
            self.hidden_size = config.hidden_size
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
                'q_proj': 'q',
                'k_proj': 'k',
                'v_proj': 'v',
                'o_proj': 'proj'
            }
        }
        self.patch_embed = self.visual.patch_embed
        self.blocks = []
        for block in self.visual.blocks.children():
            layer_id = len(self.blocks)
            self.blocks.append(Decoder(block, layer_id, self))
        self.merger = self.visual.merger

class Qwen3Vision(Qwen2Vision):
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.patch_size = 16
        self.image_height = 480
        self.image_width = 480

        self.image_height = 256
        self.image_width = 256

        self.min_pixels = 65536
        self.max_pixels = 16777216
        self.merge_unit = self.merge_size * self.merge_size
        self.deepstack_visual_indexes = visual.deepstack_visual_indexes
        self.num_grid_per_side = visual.num_grid_per_side
        self.pos_embed = visual.pos_embed
        self.deepstack_merger_list = visual.deepstack_merger_list

        # deepstack
        self.deepstack_feature_list = []
        self.deepstack_embeds = None
        self.norm_mean = self.norm_std = [0.5, 0.5, 0.5]
        image_mean = np.array(self.norm_mean) * 255.0
        image_norm = 1 / (np.array(self.norm_std) * 255.0)
        self.llm_config['image_mean'] = image_mean.tolist()
        self.llm_config['image_norm'] = image_norm.tolist()
        self.llm_config['num_grid_per_side'] = self.num_grid_per_side
        self.llm_config['has_deepstack'] = True

    def get_idx_weight(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device)
        merge_size = self.merge_size
        idx_tensor = idx_tensor.repeat(1, t)
        idx_tensor = idx_tensor.view(4, t, h // merge_size, merge_size, w // merge_size, merge_size).permute(0, 1, 2, 4, 3, 5).reshape(4, -1)
        weight_tensor = weight_tensor.repeat(1, t)
        weight_tensor = weight_tensor.view(4, t, h // merge_size, merge_size, w // merge_size, merge_size).permute(0, 1, 2, 4, 3, 5).reshape(4, -1)
        return idx_tensor, weight_tensor

    def embed(self, input_ids, images = None, videos = None):
        input_embeds = self.embed_(input_ids)
        if self.image_embeds is not None and len(self.image_embeds) > 0:
            image_mask = (input_ids == self.image_pad_id).squeeze()
            input_embeds[image_mask] = torch.concat(self.image_embeds, dim=0).to(input_embeds.dtype)
            # deepsatck_embeds
            self.deepstack_embeds = torch.zeros_like(input_embeds).transpose(0, 1).repeat(3, 1, 1)
            self.deepstack_embeds[:, image_mask, :] = torch.concat(self.deepstack_feature_list, dim=1)
        return input_embeds

    def deepstacks(self):
        deepstack_embeds = self.deepstack_embeds
        self.deepstack_feature_list = []
        self.deepstack_embeds = None
        return deepstack_embeds

    def images_forward(self, images):
        flatten_patches, grid_thw = self.vision_reshape(images)
        idx_tensor, weight_tensor = self.get_idx_weight(grid_thw)
        position_ids = self.vision_position_ids(grid_thw)
        attention_mask = self.vision_attention_mask(grid_thw)
        image_embeds, deepstack_feature = self.forward(flatten_patches, position_ids, attention_mask, idx_tensor, weight_tensor)
        self.deepstack_feature_list.append(deepstack_feature)
        return image_embeds

    def forward(self, flatten_patches, position_ids, attention_mask, idx_tensor, weight_tensor):
        rotary_pos_emb = self.rotary(position_ids)
        hidden_states = self.patch_embed(flatten_patches)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor.unsqueeze(2)
        pos_embeds = torch.sum(pos_embeds, 0, False)
        hidden_states = hidden_states + pos_embeds
        if rotary_pos_emb.dtype != hidden_states.dtype:
            rotary_pos_emb = rotary_pos_emb.to(hidden_states.dtype)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        image_embeds = self.merger(hidden_states)
        image_embeds = image_embeds.unsqueeze(1)
        deepstack_feature = torch.stack(deepstack_feature_lists)
        return image_embeds, deepstack_feature

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        patch = torch.randn([256, 1536])
        posision_ids = torch.zeros([2, 256], dtype=torch.int32)
        attention_mask = torch.zeros([1, 256, 256], dtype=torch.float)
        idx_tensor = torch.zeros([4, 256], dtype=torch.int32)
        weight_tensor = torch.randn([4, 256])
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (patch, posision_ids, attention_mask, idx_tensor, weight_tensor),
                    onnx_model,
                    input_names=['patches', 'position_ids', 'attention_mask', 'idx_tensor', 'weight_tensor'],
                    output_names=['image_embeds', 'deepstack_feature'],
                    dynamic_axes={
                        "patches": { 0: "size" },
                        "position_ids": { 1: "size" },
                        "attention_mask": { 1: "size", 2: "size" },
                        "idx_tensor": { 1: "size" },
                        "weight_tensor": { 1: "size" }
                    })
        return onnx_model

class Qwen3_5Vision(Qwen2Vision):
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.patch_size = 16
        self.image_height = 480
        self.image_width = 480

        self.image_height = 256
        self.image_width = 256

        self.min_pixels = 65536
        self.max_pixels = 16777216
        self.merge_unit = self.merge_size * self.merge_size
        self.num_grid_per_side = visual.num_grid_per_side
        self.pos_embed = visual.pos_embed
        self.norm_mean = self.norm_std = [0.5, 0.5, 0.5]
        image_mean = np.array(self.norm_mean) * 255.0
        image_norm = 1 / (np.array(self.norm_std) * 255.0)
        self.llm_config['image_mean'] = image_mean.tolist()
        self.llm_config['image_norm'] = image_norm.tolist()
        self.llm_config['num_grid_per_side'] = self.num_grid_per_side
        self.llm_config['has_deepstack'] = True

    def get_idx_weight(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device)
        merge_size = self.merge_size
        idx_tensor = idx_tensor.repeat(1, t)
        idx_tensor = idx_tensor.view(4, t, h // merge_size, merge_size, w // merge_size, merge_size).permute(0, 1, 2, 4, 3, 5).reshape(4, -1)
        weight_tensor = weight_tensor.repeat(1, t)
        weight_tensor = weight_tensor.view(4, t, h // merge_size, merge_size, w // merge_size, merge_size).permute(0, 1, 2, 4, 3, 5).reshape(4, -1)
        return idx_tensor, weight_tensor

    def embed(self, input_ids, images = None, videos = None):
        input_embeds = self.embed_(input_ids)
        if self.image_embeds is not None and len(self.image_embeds) > 0:
            image_mask = (input_ids == self.image_pad_id).squeeze()
            input_embeds[image_mask] = torch.concat(self.image_embeds, dim=0).to(input_embeds.dtype)
        return input_embeds

    def images_forward(self, images):
        flatten_patches, grid_thw = self.vision_reshape(images)
        idx_tensor, weight_tensor = self.get_idx_weight(grid_thw)
        position_ids = self.vision_position_ids(grid_thw)
        attention_mask = self.vision_attention_mask(grid_thw)
        image_embeds = self.forward(flatten_patches, position_ids, attention_mask, idx_tensor, weight_tensor)
        return image_embeds

    def forward(self, flatten_patches, position_ids, attention_mask, idx_tensor, weight_tensor):
        rotary_pos_emb = self.rotary(position_ids)
        hidden_states = self.patch_embed(flatten_patches)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor.unsqueeze(2)
        pos_embeds = torch.sum(pos_embeds, 0, False)
        hidden_states = hidden_states + pos_embeds
        if rotary_pos_emb.dtype != hidden_states.dtype:
            rotary_pos_emb = rotary_pos_emb.to(hidden_states.dtype)
        for _, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        image_embeds = self.merger(hidden_states)
        image_embeds = image_embeds.unsqueeze(1)
        return image_embeds

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        patch = torch.randn([256, 1536])
        posision_ids = torch.zeros([2, 256], dtype=torch.int32)
        attention_mask = torch.zeros([1, 256, 256], dtype=torch.float)
        idx_tensor = torch.zeros([4, 256], dtype=torch.int32)
        weight_tensor = torch.randn([4, 256])
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (patch, posision_ids, attention_mask, idx_tensor, weight_tensor),
                    onnx_model,
                    input_names=['patches', 'position_ids', 'attention_mask', 'idx_tensor', 'weight_tensor'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "patches": { 0: "size" },
                        "position_ids": { 1: "size" },
                        "attention_mask": { 1: "size", 2: "size" },
                        "idx_tensor": { 1: "size" },
                        "weight_tensor": { 1: "size" }
                    })
        return onnx_model


# SmolVLM & SmolVLM2
class Idefics3Vision(Vision):
    def __init__(self, visual, base):
        self.patch_size = visual.config.max_image_size['longest_edge']
        self.image_max_size = visual.config.size['longest_edge']
        self.image_height = self.patch_size
        self.image_width = self.image_height
        self.image_embeds = []
        self.image_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.image_norm = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        super().__init__(visual, base)
        self.visual = self.visual.float()
        self.connector = self.visual.connector.float()
        self.quant_bit = 8
        self.transformer_fuse = False

    def load(self):
        self.vision_start_token = '<fake_token_around_image>'
        self.vision_end_token = '<fake_token_around_image>'
        self.image_pad_token = '<image>'
        self.global_image_token = '<global-img>'
        self.vision_start_id = self.tokenizer.encode(self.vision_start_token)[0]
        self.vision_end_id = self.vision_start_id
        self.image_pad_id = self.tokenizer.encode(self.image_pad_token)[0]
        self.global_image_id = self.tokenizer.encode(self.global_image_token)[0]
        self.llm_config['image_size_unit'] = self.patch_size
        self.llm_config['image_size'] = self.image_height
        self.llm_config['image_max_size'] = self.image_max_size
        self.llm_config['vision_start'] = self.vision_start_id
        self.llm_config['vision_end'] = self.vision_end_id
        self.llm_config['image_pad'] = self.image_pad_id
        self.llm_config['global_image'] = self.global_image_id
        # load model
        self.patch_embedding = self.visual.embeddings.patch_embedding
        self.position_embedding = self.visual.embeddings.position_embedding
        self.encoder = self.visual.encoder
        self.post_layernorm = self.visual.post_layernorm

    def init_config(self):
        self.llm_config['is_visual'] = True
        image_mean = self.image_mean * 255.0
        image_norm = 1 / (self.image_norm * 255.0)
        self.llm_config['image_mean'] = image_mean.tolist()
        self.llm_config['image_norm'] = image_norm.tolist()

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
                    if match:
                        img_content = img_content[:match.start()] + img_content[match.end():]
                        hw = match.group(1).split(',')
                        self.image_height, self.image_width = int(hw[0]), int(hw[1])
                    if img_content.startswith('http://') or img_content.startswith('https://'):
                        image_obj = Image.open(requests.get(img_content, stream=True).raw)
                    else:
                        image_obj = Image.open(img_content)
                    img_pad_len, grid_h, grid_w = self.img_process(image_obj)
                    img_pad_str = self.image_pad_token * img_pad_len
                    if grid_h > 0 and grid_w > 0:
                        for n_h in range(grid_h):
                            for n_w in range(grid_w):
                                txt_prompt += (
                                    f"{self.vision_start_token}" + f"<row_{n_h + 1}_col_{n_w + 1}>" + img_pad_str
                                )
                            txt_prompt += "\n"
                        txt_prompt += "\n"
                    txt_prompt += (f'{self.vision_start_token}{self.global_image_token}{img_pad_str}{self.vision_end_token}')
                else:
                    txt_prompt += part
        else:
            txt_prompt = prompt
        input_ids = self.tokenizer(txt_prompt, return_tensors="pt")['input_ids']
        return input_ids

    def images_forward(self, images):
        return self.forward(images)

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding.weight
        encoder_output = self.encoder(embeddings)[0]
        last_hidden_state = self.post_layernorm(encoder_output)
        image_hidden_states = self.connector(last_hidden_state)
        image_hidden_states = image_hidden_states.unsqueeze(2)
        return image_hidden_states

    def get_size(self, height: int, width: int):
        vision_encoder_max_size = self.patch_size
        aspect_ratio = width / height
        if width >= height:
            width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
            height = int(width / aspect_ratio)
            height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
        elif height > width:
            height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
            width = int(height * aspect_ratio)
            width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
        if height > self.image_max_size:
            height = self.image_max_size
        if width > self.image_max_size:
            width = self.image_max_size
        return height, width

    def vision_reshape(self, images):
        batch, channel, height, width = images.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = images.reshape(
            batch,
            channel,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        flatten_patches = patches.reshape(
            batch * grid_h * grid_w, channel, self.patch_size, self.patch_size
        )
        return flatten_patches, grid_h, grid_w

    def img_process(self, image):
        from transformers.image_transforms import (
            convert_to_rgb,
            resize,
            rescale,
            normalize
        )
        from transformers.image_utils import (
            PILImageResampling,
            infer_channel_dimension_format,
            to_numpy_array
        )
        image = convert_to_rgb(image)
        image = to_numpy_array(image)
        resized_height, resized_width = self.get_size(self.image_height, self.image_width)
        format = infer_channel_dimension_format(image)
        resample = PILImageResampling.LANCZOS
        global_image = resize(image, size=(self.patch_size, self.patch_size), resample=resample, input_data_format=format)
        def preprocess(image):
            image = rescale(image, scale=1 / 255.0, input_data_format=format)
            image = normalize(image=image, mean=self.image_mean, std=self.image_norm, input_data_format=format)
            image = np.expand_dims(image, [0])
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return image
        global_image = preprocess(global_image)
        if resized_height > self.patch_size or resized_width > self.patch_size:
            image = resize(image, size=(resized_height, resized_width), resample=resample, input_data_format=format)
            image = preprocess(image)
            image, grid_h, grid_w = self.vision_reshape(image)
            image = torch.concat([image, global_image], dim=0)
        else:
            grid_h, grid_w = 0, 0
            image = global_image
        image_embed = self.images_forward(image)
        num_images, img_pad_len, _, vision_hidden_size = image_embed.shape
        self.image_embeds.append(image_embed.reshape(-1, 1, vision_hidden_size))
        return img_pad_len, grid_h, grid_w

    def embed(self, input_ids, images = None, videos = None):
        input_embeds = self.embed_(input_ids)
        if self.image_embeds is not None and len(self.image_embeds) > 0:
            image_mask = (input_ids == self.image_pad_id).squeeze()
            input_embeds[image_mask] = torch.concat(self.image_embeds, dim=0).to(input_embeds.dtype)
        return input_embeds

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        pixel_values = torch.randn([1, 3, self.patch_size, self.patch_size])
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (pixel_values),
                    onnx_model,
                    input_names=['pixel_values'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "pixel_values": { 0: "size" },
                    })
        return onnx_model

# FastVLM
class MobileCLIPVision(QwenVision):
    def __init__(self, visual, base):
        super().__init__(visual, base)
        self.visual = visual.float()
        self.mm_projector = self.visual.mm_projector.float()
        self.quant_bit = 8
        self.group_conv_native = False

    def init_config(self):
        self.llm_config['is_visual'] = True
        image_mean = np.array([0.0, 0.0, 0.0])
        image_norm = np.array([1.0, 1.0, 1.0]) / 255.0
        self.llm_config['image_mean'] = image_mean.tolist()
        self.llm_config['image_norm'] = image_norm.tolist()

    def load(self):
        self.image_size = self.visual.config['image_cfg']['image_size']
        self.image_start_id = -200
        self.llm_config['image_size'] = self.image_size
        self.llm_config['vision_start'] = -200
        self.llm_config['vision_end'] = -200
        self.llm_config['image_pad'] = -200

    def forward(self, images):
        image_features = self.visual(images)
        image_features = self.mm_projector(image_features)
        image_features = image_features.permute(1, 0, 2)
        return image_features

class MiniCPMVision(Vision):
    def __init__(self, visual, base):
        self.scale_resolution = 448
        self.max_slice_nums = 9
        self.num_patches_per_side = 70
        self.patch_size = base.config.patch_size
        self.image_size = base.config.image_size
        self.image_height = self.patch_size
        self.image_width = self.image_height
        self.image_embeds = []
        self.image_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.image_norm = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        super().__init__(visual, base)
        self.quant_bit = base.args.quant_bit
        self.transformer_fuse = False
        # rebuild visual
        self.visual = self.visual.float()
        self.patch_embedding = self.visual.embeddings.patch_embedding
        self.position_embedding = self.visual.embeddings.position_embedding
        self.encoder = self.visual.encoder
        self.post_layernorm = self.visual.post_layernorm
        # rebuild resampler
        self.resampler = self.visual.resampler.float()
        attrs = ['query', 'kv_proj', 'ln_kv', 'ln_q', 'attn', 'ln_post', 'proj', 'pos_embed', 'embed_dim']
        for attr in attrs:
            setattr(self, attr, getattr(self.resampler, attr))

    def load(self):
        pass

    def init_config(self):
        self.llm_config['is_visual'] = True
        image_mean = self.image_mean * 255.0
        image_norm = 1 / (self.image_norm * 255.0)
        self.llm_config['image_mean'] = image_mean.tolist()
        self.llm_config['image_norm'] = image_norm.tolist()
        # vision tokens
        self.vision_start_token = '<image>'
        self.vision_end_token = '</image>'
        self.image_pad_token = '<unk>'
        self.vision_id_start_token = '<image_id>'
        self.vision_id_end_token = '</image_id>'
        self.vision_slice_start_token = '<slice>'
        self.vision_slice_end_token = '</slice>'
        self.vision_start_id = self.tokenizer.encode(self.vision_start_token)[-1]
        self.vision_end_id = self.tokenizer.encode(self.vision_end_token)[-1]
        self.image_pad_id = self.tokenizer.encode(self.image_pad_token)[-1]
        self.vision_id_start_id = self.tokenizer.encode(self.vision_id_start_token)[-1]
        self.vision_id_end_id = self.tokenizer.encode(self.vision_id_end_token)[-1]
        self.vision_slice_start_id = self.tokenizer.encode(self.vision_slice_start_token)[-1]
        self.vision_slice_end_id = self.tokenizer.encode(self.vision_slice_end_token)[-1]
        self.llm_config['image_size_unit'] = self.patch_size
        self.llm_config['image_size'] = self.image_size
        # self.llm_config['image_max_size'] = self.image_max_size
        self.llm_config['vision_start'] = self.vision_start_id
        self.llm_config['vision_end'] = self.vision_end_id
        self.llm_config['image_pad'] = self.image_pad_id
        self.llm_config['vision_id_start_id'] = self.vision_id_start_id
        self.llm_config['vision_id_end_id'] = self.vision_id_end_id
        self.llm_config['vision_slice_start_id'] = self.vision_slice_start_id
        self.llm_config['vision_slice_end_id'] = self.vision_slice_end_id

    def str_to_ids(self, prompt):
        if '<img>' in prompt and '</img>' in prompt:
            import re
            import requests
            from PIL import Image
            pattern = r'(<img>.*?</img>)'
            parts = re.split(pattern, prompt)
            txt_prompt = ''
            for part in parts:
                idx = 0
                if re.match(pattern, part):
                    img_content = re.search(r'<img>(.*?)</img>', part).group(1)
                    # find <hw></hw> in image_content
                    match = re.search(r'<hw>(.*?)</hw>', img_content)
                    if match:
                        img_content = img_content[:match.start()] + img_content[match.end():]
                        hw = match.group(1).split(',')
                        self.image_height, self.image_width = int(hw[0]), int(hw[1])
                    if img_content.startswith('http://') or img_content.startswith('https://'):
                        image_obj = Image.open(requests.get(img_content, stream=True).raw)
                    else:
                        image_obj = Image.open(img_content)
                    img_pad_len, num_images = self.img_process(image_obj)
                    img_pad_str = self.image_pad_token * img_pad_len
                    # image id
                    txt_prompt += (f"{self.vision_id_start_token}{idx}{self.vision_id_end_token}")
                    idx += 1
                    # global image
                    txt_prompt += (f'{self.vision_start_token}{img_pad_str}{self.vision_end_token}')
                    # slices image
                    for s in range(num_images - 1):
                        txt_prompt += (f'{self.vision_slice_start_token}{img_pad_str}{self.vision_slice_end_token}')
                else:
                    txt_prompt += part
        else:
            txt_prompt = prompt
        input_ids = self.tokenizer(txt_prompt, return_tensors="pt")['input_ids']
        return input_ids

    def calculate_image_processing_plan(
        self,
        original_size: Tuple[int, int],
        max_slice_nums: int = 9,
        scale_resolution: int = 448,
        patch_size: int = 14,
    ):
        def _get_target_size(size: Tuple[int, int], upscale: bool) -> Tuple[int, int]:
            h, w = size
            if not (upscale or (w * h > scale_resolution * scale_resolution)):
                target_w, target_h = w, h
            else:
                r = w / h if h != 0 else 0
                if r > 0:
                    target_h = int(scale_resolution / math.sqrt(r))
                    target_w = int(target_h * r)
                else:
                    target_h, target_w = 0, scale_resolution

            final_h = max(round(target_h / patch_size) * patch_size, patch_size)
            final_w = max(round(target_w / patch_size) * patch_size, patch_size)
            return final_h, final_w

        original_height, original_width = original_size
        best_grid = None
        refine_image_size = None

        if original_width > 0 and original_height > 0:
            ratio = (original_width * original_height) / (scale_resolution * scale_resolution)
            multiple = min(math.ceil(ratio), max_slice_nums)
            if multiple > 1:
                candidates = []
                for num in {multiple - 1, multiple, multiple + 1}:
                    if 1 < num <= max_slice_nums:
                        m = 1
                        while m * m <= num:
                            if num % m == 0:
                                candidates.append((m, num // m))
                                if m * m != num:
                                    candidates.append((num // m, m))
                            m += 1
                if candidates:
                    log_ratio = math.log(original_width / original_height)
                    best_grid = min(candidates, key=lambda g: abs(log_ratio - math.log(g[1] / g[0])) if g[0] != 0 else float('inf'))

        if best_grid is None:
            source_image_size = _get_target_size(original_size, upscale=True)
        else:
            source_image_size = _get_target_size(original_size, upscale=False)
            patch_h = original_height / best_grid[0]
            patch_w = original_width / best_grid[1]
            best_patch_size = _get_target_size((patch_h, patch_w), upscale=True)
            refine_image_size = (best_patch_size[0] * best_grid[0], best_patch_size[1] * best_grid[1])

        return source_image_size, refine_image_size, best_grid

    def vision_reshape(self, images, best_grid, patch_size):
        channel, height, width = images.shape
        grid_h, grid_w = best_grid
        sub_height, sub_width = height // grid_h, width // grid_w
        num_patches_h = sub_height // patch_size
        num_patches_w = sub_width // patch_size
        expanded_view = images.reshape(
            channel,
            grid_h,
            num_patches_h,
            patch_size,
            grid_w,
            num_patches_w,
            patch_size
        )
        permuted_view = expanded_view.permute(1, 4, 0, 3, 2, 5, 6)
        flatten_patches = permuted_view.reshape(
            grid_h * grid_w, channel, patch_size, num_patches_h * num_patches_w * patch_size
        )
        tgt_sizes = torch.tensor([[num_patches_h, num_patches_w]] * (grid_h * grid_w))
        return flatten_patches, tgt_sizes

    def gen_position_ids(self, tgt_sizes: torch.Tensor, num_patches_per_side: int) -> torch.Tensor:
        batch_size = tgt_sizes.size(0)
        num_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).long()
        max_patches = num_patches.max().item() if batch_size > 0 else 0
        all_position_ids = torch.zeros(batch_size, max_patches, dtype=torch.long)
        for i in range(batch_size):
            nb_patches_h = tgt_sizes[i, 0].item()
            nb_patches_w = tgt_sizes[i, 1].item()
            num_current_patches = num_patches[i].item()
            i_coords = torch.arange(nb_patches_h, dtype=torch.float32).unsqueeze(1)
            j_coords = torch.arange(nb_patches_w, dtype=torch.float32).unsqueeze(0)
            bucket_h = (i_coords / nb_patches_h * num_patches_per_side).floor()
            bucket_w = (j_coords / nb_patches_w * num_patches_per_side).floor()
            pos_ids = bucket_h * num_patches_per_side + bucket_w
            pos_ids_flat = pos_ids.flatten().long()
            all_position_ids[i, :num_current_patches] = pos_ids_flat
        return all_position_ids

    def img_process(self, image):
        from transformers.image_transforms import (
            convert_to_rgb,
            resize,
            rescale,
            normalize
        )
        from transformers.image_utils import (
            PILImageResampling,
            infer_channel_dimension_format,
            to_numpy_array
        )
        image = convert_to_rgb(image)
        image = to_numpy_array(image)
        h, w, c = image.shape
        global_size, refine_size, best_grid = self.calculate_image_processing_plan((h, w))
        def preprocess(image, tsize):
            format = infer_channel_dimension_format(image)
            resample = PILImageResampling.BICUBIC
            image = resize(image, size=tsize, resample=resample, input_data_format=format)
            image = rescale(image, scale=1 / 255.0, input_data_format=format)
            image = normalize(image=image, mean=self.image_mean, std=self.image_norm, input_data_format=format)
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)
            return image
        global_image = preprocess(image, global_size)
        refine_image = preprocess(image, refine_size)
        global_patch, global_tgt_sizes = self.vision_reshape(global_image, (1, 1), self.patch_size)
        refine_patches, refine_tgt_sizes = self.vision_reshape(refine_image, best_grid, self.patch_size)
        # concat global image and slices
        global_len = global_patch.shape[-1]
        refine_len = refine_patches.shape[-1]
        if refine_len > global_len:
            global_patch = F.pad(global_patch, (0, refine_len - global_len))
        all_pixel_values = torch.cat([global_patch, refine_patches], dim=0)
        # tgt sizes and masks
        tgt_sizes = torch.cat([global_tgt_sizes, refine_tgt_sizes], dim=0)
        image_embed = self.images_forward(all_pixel_values, tgt_sizes)
        num_images, img_pad_len, vision_hidden_size = image_embed.shape
        self.image_embeds.append(image_embed.reshape(-1, 1, vision_hidden_size))
        return img_pad_len, num_images

    def embed(self, input_ids, images = None, videos = None):
        input_embeds = self.embed_(input_ids)
        if self.image_embeds is not None and len(self.image_embeds) > 0:
            image_mask = (input_ids == self.image_pad_id).squeeze()
            input_embeds[image_mask] = torch.concat(self.image_embeds, dim=0).to(input_embeds.dtype)
        return input_embeds

    def images_forward(self, pixel_values, tgt_sizes):
        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
        B = tgt_sizes.shape[0]
        position_ids = self.gen_position_ids(tgt_sizes, self.num_patches_per_side)
        attention_mask = torch.zeros((B, max_patches), dtype=torch.float32)
        attention_mask[0, tgt_sizes[0][0] * tgt_sizes[0][1]:] = torch.finfo(torch.float32).min
        return self.forward(pixel_values, position_ids, attention_mask, tgt_sizes)

    def visual_forward(self, pixel_values, position_ids, attention_mask):
        L = attention_mask.shape[1]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, L, -1) # 2D -> 4D
        patch_embeds = self.patch_embedding(pixel_values)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = patch_embeds.flatten(2).transpose(1, 2) + pos_embeds
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = encoder_outputs[0]
        return self.post_layernorm(last_hidden_state)

    def resampler_forward(self, x, tgt_sizes, attention_mask):
        bs = x.shape[0]
        N = bs - 1
        D = self.embed_dim
        gh, gw = tgt_sizes[0]
        glen = gh * gw
        sh, sw = tgt_sizes[1]
        slen = sh * sw
        # global image pos
        pos_embed_global = self.pos_embed[:gh, :gw, :].reshape(glen, 1, D)
        pad_tuple = (0, 0, 0, 0, 0, slen - glen)
        pos_embed_global = F.pad(pos_embed_global, pad_tuple, "constant", 0)
        # slice image pos
        pos_embed_slice = self.pos_embed[:sh, :sw, :].reshape(slen, D)
        pos_embed_slice = pos_embed_slice.unsqueeze(1).repeat(1, N, 1)
        pos_embed = torch.cat([pos_embed_global, pos_embed_slice], dim=1)
        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D
        q = self.ln_q(self.query)  # Q * D
        out = self.attn(
            q.unsqueeze(1).repeat(1, bs, 1),
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=attention_mask)[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D
        x = self.ln_post(x)
        return x @ self.proj

    def forward(self, pixel_values, position_ids, attention_mask, tgt_sizes):
        # rewrite position_ids in visual and pos_embed in resampler for onnx export
        x = self.visual_forward(pixel_values, position_ids, attention_mask)
        vision_embedding = self.resampler_forward(x, tgt_sizes, attention_mask)
        return vision_embedding

    @spinner_run(f'export visual to ')
    def export(self, onnx_path):
        num_grids = 5
        num_patches = 2
        pixel_values = torch.randn([num_grids, 3, self.patch_size, num_patches * num_patches * self.patch_size])
        attention_mask = torch.zeros([num_grids, num_patches * num_patches], dtype=torch.float32)
        tgt_sizes = torch.tensor([[num_patches, num_patches]] * num_grids, dtype=torch.int32)
        position_ids = self.gen_position_ids(tgt_sizes, self.num_patches_per_side)
        onnx_model = f'{onnx_path}/visual.onnx'
        onnx_export(self, (pixel_values, position_ids, attention_mask, tgt_sizes),
                    onnx_model,
                    input_names=['pixel_values', 'position_ids', 'attention_mask', 'tgt_sizes'],
                    output_names=['image_embeds'],
                    dynamic_axes={
                        "pixel_values": { 0: "num", 3: "size" },
                        "position_ids": { 0: "num", 1: "size" },
                        "attention_mask": { 0: "num", 1: "size" },
                        "tgt_sizes": { 0: "num" }
                    })
        return onnx_model
