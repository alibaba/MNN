import os
import gc
import sys
import math
import copy
import json
import time
import base64
import logging
import inspect
import warnings
import argparse
import functools
import traceback
from collections import defaultdict
from typing import Optional, Tuple, List, Union, Dict

from tqdm import tqdm
from yaspin import yaspin

import onnx
import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

RESET = "\033[0m"
GREEN = "\033[32;1m"
YELLOW = "\033[33;4m"
EXPORT_LOG = '.export.log'

# ignore warnning info
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def spinner_run(text='Processing...', hide=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with yaspin(text=text, color="cyan") as spinner:
                start = time.time()
                try:
                    if hide: spinner.hide()
                    result = func(*args, **kwargs)
                    if hide: spinner.show()
                except Exception as e:
                    spinner.fail("💥 Failed")
                    traceback.print_exc()
                    exit(1)
                end = time.time()
                during = f'[{end-start:05.2f} s]'.replace('[0', '[ ')
                padding = ' ' * (64 - len(spinner.text) - len(result))
                spinner.text = f'{spinner.text}{YELLOW}{result}{RESET}{padding}{GREEN}{during}{RESET}'
                spinner.ok("✅ Done")
                return result
        return wrapper
    return decorator

class ModelMapper:
    def __init__(self):
        self.attrs = []
        self.mapper = dict()
        self.regist_models()

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

    def regist_models(self):
        self.defualt_map()
        # regist models
        self.regist_llama()
        self.regist_mllama()
        self.regist_qwen()
        self.regist_glm()
        self.regist_glm2()
        self.regist_phi()
        self.regist_gemma2()
        self.register_openelm()

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

    def regist_mllama(self):
        mllama_map = {
            'config': {
                'hidden_size': 'text_config.hidden_size',
                'num_attention_heads': 'text_config.num_attention_heads',
                'num_hidden_layers': 'text_config.num_hidden_layers',
                'num_key_value_heads': 'text_config.num_key_value_heads',
                'rope_theta': 'text_config.rope_theta'
            },
            'model': {
                'lm_': 'language_model.lm_head',
                'embed_': 'language_model.model.embed_tokens',
                'blocks_': 'language_model.model.layers',
                'final_layernorm_': 'language_model.model.norm',
                'visual': 'vision_model',
                'multi_modal_projector': 'multi_modal_projector'
            },
            'decoder': {
                'self_attn': 'self_attn',
                'cross_attn': 'cross_attn',
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
                'cross_attn_attn_gate': 'cross_attn_attn_gate',
                'cross_attn_mlp_gate': 'cross_attn_mlp_gate'
            }
        }
        self.regist('mllama', mllama_map)

    def regist_qwen(self):
        qwen_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'rope_theta': 'rotary_emb_base',
            },
            'model': {
                'lm_': 'lm_head',
                'embed_': 'transformer.wte',
                'blocks_': 'transformer.h',
                'final_layernorm_': 'transformer.ln_f',
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

    def regist_glm(self):
        glm_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_layers'
            },
            'model': {
                'lm_': 'lm_head',
                'embed_': 'transformer.word_embeddings',
                'blocks_': 'transformer.layers',
                'final_layernorm_': 'transformer.final_layernorm',
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
            },
            'model': {
                'lm_': 'transformer.output_layer',
                'embed_': 'transformer.embedding.word_embeddings',
                'blocks_': 'transformer.encoder.layers',
                'final_layernorm_': 'transformer.encoder.final_layernorm',
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
                'lm_': 'lm_head.linear',
                'embed_': 'transformer.embd.wte',
                'blocks_': 'transformer.h',
                'final_layernorm_': 'lm_head.ln',
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

    def regist_gemma2(self):
        gemma2_config = copy.deepcopy(self.default_config)
        gemma2_config['head_dim'] = 'head_dim'
        gemma2_decoder = copy.deepcopy(self.default_decoder)
        gemma2_decoder['pre_feedforward_layernorm'] = 'pre_feedforward_layernorm'
        gemma2_decoder['post_feedforward_layernorm'] = 'post_feedforward_layernorm'
        gemma2_map = {
            'config': gemma2_config,
            'model': self.defualt_model,
            'decoder': gemma2_decoder,
            'attention': self.default_attention
        }
        self.regist('gemma2', gemma2_map)

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
            'lm_': 'lm_head',
            'embed_': 'transformer.token_embeddings',
            'blocks_': 'transformer.layers',
            'final_layernorm_': 'transformer.norm'
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

    def defualt_map(self):
        # default map is `LlamaForCausalLM`
        self.config_key = 'config'
        self.model_key = 'model'
        self.decoder_key = 'decoder'
        self.attention_key = 'attention'
        self.default_config = {
            'hidden_size': 'hidden_size',
            'num_attention_heads': 'num_attention_heads',
            'num_hidden_layers': 'num_hidden_layers',
            'num_key_value_heads': 'num_key_value_heads',
            'rope_theta': 'rope_theta'
        }
        self.defualt_model = {
            'lm_': 'lm_head',
            'embed_': 'model.embed_tokens',
            'blocks_': 'model.layers',
            'final_layernorm_': 'model.norm',
            'visual': 'visual'
        }
        self.default_decoder = {
            'self_attn': 'self_attn',
            'mlp': 'mlp',
            'input_layernorm': 'input_layernorm',
            'post_attention_layernorm': 'post_attention_layernorm'
        }
        self.default_attention = {
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj'
        }
        self.default_map = {
            'config': self.default_config,
            'model': self.defualt_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }

    @staticmethod
    def do_map(dst, src, map):
        for dst_attr, src_attr in map.items():
            attributes = src_attr.split('.')
            obj = src
            for attr in attributes:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    obj = None
                    break
            setattr(dst, dst_attr, obj)

# Quant class

# awq quantizer start
class AwqQuantizer:
    def __init__(
        self,
        model,
        modules_to_not_convert=None,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        self.awq_model = model
        self.model = model
        self.tokenizer = model.tokenizer
        self.w_bit = model.quant_bit
        self.group_size = model.quant_block
        self.zeropoint = not model.symmetric
        self.calib_data = 'ag_news'
        self.split = 'test'
        self.duo_scaling = True
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        # zero point quantization
        if self.zeropoint:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            offset = 1 << (self.w_bit - 1)
            clip_max = offset - 1
            clip_min = -offset
            scales = (max_val - min_val) / (clip_max - clip_min)
            zeros =  - torch.round(min_val / scales) + clip_min
            qw = torch.round(w / scales) + zeros
            qw = torch.clamp(qw, clip_min, clip_max)
            w = (qw - zeros) * scales
            zeros = min_val.view(org_w_shape[0], -1)
        else:
            abs_max = w.abs().amax(dim=1, keepdim=True)
            offset = 1 << (self.w_bit - 1)
            clip_max = offset - 1
            clip_min = -clip_max
            scales = abs_max / clip_max
            w = torch.clamp(torch.round(w / scales), clip_min, clip_max)  * scales
            zeros = None

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    def quantize(self):
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                best_device = AwqQuantizer.get_best_device()

                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)
            # print(f'# {i} inps shape: {self.inps.shape}, inps.max: {self.inps.max()}')

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = AwqQuantizer.get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = AwqQuantizer.exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            input_feat = self._get_input_feat(self.modules[i], named_linears)
            AwqQuantizer.clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config = []
            # q, k, v proj
            module_config.append(
                dict(
                    prev_op=self.modules[i].input_layernorm,
                    layers=[
                        self.modules[i].self_attn.q_proj,
                        self.modules[i].self_attn.k_proj,
                        self.modules[i].self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=self.modules[i].self_attn,
                    kwargs=self.module_kwargs,
                )
            )
            # o_proj
            if self.modules[i].self_attn.v_proj.weight.shape == self.modules[i].self_attn.o_proj.weight.shape:
                module_config.append(
                    dict(
                        prev_op=self.modules[i].self_attn.v_proj,
                        layers=[self.modules[i].self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )
            # mlp gate
            module_config.append(
                dict(
                    prev_op=self.modules[i].post_attention_layernorm,
                    layers=[self.modules[i].mlp.gate_proj, self.modules[i].mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=self.modules[i].mlp,
                )
            )
            # mlp down
            module_config.append(
                dict(
                    prev_op=self.modules[i].mlp.up_proj,
                    layers=[self.modules[i].mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            # print(scales_list); exit(0)
            AwqQuantizer.apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                AwqQuantizer.apply_clip(self.modules[i], clip_list)

            AwqQuantizer.clear_memory()

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            # print(module, x, module_kwargs); exit(0)
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[torch.nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        AwqQuantizer.clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)

        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        AwqQuantizer.clear_memory(x_sum)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            AwqQuantizer.get_op_name(module, prev_op),
            tuple([AwqQuantizer.get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[torch.nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        ord_weights = []
        for fc in linears2scale:
            ord_weights.append(fc.weight.data.clone())

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()

            for fc, ord_weight in zip(linears2scale, ord_weights):
                fc.weight.data = ord_weight.clone()

        del ord_weights

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(AwqQuantizer.get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        AwqQuantizer.clear_memory(input_feat)
        AwqQuantizer.clear_memory(org_out)

        return best_max_val.squeeze(1)

    @staticmethod
    @torch.no_grad()
    def apply_clip(module, clip_list: Tuple[str, torch.Tensor]):
        for name, max_val in clip_list:
            layer: torch.nn.Linear = AwqQuantizer.get_op_by_name(module, name)
            layer.to(AwqQuantizer.get_best_device())
            max_val = max_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)
            layer.cpu()

    @staticmethod
    @torch.no_grad()
    def scale_fc_fcs(fc1: torch.nn.Linear, fcs: List[torch.nn.Linear], scales: torch.Tensor):
        if not isinstance(fcs, list):
            fcs = [fcs]

        scales = scales.to(fc1.weight.device)

        fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
        if fc1.bias is not None:
            fc1.bias.div_(scales.view(-1))

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))

        for p in fc1.parameters():
            assert torch.isnan(p).sum() == 0
        for fc in fcs:
            for p in fc.parameters():
                assert torch.isnan(p).sum() == 0

    @staticmethod
    def is_allowed_act_fns(op):
        from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation
        allowed_act_fns = [
            torch.nn.GELU,
            NewGELUActivation,
            PytorchGELUTanh,
            GELUActivation,
        ]
        return (op in allowed_act_fns)

    @staticmethod
    def is_allowed_norms(op):
        if isinstance(op, torch.nn.LayerNorm):
            return True
        if any(t in str(type(op)) for t in ['LlamaRMSNorm', 'GemmaRMSNorm', 'CohereLayerNorm']):
            return True
        return False

    @staticmethod
    @torch.no_grad()
    def scale_fc_fc(fc1: torch.nn.Linear, fc2: torch.nn.Linear, scales: torch.Tensor):
        assert isinstance(fc1, torch.nn.Linear)
        assert isinstance(fc2, torch.nn.Linear)

        scales = scales.to(fc1.weight.device)
        fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
        if fc1.bias is not None:
            fc1.bias.div_(scales.view(-1))

        fc2.weight.mul_(scales.view(1, -1))

        for p in fc1.parameters():
            assert torch.isnan(p).sum() == 0
        for p in fc2.parameters():
            assert torch.isnan(p).sum() == 0

    @staticmethod
    @torch.no_grad()
    def scale_ln_fcs(ln: torch.nn.Linear, fcs: List[torch.nn.Linear], scales: torch.Tensor):
        if not isinstance(fcs, list):
            fcs = [fcs]

        scales = scales.to(ln.weight.device)

        # GemmaRMSNorm is different from Llama's in that it multiplies
        # (1 + weight) to the output, instead of just weight.
        if 'GemmaRMSNorm' in str(type(ln)):
            ln.weight += 1
            ln.weight.div_(scales)
            ln.weight -= 1
        else:
            ln.weight.div_(scales)

        if hasattr(ln, "bias") and ln.bias is not None:
            ln.bias.div_(scales)

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))

        for p in ln.parameters():
            assert torch.isnan(p).sum() == 0
        for fc in fcs:
            for p in fc.parameters():
                assert torch.isnan(p).sum() == 0

    @staticmethod
    @torch.no_grad()
    def scale_gelu_fc(gelu, fc: torch.nn.Linear, scales: torch.Tensor):
        assert AwqQuantizer.is_allowed_act_fns(gelu)
        assert isinstance(fc, torch.nn.Linear)

        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

    @staticmethod
    def apply_scale(module, scales_list, input_feat_dict=None):
        for prev_op_name, layer_names, scales in scales_list:
            prev_op = AwqQuantizer.get_op_by_name(module, prev_op_name)
            layers = [AwqQuantizer.get_op_by_name(module, name) for name in layer_names]

            best_device = AwqQuantizer.get_best_device()
            prev_op.to(best_device)
            for layer in layers:
                layer.to(best_device)
            scales.to(best_device)
            if (
                isinstance(prev_op, torch.nn.Linear)
                and type(layers) == list
                and isinstance(layers[0], torch.nn.Linear)
            ):
                if len(layers) == 1:
                    AwqQuantizer.scale_fc_fc(prev_op, layers[0], scales)
                else:
                    AwqQuantizer.scale_fc_fcs(prev_op, layers, scales)
            elif (
                AwqQuantizer.is_allowed_norms(prev_op)
                or "rmsnorm" in str(prev_op.__class__).lower()
            ):
                AwqQuantizer.scale_ln_fcs(prev_op, layers, scales)

            elif AwqQuantizer.is_allowed_act_fns(prev_op):
                #new_module = ScaledActivation(prev_op, scales)
                #set_op_by_name(module, prev_op_name, new_module)
                AwqQuantizer.scale_gelu_fc(prev_op, layers[0], scales)
            else:
                raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

            # apply the scaling to input feat if given; prepare it for clipping
            if input_feat_dict is not None:
                for layer_name in layer_names:
                    # Skip the modules that are not quantized
                    if layer_name in input_feat_dict:
                        inp = input_feat_dict[layer_name]
                        inp.div_(scales.view(1, -1).to(inp.device))

            prev_op.cpu()
            for layer in layers:
                layer.cpu()
            scales.cpu()

    @staticmethod
    def exclude_layers_to_not_quantize(linear_layers, modules_to_not_convert):
        if modules_to_not_convert is None:
            return linear_layers

        filtered_layers = {}
        for name, linear_layer in linear_layers.items():
            if not any(key in name for key in modules_to_not_convert):
                filtered_layers[name] = linear_layer
        return filtered_layers

    @staticmethod
    def get_named_linears(module):
        return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}

    @staticmethod
    def get_op_by_name(module, op_name):
        # get the op by its name relative to the module
        for name, m in module.named_modules():
            if name == op_name:
                return m
        raise ValueError(f"Cannot find op {op_name} in module {module}")

    @staticmethod
    def get_calib_dataset(
        data: Union[str, List[str], List[List[int]]] = "pileval",
        tokenizer=None,
        n_samples=128,
        max_seq_len=512,
        split="train",
        text_column="text",
    ):
        if isinstance(data, str):
            from datasets import load_dataset
            if data == "pileval":
                dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            else:
                dataset = load_dataset(data, split=split)
            # dataset = dataset.shuffle(seed=42)
        elif isinstance(data, list):
            if isinstance(data[0], str):
                dataset = [{text_column: text} for text in data]
            elif isinstance(data[0][0], int):
                dataset = data
            else:
                raise NotImplementedError(
                    "Either pass a string to a huggingface dataset or a list"
                    "that is preprocessed with one sample of text per element"
                    " or a list of list of int for tokenized words."
                )
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )

        samples = []
        n_run = 0
        for data in dataset:
            if isinstance(data, list):
                line_encoded = data
            else:
                line = data[text_column]
                line = line.strip()
                line_encoded = tokenizer.encode(line)
            if len(line_encoded) > max_seq_len:
                continue
            sample = torch.tensor([line_encoded])
            if sample.numel() == 0:
                continue
            samples.append(sample)
            n_run += 1
            if n_run == n_samples:
                break
        # now concatenate all samples and split according to max sequence length
        cat_samples = torch.cat(samples, dim=1)
        n_split = cat_samples.shape[1] // max_seq_len
        logging.debug(f" * Split into {n_split} blocks")
        return [
            cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
        ]

    @staticmethod
    def get_best_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"

    @staticmethod
    def clear_memory(weight=None):
        if weight is not None:
            del weight
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_op_name(module, op):
        # get the name of the op relative to the module
        for name, m in module.named_modules():
            if m is op:
                return name
        raise ValueError(f"Cannot find op {op} in module {module}")

    @staticmethod
    def append_str_prefix(x, prefix):
        if isinstance(x, str):
            return prefix + x
        elif isinstance(x, tuple):
            return tuple([AwqQuantizer.append_str_prefix(y, prefix) for y in x])
        elif isinstance(x, list):
            return [AwqQuantizer.append_str_prefix(y, prefix) for y in x]
        else:
            return x

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.blocks
        samples = AwqQuantizer.get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split
        )
        # samples = torch.cat(samples, dim=0)
        samples = torch.cat(samples[:1], dim=0) # just using 1 batch
        inps = []
        layer_kwargs = {}
        # build inps
        self.model.seq_len = samples.numel()
        self.model.context_len = samples.numel() - 2
        self.model.token_len = 0
        best_device = AwqQuantizer.get_best_device()
        inps = self.model.embedding(samples).to(best_device)
        position_ids = self.model.get_position_ids()
        rotary_pos_emb = self.model.rotary(position_ids)
        attention_mask = self.model.get_attention_mask()
        layer_kwargs["rotary_pos_emb"] = rotary_pos_emb.to(best_device)
        layer_kwargs["attention_mask"] = attention_mask.to(best_device)
        del samples
        AwqQuantizer.clear_memory()
        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)
        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs
# awq quantizer end

# Export class

# custom op start
class FakeLinearOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, in_features, out_features, has_bias, name):
        # These become the operator attributes.
        kwargs = {
            "in_features_i": in_features,
            "out_features_i": out_features,
            "has_bias_i": has_bias,
            "name_s": name
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(input)[:-1] + [out_features]
        output_type = input.type().with_sizes(out_sizes)
        return g.op("LlmExporter::FakeLinear", input, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, in_features, out_features, has_bias, name):
        out_shape = list(input.shape)[:-1] + [out_features]
        return input.new_zeros(out_shape)

class FakeLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias, name):
        super(FakeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.name = name

    def forward(self, x):
        return FakeLinearOp.apply(x, self.in_features, self.out_features, self.has_bias, self.name)

class FusedAttentionOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query, key, value, attention_mask, hidden_size, name):
        # These become the operator attributes.
        kwargs = {
            "hidden_size_i": hidden_size,
            "name_s": name
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(query)
        output_type = query.type().with_sizes(out_sizes)
        return g.op("LlmExporter::FusedAttention", query, key, value, attention_mask, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, query, key, value, attention_mask, hidden_size, name):
        out_shape = list(query.shape)[:2] + [hidden_size]
        return query.new_zeros(out_shape)

class FusedAttention(torch.nn.Module):
    def __init__(self, hidden_size, name):
        super(FusedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.name = name

    def forward(self, query, key, value, attention_mask):
        return FusedAttentionOp.apply(query, key, value, attention_mask, self.hidden_size, self.name)

# custom op end

class OnnxRebuilder:
    def __init__(self, onnx_path, weight_ops):
        self.weight_ops = weight_ops
        self.onnx_model = onnx.load(onnx_path)
        self.dst_path = onnx_path
        self.onnx_weight_path = f'{onnx_path}.data'
        self.onnx_weight_offset = 0

    def make_external(self, name, data, shape):
        # write to external weight
        length = self.onnx_weight.write(data.tobytes())
        location = os.path.basename(self.onnx_weight_path)
        offset = self.onnx_weight_offset
        self.onnx_weight_offset += length
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.extend(shape)
        # external info
        tensor.data_location = onnx.TensorProto.EXTERNAL
        for k, v in { "location": location, "offset": offset, "length": length }.items():
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)
        self.onnx_model.graph.initializer.append(tensor)

    def build_weight(self, name, has_bias, ic, oc):
        assert(name in self.weight_ops)
        linear = self.weight_ops[name]
        assert(linear.in_features == ic and
               linear.out_features == oc and
               (linear.bias is not None) == has_bias)
        weight_name, bias_name = f'{name}_weight', f'{name}_bias'
        weight = linear.weight.data.transpose(1, 0).flatten().float().numpy()
        self.make_external(weight_name, weight, [ic, oc])
        if has_bias:
            bias = linear.bias.data.flatten().float().numpy()
            self.make_external(bias_name, bias, [oc])
        return weight_name, bias_name

    def rebuild(self):
        from onnx import helper
        new_nodes = []
        self.onnx_weight = open(self.onnx_weight_path, 'wb')
        for node in self.onnx_model.graph.node:
            if node.op_type == 'FakeLinear':
                attributes = {a.name: a for a in node.attribute}
                name = attributes.get('name').s.decode('utf-8')
                has_bias = attributes.get('has_bias').i
                ic = attributes.get('in_features').i
                oc = attributes.get('out_features').i
                weight, bias = self.build_weight(name, has_bias, ic, oc)
                if has_bias:
                    # fakelinear -> matmul + add
                    middle_tensor = f'{name}_matmul'
                    new_nodes.append(helper.make_node('MatMul', [node.input[0], weight], [middle_tensor], name))
                    new_nodes.append(helper.make_node('Add', [middle_tensor, bias], node.output, f'{name}/Add'))
                else:
                    # fakelinear -> matmul
                    new_nodes.append(helper.make_node('MatMul', [node.input[0], weight], node.output, name))
            else:
                new_nodes.append(node)
        self.onnx_weight.close()
        del self.onnx_model.graph.node[:]
        self.onnx_model.graph.node.extend(new_nodes)
        onnx.save(self.onnx_model, self.dst_path)
        return self.onnx_weight_path

class MNNConveter:
    def __init__(self, onnx_path, weight_ops, config):
        self.weight_ops = weight_ops
        self.config = config
        self.quant_block = config.args.quant_block
        self.quant_bit = config.args.quant_bit
        self.lm_quant_bit = config.args.lm_quant_bit
        self.symmetric = config.args.sym
        self.mnn_weight_offset = 0
        self.onnx_model_path = onnx_path
        self.mnn_name = os.path.basename(onnx_path).replace('.onnx', '.mnn')
        self.mnn_model_path = os.path.join(config.args.dst_path, self.mnn_name)
        self.mnn_weight_path = f'{self.mnn_model_path}.weight'
        if os.path.exists(config.args.mnnconvert):
            self.mnnconvert = config.args.mnnconvert
        else:
            self.mnnconvert = None

    def convert(self, convert_args):
        sfd = os.dup(1)
        log_fp = open(EXPORT_LOG, "a")
        log_fd = log_fp.fileno()
        # mnnconvert ... > .export.log
        os.dup2(log_fd, 1)
        try:
            sys.argv = convert_args
            sys.argc = len(convert_args)
            if self.mnnconvert is None:
                from MNN.tools import mnnconvert
                mnnconvert.main()
            else:
                convert_args[0] = self.mnnconvert
                cmd = ' '.join(convert_args)
                message = os.popen(cmd).read()
                print(message)
            sys.argv = []
        finally:
            os.dup2(sfd, 1)
            os.close(log_fd)

    @spinner_run(f'convert onnx model to ')
    def onnx2mnn(self, onnx_path, mnn_path, args = []):
        convert_args = [
            '',
            '-f',
            'ONNX',
            '--modelFile',
            str(onnx_path),
            '--MNNModel',
            str(mnn_path),
            '--transformerFuse',
            '--allowCustomOp'
        ]
        convert_args += args
        self.convert(convert_args)
        return mnn_path

    def mnn2json(self, mnn_path, json_path):
        convert_args = [
            '',
            '-f',
            'MNN',
            '--modelFile',
            str(mnn_path),
            '--JsonFile',
            str(json_path)
        ]
        self.convert(convert_args)
        return json_path

    def json2mnn(self, json_path, mnn_path):
        convert_args = [
            '',
            '-f',
            'JSON',
            '--modelFile',
            str(json_path),
            '--MNNModel',
            str(mnn_path)
        ]
        self.convert(convert_args)
        return mnn_path

    def removeDupOps(self, mnn_path):
        convert_args = [
            '',
            '-f',
            'MNN',
            '--modelFile',
            str(mnn_path),
            '--MNNModel',
            str(mnn_path),
            '--optimizeLevel=1'
        ]
        self.convert(convert_args)
        return mnn_path

    def export(self, quant_bit = None, quant_block = None):
        if self.weight_ops is None:
            if quant_bit is None:
                quant_bit = self.quant_bit
            if quant_block is None:
                quant_block = self.quant_block
            if quant_bit == 16:
                quant_args = ['--fp16']
            else:
                quant_args = [
                    '--weightQuantBits',
                    str(quant_bit),
                    '--weightQuantBlock',
                    str(quant_block)
                ]
            self.onnx2mnn(self.onnx_model_path, self.mnn_model_path, quant_args)
        else:
            mnn_json = f'{self.mnn_model_path}.json'
            self.onnx2mnn(self.onnx_model_path, self.mnn_model_path)
            self.mnn2json(self.mnn_model_path, mnn_json)
            self.rebuild(mnn_json)
            self.json2mnn(mnn_json, self.mnn_model_path)
            self.removeDupOps(self.mnn_model_path)

    @spinner_run(f'quant model weight to ', True)
    def rebuild(self, json_path):
        mnn_graph = json.load(open(json_path, 'rt'))
        new_ops = []
        with open(self.mnn_weight_path, 'wb') as self.mnn_weight:
            for op in tqdm(mnn_graph['oplists'], 'Quant weights'):
                if op['type'] == 'Extra':
                    new_ops += self.rebuild_op(op, mnn_graph)
                else:
                    new_ops.append(op)
        mnn_graph['oplists'] = new_ops
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(mnn_graph, file, ensure_ascii=False, indent=4)
        return self.mnn_weight_path

    def quant(self, weight, quant_bit, quant_block, symmetric):
        if torch.cuda.is_available():
            weight = weight.cuda()
        if torch.backends.mps.is_available():
            weight = weight.to('mps')
        oc, ic = weight.shape
        if quant_block == 0:
            block_size = ic
        else:
            block_size = quant_block
        block_num = ic // block_size
        weight = weight.reshape(oc, block_num, block_size)
        offset = 1 << (quant_bit - 1)
        clip_max = offset - 1
        if symmetric:
            clip_min = -clip_max
            abs_max, _ = torch.max(torch.abs(weight), axis=-1, keepdims=True)
            scale = abs_max / clip_max
            q_weight = torch.round(weight / scale)
            q_weight = (torch.clamp(q_weight.flatten(), clip_min, clip_max) + offset).to(torch.uint8)
            alpha = scale.flatten()
        else:
            clip_min = -offset
            max_val, _ = torch.max(weight, axis=-1, keepdims=True)
            min_val, _ = torch.min(weight, axis=-1, keepdims=True)
            scale = (max_val - min_val) / (clip_max - clip_min)

            if False:
                q_weight = np.round((weight - min_val) / scale) + clip_min
                zeros =  min_val - scale * clip_min
            else:
                q_weight = torch.round(weight / scale) - torch.round(min_val / scale) + clip_min
                zeros =  (torch.round(min_val / scale) - clip_min) * scale
            q_weight = (torch.clamp(q_weight.flatten(), clip_min, clip_max) + offset).to(torch.uint8)
            alpha = torch.stack([zeros.flatten(), scale.flatten()], axis=-1).flatten()

        q_weight = q_weight.reshape(-1, 2)
        if quant_bit == 4:
            q_weight = q_weight[:, 0] * 16 + q_weight[:, 1]

        clip_min = 1

        if q_weight.device is not torch.device('cpu'):
            return q_weight.cpu(), alpha.float().cpu(), clip_min
        return q_weight, alpha.float(), clip_min

    def write_weight(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        return self.mnn_weight.write(data.tobytes())

    def write_header(self, ic, oc, quant_bit):
        dim_num = self.mnn_weight.write(b'\x02')
        shape_dtype = np.int16
        if oc > 65535 or ic > 65535:
            shape_dtype = np.int32
        dim_length = self.write_weight(np.array([oc, ic]).astype(shape_dtype))
        offset = 1 << (quant_bit - 1)
        weight_map = [i for i in range(-offset, offset)]
        if len(weight_map) == 256:
            weight_map.insert(0, 0)
        else:
            weight_map.insert(0, len(weight_map))
        map_length = self.write_weight(np.array(weight_map, dtype=np.int8))
        header_length = dim_num + dim_length + map_length
        return header_length, shape_dtype == np.int32

    def build_weight(self, linear, quant_bit, quant_block, symmetric):
        ic, oc = linear.in_features, linear.out_features
        if quant_bit == 16:
            half_weight = linear.weight.data.flatten().half()
            weight_len = self.write_weight(half_weight)
            alpha_len, q_min, shape_int32 = 0, 0, False
        else:
            assert(quant_bit in (4, 8))
            q_weight, alpha, q_min = self.quant(linear.weight.data, quant_bit, quant_block, symmetric)
            header_len, shape_int32 = self.write_header(ic, oc, quant_bit)
            weight_len = self.write_weight(q_weight) + header_len
            alpha_len = self.write_weight(alpha)
        if linear.bias is not None:
            bias = linear.bias.data.flatten().float()
            bias_length = self.write_weight(bias)
        else:
            bias_length = 0
            # bias = np.zeros([oc], dtype=np.float32)
            # bias_length = self.write_weight(bias)
        external = [self.mnn_weight_offset, weight_len, alpha_len, bias_length, 0]
        self.mnn_weight_offset += (weight_len + alpha_len + bias_length)
        return external, q_min, shape_int32, header_len

    def build_tensor(self, graph, tensor_name):
        tensor_idx = [len(graph['tensorName'])]
        graph['tensorName'].append(tensor_name)
        return tensor_idx

    def rebuild_op(self, op, graph):
        op_type = op['main']['type']
        if op_type == 'FakeLinear':
            return self.rebuild_linear(op, graph)
        if op_type == 'FusedAttention':
            return self.rebuild_attnention(op, graph)

    def rebuild_attnention(self, op, graph):
        attrs = op['main']['attr']
        for attr in attrs:
            if attr['key'] == 'name':
                name = attr['s']
        origin_input = op['inputIndexes']
        origin_output = op['outputIndexes']
        fused_attention = {
            "inputIndexes": origin_input,
            "main_type": "AttentionParam",
            "main": { "kv_cache": True },
            "name": name,
            "outputIndexes": origin_output,
            "type": "Attention",
            "defaultDimentionFormat": "NHWC"
        }
        return [fused_attention]

    def rebuild_linear(self, op, graph):
        attrs = op['main']['attr']
        for attr in attrs:
            if attr['key'] == 'name':
                name = attr['s']
            elif attr['key'] == "in_features":
                ic = attr["i"]
            elif attr['key'] == "out_features":
                oc = attr["i"]
            elif attr['key'] == "has_bias":
                has_bias = attr["i"]
        linear = self.weight_ops[name]
        assert(linear.in_features == ic and
               linear.out_features == oc and
               (linear.bias is not None) == has_bias)

        is_lm = 'lm_head' in name
        quant_bit = self.lm_quant_bit if is_lm else self.quant_bit
        block_size = ic if self.quant_block == 0 else self.quant_block
        external, q_min, shape_int32, header_len = self.build_weight(linear, quant_bit, self.quant_block, self.symmetric)
        if is_lm and self.config.tie_word_embeddings:
            weight_offset = external[0] + header_len
            alpha_offset = external[0] + external[1]
            alpha_size = external[2]
            self.config.llm_config['tie_embeddings'] = [weight_offset, alpha_offset, alpha_size, quant_bit, self.quant_block]

        origin_input = op['inputIndexes']
        origin_output = op['outputIndexes']
        # build new tensor
        pre_reshape_name = f'{name}/pre_reshape'
        pre_convert_name = f'{name}/pre_convert'
        conv_name = name
        post_convert_name = f'{name}/post_convert'
        post_reshape_name = f'{name}/post_reshape'
        pre_reshape_output = self.build_tensor(graph, pre_reshape_name)
        pre_convert_output = self.build_tensor(graph, pre_convert_name)
        conv_output = self.build_tensor(graph, conv_name)
        post_convert_output = self.build_tensor(graph, post_convert_name)
        # [batch, seq, hidden_size_i] -[Linear] -> [batch, seq, hidden_size_o]
        # [1, seq, hidden_size_i] ->[Reshape]-> [seq, hidden_size_i, 1, 1]
        # -[Convert]-[Convolution]-[Convert]-> [Reshape] -> [1, seq, hidden_size_o]
        pre_reshape = {
            "name": pre_reshape_name,
            "type": "Reshape",
            "inputIndexes": origin_input,
            "outputIndexes": pre_reshape_output,
            "main_type": "Reshape",
            "main": {
                "dims": [-1, ic, 1, 1],
                "dimType": "NCHW"
            },
            "defaultDimentionFormat": "NHWC"
        }
        pre_convert = {
            "name": pre_convert_name,
            "inputIndexes": pre_reshape_output,
            "outputIndexes": pre_convert_output,
            "type": "ConvertTensor",
            "main_type": "TensorConvertInfo",
            "main": {
                "source": "NCHW",
                "dest": "NC4HW4"
            },
            "defaultDimentionFormat": "NHWC"
        }

        if quant_bit == 16:
            quanParameter = { "type": 3 }
        else:
            if self.symmetric:
                aMin = 0
                readType = 0
            else:
                aMin = q_min
                readType = oc * (ic // block_size)

            quanParameter = {
                "quantScale": 1.0, "scaleIn": 0.0, "scaleOut": 0.0,
                "useInt32": False, "has_scaleInt": False, "shapeInt32": shape_int32,
                "type": 1, "aMax": 0, "aMin": aMin, "readType": readType, "weightSize": 0
            }
        conv_op = {
            "name": conv_name,
            "inputIndexes": pre_convert_output,
            "outputIndexes": conv_output,
            "type": "Convolution",
            "main_type": "Convolution2D",
            "main": {
                'common': {
                    'dilateX': 1, 'dilateY': 1, 'strideX': 1, 'strideY': 1,
                    'kernelX': 1, 'kernelY': 1, 'padX': 0, 'padY': 0, 'group': 1,
                    'outputCount': oc, 'relu': False, 'padMode': 'CAFFE',
                    'relu6': False, 'inputCount': ic, 'hasOutputShape': False
                },
                "quanParameter": quanParameter,
                "external": external
            },
            "defaultDimentionFormat": "NHWC"
        }
        post_convert = {
            "name": post_convert_name,
            "inputIndexes": conv_output,
            "outputIndexes": post_convert_output,
            "type": "ConvertTensor",
            "main_type": "TensorConvertInfo",
            "main": {
                "source": "NC4HW4",
                "dest": "NCHW"
            },
            "defaultDimentionFormat": "NHWC"
        }
        post_reshape = {
            "name": post_reshape_name,
            "type": "Reshape",
            "inputIndexes": post_convert_output,
            "outputIndexes": origin_output,
            "main_type": "Reshape",
            "main": {
                "dims": [1, -1, oc],
                "dimType": "NCHW"
            },
            "defaultDimentionFormat": "NHWC"
        }
        return [pre_reshape, pre_convert, conv_op, post_convert, post_reshape]

# some wrapper class for export
class Embedding(torch.nn.Module):
    def __init__(self, embed, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed = embed
        if config.model_type == 'gemma2':
            normalizer = torch.tensor(self.hidden_size**0.5)
            self.embed.weight.data *= normalizer

    def forward(self, input_ids):
        inputs_embeds = self.embed(input_ids).view(-1, 1, self.hidden_size)
        return inputs_embeds

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Attention(torch.nn.Module):
    def __init__(self, attn, layer_id, config):
        super().__init__()
        self.export_fused_attn = False
        self.fused_attn = FusedAttention(config.hidden_size, f'/layers.{layer_id}/self_attn/FusedAttention')
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        if isinstance(config.num_attention_heads, list):
            self.num_heads = config.num_attention_heads[layer_id]
            self.num_key_value_heads = config.num_key_value_heads[layer_id]
        else:
            self.head_dim = config.head_dim
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rotary = config.rotary

        ModelMapper.do_map(self, attn, config.model_map['attention'])

        if hasattr(self, 'qkv_proj') and self.qkv_proj is not None:
            # split qkv linear to q, k, v
            split_sizes = [self.hidden_size] * 3
            if self.qkv_proj.weight.shape[0] != self.hidden_size * 3:
                # M/GQA
                split_sizes = [
                    self.num_heads * self.head_dim,           # q_size
                    self.num_key_value_heads * self.head_dim, # k_size
                    self.num_key_value_heads * self.head_dim  # v_size
                ]

            self.q_proj = torch.nn.Linear(self.hidden_size, split_sizes[0])
            self.k_proj = torch.nn.Linear(self.hidden_size, split_sizes[1])
            self.v_proj = torch.nn.Linear(self.hidden_size, split_sizes[2])
            if config.model_type == 'chatglm':
                # chatglm-6b
                qkv_weight = self.qkv_proj.weight.data.view(self.num_heads, 3, self.head_dim, self.hidden_size)
                self.q_proj.weight.data = qkv_weight[:, 0, :, :].reshape(self.hidden_size, self.hidden_size)
                self.k_proj.weight.data = qkv_weight[:, 1, :, :].reshape(self.hidden_size, self.hidden_size)
                self.v_proj.weight.data = qkv_weight[:, 2, :, :].reshape(self.hidden_size, self.hidden_size)
                qkv_bias = self.qkv_proj.bias.data.view(self.num_heads, 3, self.head_dim)
                self.q_proj.bias.data = qkv_bias[:, 0, :].reshape(self.hidden_size)
                self.k_proj.bias.data = qkv_bias[:, 1, :].reshape(self.hidden_size)
                self.v_proj.bias.data = qkv_bias[:, 2, :].reshape(self.hidden_size)
            else:
                # other
                qw, kw, vw = torch.split(self.qkv_proj.weight, split_sizes)
                self.q_proj.weight.data = qw
                self.k_proj.weight.data = kw
                self.v_proj.weight.data = vw
                if self.qkv_proj.bias is not None:
                    qb, kb, vb = torch.split(self.qkv_proj.bias, split_sizes)
                    self.q_proj.bias.data = qb
                    self.k_proj.bias.data = kb
                    self.v_proj.bias.data = vb
                else:
                    self.q_proj.bias.data = torch.zeros(split_sizes[0])
                    self.k_proj.bias.data = torch.zeros(split_sizes[1])
                    self.v_proj.bias.data = torch.zeros(split_sizes[2])
            self.q_proj.weight.requires_grad = False
            self.k_proj.weight.requires_grad = False
            self.v_proj.weight.requires_grad = False
            self.q_proj.bias.requires_grad = False
            self.k_proj.bias.requires_grad = False
            self.v_proj.bias.requires_grad = False


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        if cross_attention_states is not None:
            hidden_states = cross_attention_states
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        # openelm model has qk_norm
        if hasattr(self, 'q_norm') and self.q_norm is not None and \
           hasattr(self, 'k_norm') and self.k_norm is not None :
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1]

        # rope
        cos, sin = rotary_pos_emb[0], rotary_pos_emb[1]
        query_states = self.rotary.apply_rotary_pos(query_states, cos, sin)
        key_states = self.rotary.apply_rotary_pos(key_states, cos, sin)

        if self.export_fused_attn:
            attn_output = self.fused_attn(query_states, key_states, value_states, attention_mask)
            attn_output = self.o_proj(attn_output)
            return attn_output, past_key_value

        # kv cache
        if past_key_value is not None:
            past_key, past_value = past_key_value[0], past_key_value[1]
            key_states = torch.cat((past_key, key_states), dim=1)
            value_states = torch.cat((past_value, value_states), dim=1)

        past_key_value = torch.stack((key_states, value_states))
        query_states = query_states.transpose(1, 2)
        key_states = key_states.permute([0, 2, 3, 1])
        value_states = value_states.transpose(1, 2)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #------- attention ----------
        # query_states @ key_states
        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
        # attention_mask
        if attention_mask.dtype in (torch.bool, torch.int32):
            # chatglm
            attn_weights.masked_fill_(attention_mask, -10000.0)
        else:
            attn_weights = attn_weights + attention_mask
        # upcast softmax to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights @ value_states
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class Rotary(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.rotary_dim = config.head_dim
        self.model_type = config.model_type
        if hasattr(config, 'rotary_dim'):
            self.rotary_dim = config.rotary_dim
        if self.model_type == 'chatglm':
            self.rotary_dim = config.head_dim // 2
        self.theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))

    def forward(self, position_ids):
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * self.theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        if self.model_type != 'chatglm2':
            rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

    def apply_rotary_pos(self, x, cos, sin):
        if self.model_type == 'chatglm':
            return self.chatglm_rotary_pos(x, cos, sin)
        if self.model_type == 'chatglm2':
            return self.chatglm2_rotary_pos(x, cos, sin)
        if self.model_type == 'phi-msft':
            return self.phi_rotary_pos(x, cos, sin)
        return self.llama_rotary_pos(x, cos, sin)

    def llama_rotary_pos(self, x, cos, sin):
        x = (x * cos) + (rotate_half(x) * sin)
        return x

    def phi_rotary_pos(self, x, cos, sin):
        x, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        x = (x * cos) + (rotate_half(x) * sin)
        return torch.cat((x, x_pass), dim=-1)

    def chatglm2_rotary_pos(self, x, cos, sin):
        x, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        b, s, n, h = x.shape
        xshaped = x.view(b, s, n, h//2, 2)
        x = torch.concat(
            [
                xshaped[..., 0] * cos - xshaped[..., 1] * sin,
                xshaped[..., 1] * cos + xshaped[..., 0] * sin,
            ],
            -1,
        )
        return torch.cat((x, x_pass), dim=-1)

    def chatglm_rotary_pos(self, x, cos, sin):
        seq = x.shape[1]
        x1, x2 = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        cos1, sin1 = cos[:, :seq, ...], sin[:, :seq, ...]
        cos2, sin2 = cos[:, seq:, ...], sin[:, seq:, ...]
        x1 = (x1 * cos1) + (rotate_half(x1) * sin1)
        x2 = (x2 * cos2) + (rotate_half(x2) * sin2)
        return torch.cat((x1, x2), dim=-1)

class VisionRotary(Rotary):
    def __init__(self, config):
        super().__init__(config)

    # support [h_pos, w_pos]
    def forward(self, position_ids):
        # [2, patch_len, 1]
        position_ids = position_ids.float().unsqueeze(-1)
        idx_theta = position_ids * self.theta
        # [patch_len, rotary_dim]
        idx_theta = idx_theta.permute(1, 0, 2).reshape(-1, self.rotary_dim)
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

class Decoder(torch.nn.Module):
    def __init__(self, decoder, layer_id, config):
        super().__init__()
        self.cross_decoder = False
        ModelMapper.do_map(self, decoder, config.model_map['decoder'])
        # mllama has cross_attn
        if hasattr(self, 'cross_attn') and self.cross_attn is not None:
            self.cross_decoder = True
            self.self_attn = Attention(self.cross_attn, layer_id, config)
        else:
            self.self_attn = Attention(self.self_attn, layer_id, config)
        self.hidden_size = config.hidden_size
        # chatglm
        self.alpha = (2 * config.num_hidden_layers) ** 0.5 if config.model_type == 'chatglm' else 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        norm_hidden_states = hidden_states
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cross_attention_states=cross_attention_states,
        )
        # Fully Connected
        if not hasattr(self, 'post_attention_layernorm'):
            # phi
            feed_forward_hidden_states = self.mlp(norm_hidden_states)
            hidden_states = hidden_states + feed_forward_hidden_states + residual
        elif self.alpha != 1.0:
            # chatglm-6b
            hidden_states = norm_hidden_states * self.alpha + hidden_states
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_input * self.alpha + mlp_output
        elif hasattr(self, 'pre_feedforward_layernorm'):
            # gemma2
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
        elif cross_attention_mask is not None and self.cross_decoder:
            hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = cross_attention_mask * hidden_states
            hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states
        else:
            # general
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states, present_key_value

class Lm(torch.nn.Module):
    def __init__(self, lm_, final_layernorm_, config):
        super().__init__()
        self.final_layernorm = final_layernorm_
        self.lm = lm_
        self.hidden_size = config.hidden_size
        self.ppl = config.args.ppl

    def forward(self, hidden_states):
        if not self.ppl:
            # just need last logit for predict next token
            hidden_states = hidden_states.view(-1, self.hidden_size)[-1].view(1, 1, self.hidden_size)
        hidden_states = self.final_layernorm(hidden_states)
        m_logits = self.lm(hidden_states)
        return m_logits

# visual start
class Visual(torch.nn.Module):
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
    def get_visual(model_type):
        visual_models = {
            'qwen': QwenVisual,
            'qwen2_vl': Qwen2Visual,
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

class QwenVisual(Visual):
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

class Qwen2Visual(Visual):
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

class MllamaVision(Visual):
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
# visual end

# audio start
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
# audio end

class LlmExporter(torch.nn.Module):
    '''
    Base class for all llm model export. Inherits from [`torch.nn.Module`].
    '''

    def __init__(self, args):
        super().__init__()
        self.init_from_args(args)
        self.load_model(args.path)

    def init_from_args(self, args):
        self.args = args
        self.max_length = 128
        self.stop_ids = []
        self.dst_name = 'llm'
        # load config from args
        self.onnx_path = os.path.join(self.args.dst_path, 'onnx')
        if self.args.tokenizer_path is None:
            self.args.tokenizer_path = self.args.path
        if args.lm_quant_bit is None:
            self.args.lm_quant_bit = self.args.quant_bit
        # init export dst dir
        if not os.path.exists(self.args.dst_path):
            os.makedirs(self.args.dst_path)
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)

    def load_pretrained(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, trust_remote_code=True, use_fast=False)
        if 'Qwen2-VL' in model_path:
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype='auto').eval()
        elif 'Qwen2-Audio' in model_path:
            from transformers import Qwen2AudioForConditionalGeneration
            self.audio = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
            self.model = self.audio.language_model
        elif 'Llama-3.2' in model_path and 'Vision' in model_path:
            from transformers import MllamaForConditionalGeneration
            self.model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype='auto').eval()
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', trust_remote_code=True).eval()
            except:
                self.model = AutoModel.from_pretrained(model_path, torch_dtype='auto', trust_remote_code=True).eval()
        self.config = self.model.config
        if self.args.lora_path is not None:
            from peft import PeftModel
            adapter = PeftModel.from_pretrained(self.model, model_id=self.args.lora_path)
            self.model = adapter.merge_and_unload(progressbar=True)

    @staticmethod
    def has_attr(obj, attr):
        return hasattr(obj, attr) and getattr(obj, attr) is not None

    @spinner_run(f'load pretrained model ', True)
    def load_model(self, model_path):
        self.load_pretrained(model_path)
        self.attention_mask_type = 'float'
        # load tokenizer info
        self.stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'im_end_id'):
            self.stop_ids.append(self.tokenizer.im_end_id)
        eot_id = self.tokenizer.encode('<|eot_id|>')
        if len(eot_id) == 1:
            self.stop_ids.append(eot_id[0])
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            eos_token_id = self.model.generation_config.eos_token_id
            from collections.abc import Iterable
            if isinstance(eos_token_id, int):
                self.stop_ids.append(eos_token_id)
            elif isinstance(eos_token_id, Iterable):
                for id in eos_token_id:
                    self.stop_ids.append(id)
        self.stop_ids = [stop_id for stop_id in self.stop_ids if stop_id is not None]
        self.stop_ids = list(set(self.stop_ids))
        self.visual = None
        model_mapper = ModelMapper()

        self.tie_word_embeddings = (hasattr(self.config, 'tie_word_embeddings') and self.config.tie_word_embeddings)
        self.model_type, self.model_map = model_mapper.get_map(self.config)

        if self.args.awq:
            self.model.float()
        if self.args.export is not None:
            # set norm's weight as float for export
            def visit_module(module):
                if not isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
                    module.float()
                for name, child in module.named_children():
                    visit_module(child)
            visit_module(self.model)
        # print(self.config, self.model_type, self.model_map, self.model)
        # exit(0)
        # load config info
        ModelMapper.do_map(self, self.config, self.model_map['config'])
        if not hasattr(self, 'num_key_value_heads') or self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if not hasattr(self, 'rope_theta') or self.rope_theta is None:
            self.rope_theta = 10000.0
        if not hasattr(self, 'head_dim') or self.head_dim is None:
            if isinstance(self.num_attention_heads, list):
                self.head_dim = [self.hidden_size // atten_head for atten_head in self.num_attention_heads]
            else:
                self.head_dim = self.hidden_size // self.num_attention_heads
        # some export info
        if isinstance(self.num_attention_heads, list):
            self.past_kv_shape = [self.num_hidden_layers, 2, 1, 0, self.num_key_value_heads[0], self.head_dim]
        else:
            self.past_kv_shape = [self.num_hidden_layers, 2, 1, 0, self.num_key_value_heads, self.head_dim]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 1: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 1: "seq_len" },
            "past_key_values" : { 3: "history_len" }
        }
        self.llm_config = {
            'hidden_size' : self.hidden_size,
            'layer_nums' : self.num_hidden_layers,
            'attention_mask': self.attention_mask_type,
            'key_value_shape': self.past_kv_shape[1:],
            "prompt_template": self.build_prompt('%s'),
            'is_visual': False
        }
        # load modules
        ModelMapper.do_map(self, self.model, self.model_map['model'])
        # rebuild modules
        if self.lm_ is None:
            out_features, in_features = self.embed_.weight.shape
            self.lm_ = torch.nn.Linear(in_features, out_features)
            self.lm_.weight = self.embed_.weight

        if self.embed_.weight is self.lm_.weight:
            import copy
            embed_copy = copy.deepcopy(self.embed_)
            self.embed = Embedding(embed_copy, self)
        else:
            self.embed = Embedding(self.embed_, self)
        # Rotary
        self.rotary = Rotary(self)
        self.blocks = []
        for block in self.blocks_.children():
            layer_id = len(self.blocks)
            self.blocks.append(Decoder(block, layer_id, self))
        self.lm = Lm(self.lm_, self.final_layernorm_, self)
        # visual model
        if self.visual is not None:
            if self.args.export is not None:
                self.visual.float()
            self.visual = Visual.get_visual(self.model_type)(self.visual, self)
        if hasattr(self, 'audio') and self.audio is not None:
            self.audio = Audio.get_audio(self.audio.config.model_type)(self.audio, self)
        else:
            self.audio = None
        return model_path

    def get_attention_mask(self) -> torch.Tensor:
        if self.model_type == 'chatglm':
            return self.chatglm_attention_mask()
        if self.token_len:
            return torch.zeros([1, 1, 1, self.seq_len], dtype=torch.float32)
        return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min

    def get_position_ids(self) -> torch.Tensor:
        if self.model_type == 'chatglm':
            return self.chatglm_position_ids()
        if self.token_len:
            return torch.tensor([[self.seq_len - 1]], dtype=torch.int)
        return torch.arange(self.seq_len, dtype=torch.int).unsqueeze(0)

    def chatglm_attention_mask(self):
        if self.token_len:
            return torch.zeros([1]).bool().reshape([1, 1, 1, 1])
        attention_mask = torch.zeros([self.seq_len, self.seq_len], dtype=torch.bool)
        for i in range(self.seq_len - 1):
            attention_mask[i][-1] = True
        attention_mask = attention_mask.reshape([1, 1, self.seq_len, self.seq_len])
        return attention_mask

    def chatglm_position_ids(self):
        if self.token_len:
            return torch.tensor([self.context_len, self.token_len + 1]).reshape([1, 2, 1])
        position_ids_0 = torch.arange(self.seq_len, dtype=torch.int)
        position_ids_1 = torch.zeros(self.seq_len, dtype=torch.int)
        position_ids_0[-1] = position_ids_0[-2]
        position_ids_1[-1] = 1
        position_ids = torch.stack([position_ids_0, position_ids_1]).view(1, 2, -1)
        return position_ids

    def visual_embed(self, input_ids):
        return self.visual.embed(input_ids)

    def audio_embed(self, input_ids):
        return self.audio.embed(input_ids)

    def embedding(self, input_ids):
        if self.visual is not None and self.token_len == 0:
            input_embeds = self.visual_embed(input_ids)
        elif self.audio is not None and self.token_len == 0:
            input_embeds = self.audio_embed(input_ids)
        else:
            input_embeds = self.embed(input_ids)
        return input_embeds

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: Optional[list[torch.Tensor]] = None,
                cross_attention_states: Optional[torch.Tensor] = None,
                cross_attention_mask: Optional[torch.Tensor] = None,
                ):
        hidden_states = input_ids # llm forward without embedding
        presents = [None for i in range(self.num_hidden_layers)]
        rotary_pos_emb = self.rotary(position_ids)
        if self.args.test and rotary_pos_emb.dtype != hidden_states.dtype:
            rotary_pos_emb = rotary_pos_emb.type(hidden_states.dtype)
        for i in range(self.num_hidden_layers):
            if self.blocks[i].cross_decoder and cross_attention_states is None:
                continue
            hidden_states, kv = self.blocks[i](hidden_states, rotary_pos_emb, attention_mask, past_key_values[i])
            presents[i] = kv
        logits = self.lm(hidden_states)
        if not self.args.ppl:
            logits = logits.reshape(-1)
        if presents[0].shape == presents[-1].shape and None not in presents:
            presents = torch.stack(presents)
        self.seq_len += 1
        self.token_len += 1
        return logits, presents

    # some test functions
    def build_prompt(self, query):
        # just for test
        if 'Qwen2' in self.args.path or 'QwQ' in self.args.path or 'reader' in self.args.path:
            return f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
        if 'Qwen' in self.args.path:
            return f'\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
        if 'Baichuan2' in self.args.path:
            return f'<reserved_106>{query}<reserved_107>'
        if 'internlm' in self.args.path:
            return f'<|User|>:{query}<eoh>\n<|Bot|>:'
        if 'TinyLlama' in self.args.path:
            return f'<s><|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\n{query}</s>\n<|assistant|>\n'
        if 'Yi' in self.args.path:
            return f'<|im_start|> user\n{query}<|im_end|>\n<|im_start|> assistant\n'
        if 'deepseek' in self.args.path:
            return f'<|begin_of_sentence|>User: {query}\n\nAssistant:'
        if 'Llama-3.1' in self.args.path:
            return f'<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        if 'Llama-3' in self.args.path:
            return f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        if 'Llama-2' in self.args.path:
            return f'[INST]{query}[/INST]'
        if 'chatglm2' in self.args.path:
            return f'[Round 1]\n\n问：{query}\n\n答：'
        if 'chatglm3' in self.args.path or 'glm-4' in self.args.path:
            return f'<|user|>\n{query}\n<|assistant|>\n'
        if 'chatglm' in self.args.path:
            return f'{query}[gMASK]<sop>'
        if 'phi-2' in self.args.path:
            return f'Instruct: {query}\nOutput:'
        if 'gemma-2' in self.args.path:
            return f'<bos><start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n'
        if 'OpenELM' in self.args.path:
            return f'<s>{query}'
        if 'SmolLM2' in self.args.path:
            return f'<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
        return query

    def str_to_ids(self, prompt):
        if self.visual is not None:
            return self.visual.str_to_ids(prompt)
        if self.audio is not None:
            return self.audio.str_to_ids(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        return input_ids

    def id_to_str(self, token_id):
        try:
            word = self.tokenizer.decode(int(token_id))
        except:
            def contains_replacement(text): return '\uFFFD' in text
            def decode_id(token_id):
                return self.tokenizer.convert_tokens_to_string(
                        self.tokenizer._convert_id_to_token(int(token_id)))
            def decode_ids(token_ids):
                return self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(token_ids))
            word = decode_id(int(token_id))
            # Smollm tokenizer will produce half chinese character, using buffer to decode
            if contains_replacement(word):
                self.decode_buffer.append(token_id)
                buffer_txt = decode_ids(self.decode_buffer)
                if not contains_replacement(buffer_txt):
                    word = buffer_txt
                    self.decode_buffer.clear()
                else:
                    word = ''
        return word

    def response(self, query):
        # self.imitate_quant()
        self.decode_buffer = []
        prompt = self.build_prompt(query)
        input_ids = self.str_to_ids(prompt)
        if self.visual is not None:
            cross_attention_states = self.visual.cross_attention_states
            cross_attention_mask = self.visual.cross_attention_mask
        else:
            cross_attention_states = None
            cross_attention_mask = None
        self.seq_len = input_ids.numel()
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [None for i in range(self.num_hidden_layers)]
        token_id = input_ids
        while self.token_len < self.max_length:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            input_ids = self.embedding(token_id)
            logits, past_key_values = self.forward(input_ids,
                                                   attention_mask,
                                                   position_ids,
                                                   past_key_values,
                                                   cross_attention_states,
                                                   cross_attention_mask)
            token_id = torch.argmax(logits)
            if token_id in self.stop_ids:
                print("", end='\n')
                break
            word = self.id_to_str(token_id)
            print(word, end="", flush=True)

    @spinner_run(f'export visual to ')
    def export_visual(self):
        if self.visual is None:
            return
        return self.visual.export(self.onnx_path)

    @spinner_run(f'export audio to ')
    def export_audio(self):
        if self.audio is None:
            return
        input_features = torch.randn((1, self.audio.feature_size, self.audio.max_length))
        model = self.audio.float()
        onnx_model = f'{self.onnx_path}/audio.onnx'
        torch.onnx.export(model, (input_features),
                        onnx_model,
                        input_names=['input_features'],
                        output_names=['audio_embeds'],
                        dynamic_axes={"input_features": {
                            0: "size"
                        }},
                        do_constant_folding=True,
                        verbose=False,
                        opset_version=15)
        return onnx_model

    @spinner_run(f'export embedding to ')
    def export_embed(self):
        import ctypes
        if hasattr(self, 'word_embeddings'):
            # embedding model's embed
            tensor_data = self.word_embeddings.weight.data.bfloat16()
        else:
            tensor_data = self.embed.embed.weight.data.bfloat16()
        data_ptr = tensor_data.untyped_storage().data_ptr()
        buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
        embedding_file = f'{self.args.dst_path}/embeddings_bf16.bin'
        with open(embedding_file, 'wb') as f:
            f.write(buffer)
        return embedding_file

    @spinner_run(f'export config to ')
    def export_config(self, mnn_config = False):
        config_json = f'{self.args.dst_path}/llm_config.json'
        with open(config_json, 'w', encoding='utf-8') as f:
            json.dump(self.llm_config, f, ensure_ascii=False, indent=4)
        if not mnn_config:
            return config_json
        with open(f'{self.args.dst_path}/config.json', 'w', encoding='utf-8') as f:
            config = {
                "llm_model": f"{self.dst_name}.mnn",
                "llm_weight": f"{self.dst_name}.mnn.weight",
                "backend_type": "cpu",
                "thread_num": 4,
                "precision": "low",
                "memory": "low"
            }
            if self.visual is not None or self.audio is not None:
                config['mllm'] = {
                    'backend_type': "cpu",
                    "thread_num": 4,
                    "precision": "low",
                    "memory": "low"
                }
            json.dump(config, f, ensure_ascii=False, indent=4)
        return config_json

    def imitate_quant(self):
        def quant_dequant(linear, quant_bit = self.args.quant_bit, quant_block = self.args.quant_block):
            weight = linear.weight.data
            oc, ic = weight.shape
            if quant_block == 0:
                block_size = ic
            else:
                block_size = quant_block
            block_num = ic // block_size
            weight = weight.reshape(oc, block_num, block_size)
            max_val = torch.max(weight, axis=-1, keepdims=True).values
            min_val = torch.min(weight, axis=-1, keepdims=True).values
            offset = 1 << (quant_bit - 1)
            clip_max = offset - 1
            clip_min = -offset
            scale = (max_val - min_val) / (clip_max - clip_min)
            q_weight = torch.round((weight - min_val) / scale) + clip_min
            q_weight = torch.clip(q_weight, clip_min, clip_max)
            dq_weight = (q_weight - clip_min) * scale + min_val
            dq_weight = dq_weight.reshape(oc, ic).float()
            linear.weight.data = dq_weight
            return linear
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                for name, child in self.blocks[i].self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.blocks[i].self_attn, name, quant_dequant(child))
                for name, child in self.blocks[i].mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.blocks[i].mlp, name, quant_dequant(child))
            self.lm.lm = quant_dequant(self.lm.lm)

    def unload_param(self):
        self.unloaded_ops = {}
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        # replace linear with fakelinear to save export memory and time
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                # different kv cache shape in different layers
                if isinstance(self.num_attention_heads, list):
                    self.blocks[i].self_attn.export_fused_attn = True
                for name, child in self.blocks[i].self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.blocks[i].self_attn, name, build_faker(child, f'/layers.{i}/self_attn/{name}/Linear'))
                for name, child in self.blocks[i].mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.blocks[i].mlp, name, build_faker(child, f'/layers.{i}/mlp/{name}/Linear'))
            self.lm.lm = build_faker(self.lm.lm, f'/lm/lm_head/Linear')

    @spinner_run(f'export model weight to ')
    def onnx_load_param(self, onnx_path):
        return OnnxRebuilder(onnx_path, self.unloaded_ops).rebuild()

    @spinner_run(f'slim the graph of ')
    def slim_onnx(self, onnx_model):
        import onnxslim
        model = onnxslim.slim(onnx_model)
        onnx.save(model, onnx_model)
        return onnx_model

    @spinner_run(f'export onnx model to ')
    def export_onnx(self):
        # unload linear weight to save export memory
        self.unload_param()
        model = self
        self.seq_len = 3
        self.token_len = 0
        input_ids = torch.arange(3, dtype=torch.long)
        attention_mask =  self.get_attention_mask()
        position_ids = self.get_position_ids()
        onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
        input_ids = self.embedding(input_ids)
        past_key_values = torch.zeros(self.past_kv_shape)

        # export to onnx
        torch.onnx.export(
            model, (input_ids, attention_mask, position_ids, past_key_values),
            onnx_model,
            input_names=[
                'input_ids', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['logits', 'presents'],
            dynamic_axes=self.model_dynamic_axes,
            do_constant_folding=True,
            verbose=False,
            opset_version=15)
        return onnx_model

    def awq_quant(self):
        self.awq_quantizer = AwqQuantizer(self)
        self.awq_quantizer.quantize()
        self.is_awq_quantized = True

    def export(self, export_type):
        if self.args.awq:
            self.awq_quant()
        export_mnn = export_type == 'mnn'
        # export tokenizer
        self.export_tokenizer()
        if export_mnn and self.tie_word_embeddings:
            pass # mnn tie_word_embeddings need't export embedding
        else:
            self.export_embed()
        if self.visual:
            visual_onnx = self.export_visual()
            if export_mnn:
                MNNConveter(visual_onnx, None, self).export(quant_bit=self.visual.quant_bit)
        if self.audio:
            audio_onnx = self.export_audio()
            if export_mnn:
                MNNConveter(audio_onnx, None, self).export(quant_bit=self.audio.quant_bit)
        # export graph to llm.onnx
        onnx_model = self.export_onnx()
        if self.args.onnx_slim:
            self.slim_onnx(onnx_model)
        if export_mnn:
            # convert onnx to mnn and quant weight
            MNNConveter(onnx_model, self.unloaded_ops, self).export()
            # delete onnx file
            if os.path.exists(onnx_model):
                import glob
                try:
                    for file in glob.glob(f'{self.onnx_path}/*'):
                        os.remove(file)
                    os.rmdir(self.onnx_path)
                except Exception as e:
                    print(f"remove onnx error: {e}")
        else:
            # export weight to llm.onnx.data
            self.onnx_load_param(onnx_model)
        # export llm_config.json and config.json
        self.export_config(export_mnn)


    @spinner_run(f'export tokenizer to ')
    def export_tokenizer(self):
        # load tokenizer file
        tokenizer_model = os.path.join(self.args.tokenizer_path, 'tokenizer.model')
        ice_text_model = os.path.join(self.args.tokenizer_path, 'ice_text.model')
        try:
            import sentencepiece as spm
            if os.path.exists(tokenizer_model):
                self.sp_model = spm.SentencePieceProcessor(tokenizer_model)
            elif os.path.exists(ice_text_model):
                self.sp_model = spm.SentencePieceProcessor(ice_text_model)
            else:
                self.sp_model = None
        except:
            self.sp_model = None
        merge_file = os.path.join(self.args.path, 'merges.txt')
        if os.path.exists(merge_file):
            self.merge_txt = merge_file
        else:
            self.merge_txt = None
        # TOKENIZER MAGIC NUMBER
        MAGIC_NUMBER = 430
        # TOKENIZER TYPE
        SENTENCEPIECE = 0; TIKTOIKEN = 1; BERT = 2; HUGGINGFACE = 3
        def write_line(fp, *args):
            for arg in args:
                for token in arg:
                    fp.write(str(token) + ' ')
            fp.write('\n')
        def write_header(fp, type, speicals, prefix = []):
            fp.write(f'{MAGIC_NUMBER} {type}\n')
            fp.write(f'{len(speicals)} {len(self.stop_ids)} {len(prefix)}\n')
            write_line(fp, speicals, self.stop_ids, prefix)

        file_path = os.path.join(self.args.dst_path, "tokenizer.txt")
        special_list = list(self.tokenizer.added_tokens_decoder.keys())
        if hasattr(self.tokenizer, 'special_tokens'):
            for k, v in self.tokenizer.special_tokens.items():
                special_list.append(v)
        if hasattr(self.tokenizer, 'gmask_token_id'):
            special_list.append(self.tokenizer.gmask_token_id)
        vocab_list = []
        prefix_list = []
        if hasattr(self.tokenizer, 'get_prefix_tokens'):
            prefix_list = self.tokenizer.get_prefix_tokens()
        if len(prefix_list) == 0:
            test_txt = 'A'
            ids = self.tokenizer.encode(test_txt)
            get_txt = self.tokenizer.decode(ids[-1])
            if len(ids) > 1 and get_txt == test_txt:
                prefix_list += ids[:-1]

        if self.sp_model is not None:
            # senetencepiece
            NORMAL = 1; UNKNOWN = 2; CONTROL = 3
            USER_DEFINED = 4; UNUSED = 5; BYTE = 6
            for i in range(self.sp_model.GetPieceSize()):
                token = self.sp_model.IdToPiece(i)
                score = self.sp_model.GetScore(i)
                token_type = NORMAL
                if self.sp_model.IsUnknown(i):
                    token_type = UNKNOWN
                elif self.sp_model.IsControl(i):
                    token_type = CONTROL
                elif self.sp_model.IsUnused(i):
                    token_type = UNUSED
                elif self.sp_model.IsByte(i):
                    token_type = BYTE
                if self.args.path == 'Chatglm_6b':
                    if '<n>' in token: token = '\n'
                    if '<|tab|>' in token: token = '\t'
                    if '<|blank_' in token: token = ' ' * int(token[8:token.find('|>')])
                if '▁' in token: token = token.replace('▁', ' ')
                token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                vocab_list.append(f'{token_encode} {score} {token_type}\n')
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, SENTENCEPIECE, special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif hasattr(self.tokenizer, 'mergeable_ranks'):
            # tikton
            vocab_list = []
            for k, v in self.tokenizer.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + "\n"
                vocab_list.append(line)
            if hasattr(self.tokenizer, 'special_tokens'):
                for k, v in self.tokenizer.special_tokens.items():
                    line = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                for k, v in self.tokenizer.added_tokens_decoder.items():
                    line = base64.b64encode(v.__str__().encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, TIKTOIKEN, special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif self.merge_txt is not None:
            # huggingface tokenizer
            merge_list = []
            vocab = self.tokenizer.get_vocab()
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            vocab_list = ['<unk>' for i in range(len(vocab))]
            # load vocab
            for k, v in vocab.items():
                vocab_list[int(v)] = k
            # load merge
            with open(self.merge_txt, 'rt') as merge:
                for line in merge.readlines():
                    merge_list.append(line)
            # write to tokenizer.txt
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, HUGGINGFACE, special_list)
                fp.write(f'{len(vocab_list)} {len(merge_list)}\n')
                for v in vocab_list:
                    fp.write(v + '\n')
                for m in merge_list:
                    fp.write(m)
        else:
            # tiktoken or bert
            if 'bert' in type(self.tokenizer).__name__.lower():
                tokenizer_type = BERT
            else:
                tokenizer_type = TIKTOIKEN
            # bert tokenizer
            def unicode_to_byte(u: int):
                if u >= 256 and u <= 288:
                    return u - 256
                if u >= 289 and u <= 322:
                    return u - 162
                if u == 323:
                    return 173
                if u == 65372: # |
                    return 124
                if u == 9601:  # _
                    return 95
                return u
            vocab = self.tokenizer.get_vocab()
            vocab_list = ['<unk>' for i in range(len(vocab))]
            for k, v in vocab.items():
                try:
                    vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k]).decode('utf-8', errors='ignore')
                except:
                    vocab_list[int(v)] = k
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, tokenizer_type, special_list)
                fp.write(f'{len(vocab_list)}\n')
                for v in vocab_list:
                    line = base64.b64encode(v.encode('utf-8')).decode("utf8") + "\n"
                    fp.write(line)
        return file_path


class EmbeddingExporter(LlmExporter):
    def __init__(self, args):
        super().__init__(args)
        self.dst_name = 'embedding'

    def word_embed(self, input_ids):
        return self.word_embeddings(input_ids.view(1, -1))

    def bge_forward(self, inputs_embeds, position_ids, attention_mask):
        # bert absolute position
        inputs_embeds = inputs_embeds.reshape(1, -1, self.hidden_size)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings + self.token_type_embeddings
        hidden_states = self.embedding_layernorm(embeddings)
        for i in range(self.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states, attention_mask)[0]
        sentence_embeddings = hidden_states[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def gte_forward(self, inputs_embeds, position_ids, attention_mask):
        # rope position
        inputs_embeds = inputs_embeds.reshape(1, -1, self.hidden_size)
        freqs = position_ids.float().reshape(-1, 1) * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        rope_embeds = torch.stack([emb.cos(), emb.sin()]).unsqueeze(-2).unsqueeze(1)
        attention_bias = 1 - attention_mask.float()
        hidden_states = self.embedding_layernorm(inputs_embeds + self.token_type_embeddings)
        for i in range(self.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states, attention_bias, rope_embeds)[0]
        sentence_embeddings = hidden_states[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def forward(self, inputs_embeds, position_ids, attention_mask):
        if self.model_type == 'bert':
            return self.bge_forward(inputs_embeds, position_ids, attention_mask)
        if self.model_type == 'new':
            return self.gte_forward(inputs_embeds, position_ids, attention_mask)
        raise RuntimeError(f'Not support embedding model: {self.model_type}!')

    def response(self, query):
        self.eval()
        input_ids = self.tokenizer(query)['input_ids']
        self.seq_len = len(input_ids)
        input_ids = torch.tensor(input_ids)
        position_ids = self.get_position_ids()
        attention_mask = self.get_attention_mask()
        inputs_embeds = self.word_embed(input_ids)
        res = self.forward(inputs_embeds, position_ids, attention_mask)
        # print(res)
        return res

    @spinner_run(f'load pretrained model ')
    def load_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_path)
        self.config._attn_implementation = 'eager'
        self.model = AutoModel.from_config(self.config)
        transformer = self.model.encoder
        self.model_type = self.config.model_type
        self.lm_ = self.model.pooler
        self.embed_ = self.model.embeddings
        self.word_embeddings = self.embed_.word_embeddings
        self.token_type_embeddings = self.embed_.token_type_embeddings.weight.data[0]
        self.embedding_layernorm = self.embed_.LayerNorm
        if hasattr(self.embed_, 'position_embeddings'):
            self.position_embeddings = self.embed_.position_embeddings
        self.hidden_size = self.word_embeddings.weight.shape[-1]
        self.blocks = transformer.layer
        if self.model_type == 'new':
            self.inv_freq = self.embed_.rotary_emb.inv_freq
        # some wrapper
        self.stop_ids = []
        self.num_hidden_layers = len(self.blocks)
        self.embed = self.embed_
        self.lm = self.lm_
        # some config for export
        self.model_dynamic_axes = {
            "input_ids" : { 1: "seq_len" },
            "position_ids" : { 1: "seq_len" },
            "attention_mask" : { 3: "seq_len" }
        }
        self.attention_mask_type = 'int'
        self.llm_config = {
            'hidden_size' : self.hidden_size,
            'layer_nums' : self.num_hidden_layers,
            'attention_mask': self.attention_mask_type,
            'key_value_shape': [],
            "prompt_template": self.build_prompt('%s'),
            'is_visual': False
        }
        return model_path

    @spinner_run(f'export onnx model to ')
    def export_onnx(self):
        model = self.eval()
        self.seq_len = 3
        input_ids = torch.arange(3, dtype=torch.long)
        position_ids = self.get_position_ids()
        attention_mask = self.get_attention_mask()
        inputs_embeds = self.word_embed(input_ids)
        onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
        torch.onnx.export(
            model, (inputs_embeds, position_ids, attention_mask),
            onnx_model,
            input_names=[
                'input_ids',
                'position_ids',
                'attention_mask'
            ],
            output_names=['sentence_embeddings'],
            dynamic_axes=self.model_dynamic_axes,
            do_constant_folding=True,
            opset_version=15)
        return onnx_model

    def export(self, export_type):
        export_mnn = 'mnn' in export_type
        self.export_tokenizer()
        self.export_config(export_mnn)
        self.export_embed()
        onnx_model = self.export_onnx()
        if self.args.onnx_slim:
            self.slim_onnx(onnx_model)
        if export_mnn:
            MNNConveter(onnx_model, None, self).export()

    def build_prompt(self, query):
        if self.model_type == 'bert':
            return f'[CLS]{query}[SEP]'
        if self.model_type == 'new':
            return f'<s> {query}</s>'

    def get_position_ids(self) -> torch.Tensor:
        return torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)

    def get_attention_mask(self) -> torch.Tensor:
        return torch.ones([1, 1, 1, self.seq_len], dtype=torch.long)


def export(path,
           type = None,
           lora_path = None,
           dst_path = './model',
           export = 'onnx',
           onnx_slim = False,
           quant_bit = 4,
           quant_block = 128,
           lm_quant_bit = None):
    args = argparse.Namespace()
    for k, v in {
        'path': path,
        'type': type,
        'lora_path': lora_path,
        'dst_path': dst_path,
        'export': export,
        'onnx_slim': onnx_slim,
        'quant_bit': quant_bit,
        'quant_block': quant_block,
        'lm_quant_bit': lm_quant_bit
    }.items():
        setattr(args, k, v)
    if 'bge' in path:
        llm_exporter = EmbeddingExporter(args)
    else:
        llm_exporter = LlmExporter(args)
    # export
    llm_exporter.export(export)

def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', type=str, required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--type', type=str, default=None,
                        help='type(`str`, *optional*):'
                        '\n\tThe pretrain llm model type.'
                        )
    parser.add_argument('--tokenizer_path', type=str, default=None, help='tokenizer path, defaut is `None` mean using `--path` value.')
    parser.add_argument('--lora_path', type=str, default=None, help='lora path, defaut is `None` mean not apply lora.')
    parser.add_argument('--dst_path', type=str, default='./model', help='export onnx/mnn model to path, defaut is `./model`.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to print verbose.')
    parser.add_argument('--test', type=str, help='test model inference with query `TEST`.')
    parser.add_argument('--export', type=str, default=None, help='export model to an onnx/mnn model.')
    parser.add_argument('--onnx_slim', action='store_true', help='Whether or not to use onnx-slim.')
    parser.add_argument('--quant_bit', type=int, default=4, help='mnn quant bit, 4 or 8, default is 4.')
    parser.add_argument('--quant_block', type=int, default=128, help='mnn quant block, default is 0 mean channle-wise.')
    parser.add_argument('--lm_quant_bit', type=int, default=None, help='mnn lm_head quant bit, 4 or 8, default is `quant_bit`.')
    parser.add_argument('--mnnconvert', type=str, default='../../../build/MNNConvert', help='local mnnconvert path, if invalid, using pymnn.')
    parser.add_argument('--ppl', action='store_true', help='Whether or not to get all logits of input tokens.')
    parser.add_argument('--awq', action='store_true', help='Whether or not to use awq quant.')
    parser.add_argument('--sym', action='store_true', help='Whether or not to using symmetric quant (without zeropoint), defualt is False.')

    args = parser.parse_args()

    model_path = args.path
    model_type = args.type

    if 'gte' in model_path or 'bge' in model_path:
        llm_exporter = EmbeddingExporter(args)
    else:
        llm_exporter = LlmExporter(args)

    # some actions
    if args.test is not None:
        llm_exporter.response(args.test)

    if args.export is not None:
        llm_exporter.export(args.export)

if __name__ == '__main__':
    main()
