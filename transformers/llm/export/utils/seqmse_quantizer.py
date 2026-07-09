import functools
import inspect
import gc
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .smooth_quantizer import SmoothQuantizer
from .torch_utils import _get_quant_block_size


class SeqMSEQuantizer:
    def __init__(
        self,
        model,
        max_calib_samples=8,
        max_calib_seq_len=256,
        num_candidates=20,
        max_tokens_per_linear=512,
    ):
        self.model = model
        self.tokenizer = model.tokenizer
        self.quant_bit = model.args.quant_bit
        self.quant_block = model.args.quant_block
        self.symmetric = model.args.sym
        self.calib_data = 'wikitext' if model.args.calib_data is None else model.args.calib_data
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.num_candidates = num_candidates
        self.max_tokens_per_linear = max_tokens_per_linear
        self.best_device = SmoothQuantizer.get_best_device()
        self.modules = self.model.blocks
        self.samples = SmoothQuantizer.get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=max_calib_samples,
            max_seq_len=max_calib_seq_len,
            split='train',
        )
        self.encodings = {}
        self.float_inputs = defaultdict(list)
        self.quant_inputs = defaultdict(list)
        self.stats = {}

    @staticmethod
    def _sanitize_kwargs(inputs_kwargs, module):
        module_signature = inspect.signature(module.forward).parameters
        return {k: v for k, v in inputs_kwargs.items() if k in module_signature}

    @staticmethod
    def _module_forward(x, module, module_kwargs):
        SmoothQuantizer.clear_block_cache(module)
        output = module(x, **module_kwargs)
        if isinstance(output, tuple):
            output = output[0]
        SmoothQuantizer.clear_block_cache(module)
        return output

    @staticmethod
    def _get_named_linears(module):
        return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}

    @staticmethod
    def _module_dtype(module):
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Linear):
                return submodule.weight.dtype
        for param in module.parameters():
            return param.dtype
        return torch.float32

    def _make_full_name(self, layer_idx, name):
        return f'/layers.{layer_idx}/{name.replace(".", "/")}/Linear'

    def _get_first_input(self, samples):
        if isinstance(samples, (list, tuple)):
            sample = torch.cat(samples, dim=1)
        else:
            sample = samples
        if sample.numel() > self.max_calib_seq_len:
            sample = sample.reshape(1, -1)[:, :self.max_calib_seq_len]
        seq_len = sample.numel()
        new_tokens = 0
        inps = self.model.embedding(sample).to(self.best_device)
        position_ids = self.model.get_position_ids(seq_len, new_tokens, sample)
        layer_kwargs = {
            "rotary_pos_emb": self.model.rotary(position_ids).to(self.best_device),
            "attention_mask": self.model.get_attention_mask(seq_len, new_tokens).to(self.best_device),
        }
        return layer_kwargs, inps

    def _collect_float_inputs(self, layer_idx, layer, named_linears, inps, module_kwargs):

        def cast_input_hook(module, x):
            x = x[0] if isinstance(x, tuple) else x
            return (x.to(module.weight.dtype),)

        def cache_input_hook(module, x, y, name):
            x = x[0] if isinstance(x, tuple) else x
            x = x.detach().cpu()
            hidden = x.shape[-1]
            x = x.reshape(-1, hidden)
            if x.shape[0] > self.max_tokens_per_linear:
                x = x[:self.max_tokens_per_linear]
            self.float_inputs[self._make_full_name(layer_idx, name)].append(x)

        handles = []
        for name, linear in named_linears.items():
            handles.append(linear.register_forward_pre_hook(cast_input_hook))
            handles.append(linear.register_forward_hook(functools.partial(cache_input_hook, name=name)))
        out = self._module_forward(inps, layer, self._sanitize_kwargs(module_kwargs, layer))
        for h in handles:
            h.remove()
        return out

    def _run_quant_layer(self, layer_idx, layer, named_linears, inps, module_kwargs):

        def cast_input_hook(module, x):
            x = x[0] if isinstance(x, tuple) else x
            return (x.to(module.weight.dtype),)

        def quant_hook(module, x, y, name):
            full_name = self._make_full_name(layer_idx, name)
            q_in = x[0] if isinstance(x, tuple) else x
            flat_q_in = q_in.detach().cpu().reshape(-1, q_in.shape[-1])
            if flat_q_in.shape[0] > self.max_tokens_per_linear:
                flat_q_in = flat_q_in[:self.max_tokens_per_linear]
            self.quant_inputs[full_name].append(flat_q_in)

            if full_name not in self.encodings:
                if full_name not in self.float_inputs:
                    return y
                ref_inputs = torch.cat(self.float_inputs[full_name], dim=0)
                quant_inputs = torch.cat(self.quant_inputs[full_name], dim=0)
                self.encodings[full_name] = self._search_linear(module, ref_inputs, quant_inputs)
                baseline_mse = self.encodings[full_name]["baseline_mse"]
                best_mse = self.encodings[full_name]["best_mse"]
                improvement = 0.0 if baseline_mse <= 0 else (baseline_mse - best_mse) / baseline_mse
                self.stats[full_name] = {
                    "baseline_mse": baseline_mse,
                    "best_mse": best_mse,
                    "improvement": improvement,
                }

            dq_weight = self._dequantize_weight_from_encoding(module, self.encodings[full_name]).to(q_in.device)
            bias = None if module.bias is None else module.bias.to(q_in.device)
            return F.linear(q_in, dq_weight, bias)

        handles = []
        for name, linear in named_linears.items():
            handles.append(linear.register_forward_pre_hook(cast_input_hook))
            handles.append(linear.register_forward_hook(functools.partial(quant_hook, name=name)))
        out = self._module_forward(inps, layer, self._sanitize_kwargs(module_kwargs, layer))
        for h in handles:
            h.remove()
        return out

    def _candidate_dequant(self, weight_blocks, ratio):
        offset = 1 << (self.quant_bit - 1)
        clip_max = offset - 1
        if self.symmetric:
            abs_max = weight_blocks.abs().amax(dim=-1, keepdim=True) * ratio
            scale = (abs_max / clip_max).clamp_min(1e-8)
            q = torch.round(weight_blocks / scale).clamp(-clip_max, clip_max)
            dq = q * scale
            zero = None
        else:
            clip_min = -offset
            w_min = weight_blocks.amin(dim=-1, keepdim=True) * ratio
            w_max = weight_blocks.amax(dim=-1, keepdim=True) * ratio
            scale = ((w_max - w_min) / (clip_max - clip_min)).clamp_min(1e-8)
            q = torch.round((weight_blocks - w_min) / scale) + clip_min
            dq = (q.clamp(clip_min, clip_max) - clip_min) * scale + w_min
            zero = w_min
        return dq, scale.squeeze(-1), None if zero is None else zero.squeeze(-1)

    @torch.no_grad()
    def _search_linear(self, linear, ref_inputs, quant_inputs):
        weight = linear.weight.detach().float().cpu()
        ref_inputs = ref_inputs.float()
        quant_inputs = quant_inputs.float()
        sample_count = min(ref_inputs.shape[0], quant_inputs.shape[0], self.max_tokens_per_linear)
        ref_inputs = ref_inputs[:sample_count]
        quant_inputs = quant_inputs[:sample_count]

        oc, ic = weight.shape
        block_size = _get_quant_block_size(ic, self.quant_block)
        block_num = ic // block_size
        weight_blocks = weight.reshape(oc, block_num, block_size)
        ref_blocks = ref_inputs.reshape(ref_inputs.shape[0], block_num, block_size)
        quant_blocks = quant_inputs.reshape(quant_inputs.shape[0], block_num, block_size)

        ratios = torch.linspace(1.0 / self.num_candidates, 1.0, self.num_candidates)
        best_loss = torch.full((oc, block_num), torch.inf)
        best_scale = torch.empty((oc, block_num))
        best_zero = None if self.symmetric else torch.empty((oc, block_num))
        ref_out = torch.einsum('tbi,obi->tob', ref_blocks, weight_blocks)
        baseline_loss = None

        for ratio in ratios:
            dq, scale, zero = self._candidate_dequant(weight_blocks, ratio)
            quant_out = torch.einsum('tbi,obi->tob', quant_blocks, dq)
            loss = (quant_out - ref_out).pow(2).mean(dim=0)
            if torch.isclose(ratio, torch.tensor(1.0)):
                baseline_loss = loss.detach().clone()
            mask = loss < best_loss
            best_loss[mask] = loss[mask]
            best_scale[mask] = scale[mask]
            if best_zero is not None:
                best_zero[mask] = zero[mask]

        return {
            "scale": best_scale.cpu(),
            "zero": None if best_zero is None else best_zero.cpu(),
            "quant_bit": self.quant_bit,
            "quant_block": self.quant_block,
            "symmetric": self.symmetric,
            "baseline_mse": float(best_loss.mean().item() if baseline_loss is None else baseline_loss.mean().item()),
            "best_mse": float(best_loss.mean().item()),
        }

    def _dequantize_weight_from_encoding(self, linear, encoding):
        weight = linear.weight.detach().float()
        oc, ic = weight.shape
        block_size = _get_quant_block_size(ic, self.quant_block)
        block_num = ic // block_size
        weight_blocks = weight.reshape(oc, block_num, block_size)
        scale = encoding["scale"].to(weight.device, weight.dtype).reshape(oc, block_num, 1)
        offset = 1 << (self.quant_bit - 1)
        clip_max = offset - 1
        if self.symmetric:
            q = torch.round(weight_blocks / scale).clamp(-clip_max, clip_max)
            dq = q * scale
        else:
            clip_min = -offset
            zero = encoding["zero"].to(weight.device, weight.dtype).reshape(oc, block_num, 1)
            q = torch.round((weight_blocks - zero) / scale) + clip_min
            dq = (q.clamp(clip_min, clip_max) - clip_min) * scale + zero
        return dq.reshape_as(weight).to(linear.weight.dtype)

    def _set_layer_quantized_weights(self, layer_idx, named_linears):
        originals = {}
        for name, linear in named_linears.items():
            full_name = self._make_full_name(layer_idx, name)
            encoding = self.encodings.get(full_name)
            if encoding is None:
                continue
            originals[linear] = linear.weight.data
            linear.weight.data = self._dequantize_weight_from_encoding(linear, encoding).to(linear.weight.device)
        return originals

    @staticmethod
    def _restore_weights(originals):
        for linear, weight in originals.items():
            linear.weight.data = weight

    def quantize(self):
        self.model.eval()
        layer_kwargs, fp_inps = self._get_first_input(self.samples)
        qt_inps = fp_inps.detach().clone()

        with torch.no_grad():
            for layer_idx, layer in enumerate(tqdm(self.modules, desc="SeqMSE")):
                layer.to(self.best_device)
                layer.float()
                layer_dtype = self._module_dtype(layer)
                fp_inps = fp_inps.to(device=self.best_device, dtype=layer_dtype)
                qt_inps = qt_inps.to(device=self.best_device, dtype=layer_dtype)
                named_linears = self._get_named_linears(layer)

                fp_next = self._collect_float_inputs(layer_idx, layer, named_linears, fp_inps, layer_kwargs)
                originals = self._set_layer_quantized_weights(layer_idx, named_linears)
                qt_next = self._run_quant_layer(layer_idx, layer, named_linears, qt_inps, layer_kwargs)
                self._restore_weights(originals)
                fp_inps = fp_next.detach().cpu()
                qt_inps = qt_next.detach().cpu()
                layer.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.model.seqmse_encodings = self.encodings
        self.model.seqmse_stats = self.stats
        if self.stats:
            baseline = sum(item["baseline_mse"] for item in self.stats.values()) / len(self.stats)
            best = sum(item["best_mse"] for item in self.stats.values()) / len(self.stats)
            improvement = 0.0 if baseline <= 0 else (baseline - best) / baseline
            print(f"SeqMSE reconstruction MSE: baseline={baseline:.6e}, best={best:.6e}, improvement={improvement * 100:.2f}%")
        return self.encodings
