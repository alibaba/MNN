import torch
import logging
import gc
import functools
import json
import inspect
from typing import Dict
from tqdm import tqdm
from collections import defaultdict
import math

logging.basicConfig(level=logging.ERROR)

class ACIQ:
    def __init__(self, size):
        self.num_bits = size
        # TODO: expose as cmd line parameters
        self.stochastic = False
        self.int_exp = False
        self.enforce_true_zero = True #params['true_zero']
        self.alpha_gaus = {2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
        self.alpha_laplace = {2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
        self.gaussian_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5)
    def alpha2DeltaOffset(self, alpha, max_value, min_value, mean):
        max_range = max_value - min_value
        if alpha <= 0 or alpha >= max_range / 2:
            delta = max_range
        else:
            delta = 2 * alpha
            min_value = max(min_value, mean - delta / 2)

        return delta, min_value

    def gemmlowpClippingQuantize(self, input):
        min_value = input.min()
        max_value = input.max()
        mean = input.mean()


        alpha = self.get_alpha_gaus(input)  # gaussian clipping

        delta, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean)

        return torch.stack([delta + min_value, min_value], 0)
    def get_max_min(self, x):
        if self.num_bits > 8:
            return torch.stack([x.max(), x.min()], 0)
        return self.gemmlowpClippingQuantize(x)
    def get_alpha_gaus(self, tensor):
        N = 1
        for i in range(len(tensor.shape)):
            N *= tensor.shape[i]
        min_value = tensor.min()
        max_value = tensor.max()

        std = ((max_value - min_value) * self.gaussian_const) / ((2 * math.log(N)) ** 0.5)
        return self.alpha_gaus[self.num_bits] * std


class SmoothQuantizer:
    def __init__(
        self,
        model,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        alpha=0.5,
        act_bit=8,
        act_sym=True,
        generate_for_npu=False
    ) -> None:
        self.act_sym = act_sym
        self.model = model
        self.tokenizer = model.tokenizer
        #self.w_bit = model.args.quant_bit
        self.act_bit = act_bit
        self.group_size = model.args.quant_block
        self.alpha = alpha
        self.generate_for_npu = generate_for_npu

        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.split = 'train'
        self.calib_data = 'wikitext' if model.args.calib_data is None else model.args.calib_data
        self.best_device = SmoothQuantizer.get_best_device()

        self.modules = self.model.blocks
        self.act_quanter = ACIQ(act_bit)
        self.moment = 0.99
        if "cpu" != self.best_device:
            for idx in range(len(self.modules)):
                SmoothQuantizer.to_device(self.modules[idx], "cpu")

        self.act_scales = [{} for _ in range(len(self.modules))]
        self.act_dict = [defaultdict(dict) for _ in range(len(self.modules))]

        self.n_parallel_calib_samples = n_parallel_calib_samples

        self.samples = self.init_quant(
            n_samples=self.max_calib_samples,
            max_seq_len=self.max_calib_seq_len,
        )

    @staticmethod
    def get_calib_dataset(
        data,
        tokenizer=None,
        n_samples=128,
        max_seq_len=512,
        split="train",
    ):
        custom_calib_data = False
        if isinstance(data, str):
            from datasets import load_dataset
            if data == "pileval":
                dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            elif data == "wikitext":
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            else:
                custom_calib_data = True
                # dataset = load_dataset(data, split=split)
                with open(data, 'r', encoding='utf-8') as f:
                    dataset = f.read().splitlines()
            # dataset = dataset.shuffle(seed=42)
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset"
                "that is preprocessed with one sample of text per element"
            )

        samples = []
        if custom_calib_data == False:
            dataset = dataset.shuffle(seed=42)
            for i in range(n_samples):
                input_ids = tokenizer(
                    dataset[i]["text"], return_tensors="pt", max_length=max_seq_len, truncation=True
                ).input_ids
                samples.append(input_ids)
        else:
            for i in range(n_samples):
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": dataset[i]}
                ]
                prompt = tokenizer.apply_chat_template(messages)
                input_ids = tokenizer(
                    prompt, return_tensors="pt", max_length=max_seq_len, truncation=True
                ).input_ids
                samples.append(input_ids)

        return samples

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


    def init_quant(self, n_samples=128, max_seq_len=512):
        samples = SmoothQuantizer.get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split
        )
        return samples

    def _get_first_input(self, sample):
        layer_kwargs = {}
        seq_len = sample.numel()
        new_tokens = 0
        inps = self.model.embedding(sample).to(self.best_device)
        position_ids = self.model.get_position_ids(seq_len, new_tokens, sample)
        rotary_pos_emb = self.model.rotary(position_ids)
        attention_mask = self.model.get_attention_mask(seq_len, new_tokens, )
        layer_kwargs["rotary_pos_emb"] = rotary_pos_emb.to(self.best_device)
        layer_kwargs["attention_mask"] = attention_mask.to(self.best_device)
        del sample
        SmoothQuantizer.clear_memory()
        return layer_kwargs, inps

    def _get_max_input(self, idx, layer, named_linears):

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            if name in self.act_scales[idx]:
                self.act_scales[idx][name] = torch.max(self.act_scales[idx][name], comming_max)
            else:
                self.act_scales[idx][name] = comming_max

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            stat_tensor(name, x)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()

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

    @staticmethod
    def to_device(module, device):
        for child_name, child_module in module.named_children():
            if child_name == 'self_attn':
                for sub_name, sub_child in child_module.named_children():
                    if sub_name != 'config':
                        sub_child.to(device)
            else:
                child_module.to(device)

    @staticmethod
    def get_named_linears(module):
        linears = {}
        for name, m in module.named_modules():
            # 兼容更多潜在的线性层类型
            if isinstance(m, torch.nn.Linear) or m.__class__.__name__ == 'Linear':
                linears[name] = m
        return linears

    @staticmethod
    def get_all_leaf_modules(module):
        targets = {}
        for name, submod in module.named_modules():
            if name == "":
                continue

            if len(list(submod.children())) == 0:
                if isinstance(submod, (torch.nn.Dropout, torch.nn.Identity)):
                    continue
                targets[name] = submod

        return targets

    @staticmethod
    @torch.no_grad()
    def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
        if not isinstance(fcs, list):
            fcs = [fcs]
        if not SmoothQuantizer.is_allowed_norms(ln):
            raise NotImplementedError(
                f"LayerNorm {ln} is not supported for smooth quantization."
            )
        for fc in fcs:
            assert isinstance(fc, torch.nn.Linear)
            assert ln.weight.numel() == fc.in_features == act_scales.numel()
        device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
        act_scales = act_scales.to(device=device, dtype=dtype)
        weight_scales = torch.cat(
            [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
        )
        weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
        scales = (
            (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(dtype)
        )

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

    @staticmethod
    def is_allowed_norms(op):
        if isinstance(op, torch.nn.LayerNorm):
            return True
        if any(t in str(type(op)) for t in ['LlamaRMSNorm', 'GemmaRMSNorm', 'CohereLayerNorm']):
            return True
        if "rmsnorm" in str(op.__class__).lower():
            return True
        return False

    def _apply_scale(self, idx, module):
        attn_ln = module.input_layernorm
        qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

        qkv_input_scales = self.act_scales[idx]["self_attn.q_proj"]
        SmoothQuantizer.smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, self.alpha)

        ffn_ln = module.post_attention_layernorm  # feed forward norm
        fcs = [module.mlp.gate_proj, module.mlp.up_proj]
        ffn_input_scales = self.act_scales[idx]["mlp.gate_proj"]
        SmoothQuantizer.smooth_ln_fcs(ffn_ln, fcs, ffn_input_scales, self.alpha)

    @torch.no_grad()
    def _get_all_static_scales(self, idx, layer, named_linears):
        def stat_io_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            max_min = self.act_quanter.get_max_min(x)
            if name not in self.act_dict[idx] or "input" not in self.act_dict[idx][name]:
                self.act_dict[idx][name]["input"] = max_min
            else:
                self.act_dict[idx][name]["input"] = max_min * (1-self.moment) + self.moment * self.act_dict[idx][name]["input"]
            if isinstance(y, tuple):
                y = y[0]
            max_min = self.act_quanter.get_max_min(y)
            if name not in self.act_dict[idx] or "output" not in self.act_dict[idx][name]:
                self.act_dict[idx][name]["output"] = max_min
            else:
                self.act_dict[idx][name]["output"] = max_min * (1-self.moment) + self.moment * self.act_dict[idx][name]["output"]
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(stat_io_hook, name=name)
                )
            )
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()

    @torch.no_grad()
    def _extract_static_scales(self):

        print("Extracting static scales...")

        def compute_scale_sym(max_min):
            bit_scale = 2 ** (self.act_bit - 1) - 1
            max_v = max_min.abs().max().item()
            scale = max_v / bit_scale
            zero = 0.0
            return [scale, zero]

        def compute_scale_zero_asym(max_min):
            bit_scale = 2 ** (self.act_bit) - 1
            max_v = max_min[0].item()
            min_v = max_min[1].item()
            # Assume has zeropoint
            if max_v < 0.0:
                max_v = 0.0
            if min_v > 0.0:
                min_v = 0.0
            scale = 1.0
            if max_v == min_v:
                scale = 1.0
            else:
                scale = (max_v - min_v) / bit_scale
            zero = round(-min_v / scale - 2 ** (self.act_bit - 1))
            if self.act_bit == 16 and self.generate_for_npu:
                zero = round(min_v / scale)
            elif self.act_bit == 16:
                print("Error: CPU only supports 8 bit feature map quantized")
            return [scale, zero]
        if self.act_sym:
            func = compute_scale_sym
        else:
            func = compute_scale_zero_asym
        for idx in range(len(self.modules)):
            for name, input_output in self.act_dict[idx].items():
                self.act_dict[idx][name]['input'] = func(input_output['input'])
                self.act_dict[idx][name]['output'] = func(input_output['output'])

    def quantize(self):

        for i in tqdm(range(len(self.samples)), desc="collecting data and computing scales..."):
            sample = self.samples[i]
            if sample.numel() == 0:
                continue
            self.module_kwargs, self.inps = self._get_first_input(sample)

            for idx in range(len(self.modules)):
                SmoothQuantizer.to_device(self.modules[idx], self.best_device)

                if self.module_kwargs.get("position_ids", None) is not None:
                    self.module_kwargs["position_ids"] = self.module_kwargs["position_ids"].to(self.best_device)

                if self.module_kwargs.get("attention_mask", None) is not None:
                    self.module_kwargs["attention_mask"] = self.module_kwargs["attention_mask"].to(self.best_device)

                named_layers = SmoothQuantizer.get_all_leaf_modules(self.modules[idx])

                self._get_max_input(idx, self.modules[idx], named_layers)
                if "cpu" != self.best_device:
                    SmoothQuantizer.to_device(self.modules[idx], "cpu")

        for idx in tqdm(range(len(self.modules)), desc="applying scales..."):
            self._apply_scale(idx, self.modules[idx])

        for i in tqdm(range(len(self.samples)), desc="collecting static activation scales..."):
            sample = self.samples[i]
            if sample.numel() == 0:
                continue
            self.module_kwargs, self.inps = self._get_first_input(sample)

            for idx in range(len(self.modules)):
                SmoothQuantizer.to_device(self.modules[idx], self.best_device)

                if self.module_kwargs.get("position_ids", None) is not None:
                    self.module_kwargs["position_ids"] = self.module_kwargs["position_ids"].to(self.best_device)

                if self.module_kwargs.get("attention_mask", None) is not None:
                    self.module_kwargs["attention_mask"] = self.module_kwargs["attention_mask"].to(self.best_device)

                named_linears = SmoothQuantizer.get_all_leaf_modules(self.modules[idx])

                self._get_all_static_scales(idx, self.modules[idx], named_linears)
                if "cpu" != self.best_device:
                    SmoothQuantizer.to_device(self.modules[idx], "cpu")
        self._extract_static_scales()

        SmoothQuantizer.clear_memory()
        for idx in range(len(self.modules)):
            SmoothQuantizer.to_device(self.modules[idx], "cpu")

    def _find_match_in_dict(self, mnn_op_name, layer_act_dict):
        """
        mnn_op_name: e.g., '/blocks.0/self_attn/q_norm/Mul_1_output_0'
        layer_act_dict: { 'self_attn.q_norm': {...}, 'self_attn.q_proj': {...} }
        """

        best_match = None
        max_len = 0

        for pt_name in layer_act_dict.keys():
            pt_path = pt_name.replace('.', '/')
            if pt_path in mnn_op_name:
                if len(pt_path) > max_len:
                    max_len = len(pt_path)
                    best_match = pt_name

        return best_match

    def _propagate_quant_info(self, mnn_ops, quant_info_dict):
        """
        量化参数传导机制。
        通过图的拓扑结构，将已知的量化参数传递给相邻的未知 Tensor。
        """
        import copy

        # 定义透传算子：输入和输出共享 Scale/Zero
        # 这些算子不改变数值分布，只改变形状或排布
        PASS_THROUGH_OPS = [
            'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten',
            'Transpose', 'Permute', 'ConvertTensor', 'Cast',
            'Slice', 'StridedSlice', 'Split', 'Concat', 'Pack'
        ]

        # 定义特殊的单向或部分传导算子
        # Gather: Output Scale == Input[0] (Data) Scale. (Input[1] 是 indices，不需要)
        DATA_SELECT_OPS = ['Gather', 'GatherV2', 'GatherND']

        print("Start propagating quantization parameters...")
        changed = True
        pass_round = 0

        # 不动点迭代：只要这轮循环有更新，就继续跑下一轮
        while changed:
            changed = False
            pass_round += 1
            update_count = 0

            for op in mnn_ops:
                op_type = op.get('type', '')
                inputs = op.get('inputIndexes', [])
                outputs = op.get('outputIndexes', [])

                if not inputs or not outputs:
                    continue

                # -----------------------------------------------
                # 策略 1: 透传算子 (双向传导)
                # Input <-> Output
                # -----------------------------------------------
                if op_type in PASS_THROUGH_OPS:
                    # 1. Forward: 任意 Input 有参数 -> 传给所有 Output
                    # 通常取第一个有参数的 input 作为源
                    source_info = None
                    for inp_idx in inputs:
                        if inp_idx in quant_info_dict:
                            source_info = quant_info_dict[inp_idx]
                            break

                    if source_info:
                        for out_idx in outputs:
                            if out_idx not in quant_info_dict:
                                quant_info_dict[out_idx] = copy.deepcopy(source_info)
                                quant_info_dict[out_idx]['index'] = out_idx # 修正 index
                                changed = True
                                update_count += 1

                    # 2. Backward: 任意 Output 有参数 -> 传给所有 Input
                    # (仅当 Input 还没有参数时)
                    target_info = None
                    for out_idx in outputs:
                        if out_idx in quant_info_dict:
                            target_info = quant_info_dict[out_idx]
                            break

                    if target_info:
                        for inp_idx in inputs:
                            if inp_idx not in quant_info_dict:
                                quant_info_dict[inp_idx] = copy.deepcopy(target_info)
                                quant_info_dict[inp_idx]['index'] = inp_idx
                                changed = True
                                update_count += 1

                # -----------------------------------------------
                # 策略 2: Gather 类 (仅数据输入 <-> 输出)
                # -----------------------------------------------
                elif op_type in DATA_SELECT_OPS:
                    data_idx = inputs[0] # 第0个是 params/data
                    out_idx = outputs[0]

                    # Forward: Data -> Output
                    if data_idx in quant_info_dict and out_idx not in quant_info_dict:
                        quant_info_dict[out_idx] = copy.deepcopy(quant_info_dict[data_idx])
                        quant_info_dict[out_idx]['index'] = out_idx
                        changed = True
                        update_count += 1

                    # Backward: Output -> Data
                    if out_idx in quant_info_dict and data_idx not in quant_info_dict:
                        quant_info_dict[data_idx] = copy.deepcopy(quant_info_dict[out_idx])
                        quant_info_dict[data_idx]['index'] = data_idx
                        changed = True
                        update_count += 1

                # -----------------------------------------------
                # 策略 3: BinaryOp (Add/Mul) - 谨慎处理
                # 通常用于 Residual Add。
                # -----------------------------------------------
                elif op_type == 'BinaryOp':
                    out_idx = outputs[0]

                    # Backward:
                    # 如果 Add 的输出已知（通常是因为连着下一个 Linear/Norm 的输入），
                    # 我们可以尝试推导输入的 Scale。
                    # 注意：对于 Add，如果 Input A 和 Input B 的范围差异巨大，直接回传可能有风险。
                    # 但在 Transformer 残差结构中，通常 Input 和 Output 的 Scale 是同数量级的。
                    if out_idx in quant_info_dict:
                        target_info = quant_info_dict[out_idx]
                        for inp_idx in inputs:
                            if inp_idx not in quant_info_dict:
                                quant_info_dict[inp_idx] = copy.deepcopy(target_info)
                                quant_info_dict[inp_idx]['index'] = inp_idx
                                changed = True
                                update_count += 1

                    # Forward:
                    # 如果所有输入都有 Scale，取 Scale 最大的那个传给输出
                    # (保守策略，避免截断)
                    else:
                        scales = []
                        valid_inputs = []
                        for inp_idx in inputs:
                            if inp_idx in quant_info_dict:
                                scales.append(quant_info_dict[inp_idx]['quantInfo']['scale'])
                                valid_inputs.append(inp_idx)

                        if len(valid_inputs) > 0:
                            # 找到 Scale 最大的那个 input 的 info
                            max_scale_idx = valid_inputs[scales.index(max(scales))]
                            source = quant_info_dict[max_scale_idx]

                            quant_info_dict[out_idx] = copy.deepcopy(source)
                            quant_info_dict[out_idx]['index'] = out_idx
                            changed = True
                            update_count += 1

            print(f"  Pass {pass_round}: Updated {update_count} tensors.")

        return quant_info_dict

    def apply(self, base_path):
        mnn = json.load(open(base_path, 'rt'))
        mnn['extraTensorDescribe'] = []

        max_val = 2 ** (self.act_bit - 1) - 1
        min_val = -max_val
        data_type = 'DT_INT16'
        if self.act_bit <= 8:
            data_type = 'DT_INT8'
        elif self.act_bit > 8 and self.act_bit <= 16:
            data_type = 'DT_INT16'

        quant_info_dict = {}
        npu_ignore_types = {'Input', 'Const', 'Extra', 'Reshape', 'ConvertTensor'}

        for op in mnn['oplists']:
            op_name = op.get('name', '')
            op_type = op.get('type', '')

            if 'lm_head' in op_name:
                continue

            should_process = False
            if not self.generate_for_npu:
                should_process = (op_type == 'Convolution')
            else:
                should_process = (op_type not in npu_ignore_types)

            if should_process:
                try:
                    import re
                    match = re.search(r'(?:blocks|layers)\.(\d+)', op_name)
                    if match:
                        layer_idx = int(match.group(1))
                    else:
                        continue
                except:
                    continue

                if layer_idx >= len(self.act_dict):
                    continue

                layer_act_dict = self.act_dict[layer_idx]
                matched_pt_name = self._find_match_in_dict(op_name, layer_act_dict)

                if matched_pt_name:
                    stats = layer_act_dict[matched_pt_name]

                    if 'input' in stats and len(op['inputIndexes']) > 0:
                        tensor_idx = op['inputIndexes'][0]

                        if tensor_idx not in quant_info_dict:
                            scale, zero = stats['input']
                            quant_info_dict[tensor_idx] = {
                                'index': tensor_idx,
                                'quantInfo': {
                                    'scale': scale,
                                    'zero': zero,
                                    'min': min_val,
                                    'max': max_val,
                                    "type": data_type
                                }
                            }

                    if 'output' in stats and len(op['outputIndexes']) > 0:
                        tensor_idx = op['outputIndexes'][0]

                        if tensor_idx not in quant_info_dict:
                            scale, zero = stats['output']
                            quant_info_dict[tensor_idx] = {
                                'index': tensor_idx,
                                'quantInfo': {
                                    'scale': scale,
                                    'zero': zero,
                                    'min': min_val,
                                    'max': max_val,
                                    "type": data_type
                                }
                            }

        if self.generate_for_npu:
            print(f"Initial collected tensors: {len(quant_info_dict)}")
            self._propagate_quant_info(mnn['oplists'], quant_info_dict)
            print(f"final collected tensors: {len(quant_info_dict)}")
        mnn['extraTensorDescribe'] = list(quant_info_dict.values())

        with open(base_path, 'w', encoding='utf-8') as f:
            json.dump(mnn, f, ensure_ascii=False, indent=4)

        return base_path










