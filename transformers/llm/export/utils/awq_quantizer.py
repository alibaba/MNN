import gc
import torch
import logging
import inspect
import functools

from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, List, Union, Dict

logging.basicConfig(level=logging.ERROR)

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
        self.w_bit = model.args.quant_bit
        self.group_size = model.args.quant_block
        self.zeropoint = not model.args.sym
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