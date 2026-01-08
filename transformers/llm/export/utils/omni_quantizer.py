import torch
import gc
import functools
import json
import inspect
from tqdm import tqdm
from collections import defaultdict
from torch import optim
from .smooth_quantizer import ACIQ, SmoothQuantizer
import torch.nn.functional as F


class OmniQuantizer:
    def __init__(
        self,
        model,
        max_calib_samples=32,
        max_calib_seq_len=128,
        act_bit=8,
        act_sym=True,
        generate_for_npu=False,
        epochs=20,
        lr=5e-3,
        wd=0.0
    ) -> None:
        self.model = model
        self.tokenizer = model.tokenizer
        self.act_bit = act_bit
        self.act_sym = act_sym
        self.generate_for_npu = generate_for_npu

        self.epochs = epochs
        self.lr = lr
        self.wd = wd

        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len

        self.calib_data = 'wikitext' if model.args.calib_data is None else model.args.calib_data
        self.split = 'train'
        self.best_device = self.get_best_device()

        self.modules = self.model.blocks
        self.act_quanter = ACIQ(act_bit)
        self.moment = 0.99

        if "cpu" != self.best_device:
            for idx in range(len(self.modules)):
                self.to_device(self.modules[idx], "cpu")

        self.act_dict = [defaultdict(dict) for _ in range(len(self.modules))]

    @staticmethod
    def get_best_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"

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
    def clear_memory(weight=None):
        if weight is not None:
            del weight
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_calib_dataset(data, tokenizer=None, n_samples=128, max_seq_len=512, split="train"):
        custom_calib_data = False
        if isinstance(data, str):
            from datasets import load_dataset
            if data == "pileval":
                dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            elif data == "wikitext":
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            else:
                custom_calib_data = True
                with open(data, 'r', encoding='utf-8') as f:
                    dataset = f.read().splitlines()
        else:
            raise NotImplementedError("Data loading error")

        samples = []
        if custom_calib_data == False:
            dataset = dataset.shuffle(seed=42)
            count = 0
            idx = 0
            while count < n_samples and idx < len(dataset):
                try:
                    text = dataset[idx]["text"]
                    # skip empty lines
                    if not text.strip():
                        idx += 1
                        continue

                    input_ids = tokenizer(
                        text, return_tensors="pt", max_length=max_seq_len, truncation=True
                    ).input_ids
                    # skip empty tokenized inputs
                    if input_ids.numel() > 0:
                        samples.append(input_ids)
                        count += 1
                except:
                    pass
                idx += 1
        else:
            for i in range(min(n_samples, len(dataset))):
                messages = [{"role": "system", "content": ""}, {"role": "user", "content": dataset[i]}]
                prompt = tokenizer.apply_chat_template(messages)
                input_ids = tokenizer(
                    prompt, return_tensors="pt", max_length=max_seq_len, truncation=True
                ).input_ids

                if input_ids.numel() > 0:
                    samples.append(input_ids)

        print(f"Collected {len(samples)} valid calibration samples.")
        return samples

    def init_quant(self, n_samples=128, max_seq_len=512):
        samples = self.get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split
        )
        return samples

    def _get_first_input(self, sample):
        sample = sample.long()
        layer_kwargs = {}
        seq_len = sample.numel()
        new_tokens = 0

        try:
            inps = self.model.embedding(sample)
        except RuntimeError:
            sample = sample.to(self.best_device)
            inps = self.model.embedding(sample)

        inps = inps.to(self.best_device)
        position_ids = self.model.get_position_ids(seq_len, new_tokens, sample)
        rotary_pos_emb = self.model.rotary(position_ids)
        attention_mask = self.model.get_attention_mask(seq_len, new_tokens)

        layer_kwargs["rotary_pos_emb"] = rotary_pos_emb.to(self.best_device)
        layer_kwargs["attention_mask"] = attention_mask.to(self.best_device)
        layer_kwargs["position_ids"] = position_ids.to(self.best_device)

        del sample
        self.clear_memory()
        return layer_kwargs, inps

    def _sanitize_kwargs(self, inputs_kwargs, module):
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs

    def _safe_forward(self, x, module, module_kwargs):
        try:
            target_dtype = next(module.parameters()).dtype
            target_device = next(module.parameters()).device
        except StopIteration:
            target_dtype = torch.float32
            target_device = x.device

        x = x.to(device=target_device, dtype=target_dtype)

        if "cuda" in str(target_device):
            with torch.cuda.amp.autocast(enabled=True, dtype=target_dtype):
                out = module(x, **module_kwargs)
        else:
            out = module(x, **module_kwargs)

        if isinstance(out, tuple):
            out = out[0]
        return out

    def _run_optimization(self, x_in, fcs, ln, act_max):
        device = self.best_device
        target_dtype = list(fcs[0].parameters())[0].dtype

        # Increase micro_batch_size for better GPU utilization
        micro_batch_size = 64

        # Pre-move weights to GPU and keep there
        weights = torch.cat([fc.weight for fc in fcs], dim=0).to(device)

        act_max = act_max.to(device=device, dtype=target_dtype)

        weight_max_per_channel = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
        weight_max_per_channel = weight_max_per_channel.max(dim=0)[0].clamp(min=1e-5).to(device)

        scales_init = (act_max.pow(0.5) / weight_max_per_channel.pow(0.5)).clamp(min=1e-5)
        scales_init = scales_init.to(device=device, dtype=target_dtype)

        log_scale = torch.nn.Parameter(torch.log(scales_init))

        with torch.no_grad():
            w_init_smooth = weights * scales_init.view(1, -1)
            clip_init = w_init_smooth.abs().max(dim=1, keepdim=True)[0]

        clip_val = torch.nn.Parameter(clip_init)

        optimizer = optim.AdamW([
            {'params': [log_scale], 'lr': self.lr},
            {'params': [clip_val], 'lr': self.lr}
        ], weight_decay=self.wd)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
        )

        # Pre-compute constants for quantization
        act_bit = self.act_bit
        act_sym = self.act_sym
        q_max_w = 2 ** (act_bit - 1) - 1

        # Inline quantize functions to reduce function call overhead
        if act_sym:
            q_max_act = 2 ** (act_bit - 1) - 1
            q_min_act = -q_max_act
        else:
            q_max_act = 2 ** act_bit - 1
            q_min_act = 0

        N = x_in.shape[0]
        num_steps = (N + micro_batch_size - 1) // micro_batch_size

        # Pre-load all data to GPU if it fits, otherwise use pinned memory
        try:
            # Try to fit all data on GPU
            x_in_gpu = x_in.to(device, dtype=target_dtype)
            use_gpu_data = True
        except RuntimeError:
            # Fall back to CPU with pinned memory for faster transfer
            x_in_gpu = x_in.pin_memory() if x_in.device.type == 'cpu' else x_in
            use_gpu_data = False

        # Pre-compute target outputs for all batches (computed once, not every epoch)
        with torch.no_grad():
            y_targets = []
            for i in range(0, N, micro_batch_size):
                if use_gpu_data:
                    x_batch = x_in_gpu[i : i + micro_batch_size]
                else:
                    x_batch = x_in_gpu[i : i + micro_batch_size].to(device, dtype=target_dtype, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True, dtype=target_dtype):
                    y_target = F.linear(x_batch, weights)
                y_targets.append(y_target.float())
                if not use_gpu_data:
                    del x_batch

        for epoch in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            total_loss = 0.0

            for batch_idx, i in enumerate(range(0, N, micro_batch_size)):
                if use_gpu_data:
                    x_micro = x_in_gpu[i : i + micro_batch_size]
                else:
                    x_micro = x_in_gpu[i : i + micro_batch_size].to(device, dtype=target_dtype, non_blocking=True)

                y_micro_target = y_targets[batch_idx]

                scale = torch.exp(log_scale)
                s_view = scale.view(1, -1)

                x_sim = x_micro / s_view
                w_sim = weights * s_view

                if act_sym:
                    act_scale = x_sim.abs().max() / q_max_act
                    act_scale = torch.clamp(act_scale, min=1e-5)
                    x_q = torch.round(x_sim / act_scale)
                    x_q = torch.clamp(x_q, q_min_act, q_max_act)
                    x_q = x_q * act_scale
                    x_q = (x_q - x_sim).detach() + x_sim
                else:
                    t_min, t_max = x_sim.min(), x_sim.max()
                    act_scale = (t_max - t_min) / q_max_act
                    act_scale = torch.clamp(act_scale, min=1e-5)
                    zero = -t_min / act_scale
                    x_q = torch.round(x_sim / act_scale + zero)
                    x_q = torch.clamp(x_q, q_min_act, q_max_act)
                    x_q = (x_q - zero) * act_scale
                    x_q = (x_q - x_sim).detach() + x_sim

                # Inline quantize_weight_with_clip
                clip_v = F.relu(clip_val) + 1e-5
                w_clamped = torch.clamp(w_sim, -clip_v, clip_v)
                w_scale = clip_v / q_max_w
                w_q = torch.round(w_clamped / w_scale) * w_scale
                w_q = (w_q - w_clamped).detach() + w_clamped

                x_q = x_q.to(dtype=target_dtype)
                w_q = w_q.to(dtype=target_dtype)

                with torch.cuda.amp.autocast(enabled=True, dtype=target_dtype):
                    y_pred = F.linear(x_q, w_q)

                loss = F.mse_loss(y_pred.float(), y_micro_target)
                loss = loss / num_steps

                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            scheduler.step()

        # Cleanup y_targets
        del y_targets

        with torch.no_grad():
            final_scale = torch.exp(log_scale).detach().view(-1)

            if 'GemmaRMSNorm' in str(type(ln)):
                ln.weight += 1
                ln.weight.div_(final_scale)
                ln.weight -= 1
            else:
                ln.weight.div_(final_scale)
            if hasattr(ln, "bias") and ln.bias is not None:
                ln.bias.div_(final_scale)

            final_clip = F.relu(clip_val).detach()
            current_idx = 0
            for fc in fcs:
                num_out = fc.weight.shape[0]
                layer_clip = final_clip[current_idx : current_idx + num_out]

                fc.weight.mul_(final_scale.view(1, -1))
                fc.weight.data = torch.clamp(fc.weight.data, -layer_clip, layer_clip)

                current_idx += num_out

        del x_in_gpu, weights, weight_max_per_channel, scales_init, log_scale, clip_val, optimizer, scheduler
        torch.cuda.empty_cache()

    def _get_robust_act_max(self, x):
        try:
            x_flat = x.reshape(-1, x.shape[-1])
            if x_flat.shape[0] > 2048:
                if x_flat.shape[0] > 10000:
                    indices = torch.randperm(x_flat.shape[0])[:10000]
                    x_sample = x_flat[indices]
                else:
                    x_sample = x_flat

                robust_max = torch.quantile(x_sample.abs().float(), 0.999, dim=0)
                return robust_max
            else:
                return x_flat.abs().max(dim=0)[0]
        except:
            return x.reshape(-1, x.shape[-1]).abs().max(dim=0)[0]

    def _extract_static_scales(self):
        print("OmniQuant: Extracting final JSON scales...")
        def compute_scale_sym(max_min):
            bit_scale = 2 ** (self.act_bit - 1) - 1
            max_v = max_min.abs().max().item()
            scale = max_v / bit_scale
            return [scale, 0.0]

        def compute_scale_zero_asym(max_min):
            bit_scale = 2 ** (self.act_bit) - 1
            max_v = max_min[0].item()
            min_v = max_min[1].item()
            if max_v < 0.0: max_v = 0.0
            if min_v > 0.0: min_v = 0.0
            scale = 1.0 if max_v == min_v else (max_v - min_v) / bit_scale
            zero = round(-min_v / scale - 2 ** (self.act_bit - 1))
            if self.act_bit == 16 and self.generate_for_npu:
                zero = round(min_v / scale)
            return [scale, zero]

        func = compute_scale_sym if self.act_sym else compute_scale_zero_asym

        for idx in range(len(self.act_dict)):
            for name, input_output in self.act_dict[idx].items():
                self.act_dict[idx][name]['input'] = func(input_output['input'])
                self.act_dict[idx][name]['output'] = func(input_output['output'])

    def _get_all_static_scales_safe(self, idx, layer, named_linears, x_in, module_kwargs):
        def stat_io_hook(m, x, y, name):
            if isinstance(x, tuple): x = x[0]
            if isinstance(y, tuple): y = y[0]

            inp_max_min = self.act_quanter.get_max_min(x.detach().float().to("cpu"))
            out_max_min = self.act_quanter.get_max_min(y.detach().float().to("cpu"))

            if name not in self.act_dict[idx] or "input" not in self.act_dict[idx][name]:
                self.act_dict[idx][name]["input"] = inp_max_min
            else:
                self.act_dict[idx][name]["input"] = inp_max_min * (1-self.moment) + self.moment * self.act_dict[idx][name]["input"]

            if name not in self.act_dict[idx] or "output" not in self.act_dict[idx][name]:
                self.act_dict[idx][name]["output"] = out_max_min
            else:
                self.act_dict[idx][name]["output"] = out_max_min * (1-self.moment) + self.moment * self.act_dict[idx][name]["output"]

        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(functools.partial(stat_io_hook, name=name)))

        sanitized_kwargs = self._sanitize_kwargs(module_kwargs, layer)

        with torch.no_grad():
            self._safe_forward(x_in, layer, sanitized_kwargs)

        for h in handles:
            h.remove()

    def _prepare_calibration_data(self):
        """Prepare calibration samples and compute initial embeddings.

        Returns:
            layer_inputs: List of (input_tensor, kwargs) tuples for each sample
        """
        # Check if we already have prepared data
        if hasattr(self, '_cached_layer_inputs') and self._cached_layer_inputs is not None:
            print("OmniQuant: Using cached calibration data...")
            return self._cached_layer_inputs

        print("OmniQuant: Initializing...")

        self.samples = self.init_quant(
            n_samples=self.max_calib_samples,
            max_seq_len=self.max_calib_seq_len,
        )

        print(f"OmniQuant: Pre-computing embeddings for {len(self.samples)} samples...")

        layer_inputs = []

        for sample in self.samples:
            # skip empty sample
            if sample.numel() == 0:
                continue
            kw, inp = self._get_first_input(sample)

            cpu_kw = {}
            for k, v in kw.items():
                if isinstance(v, torch.Tensor):
                    cpu_kw[k] = v.to("cpu")
                else:
                    cpu_kw[k] = v

            layer_inputs.append((inp.to("cpu"), cpu_kw))
            self.clear_memory()

        # Cache for potential reuse
        self._cached_layer_inputs = layer_inputs
        return layer_inputs

    def optimize_weights(self, collect_feature_map=False):
        """Phase 1: Optimize weights by adjusting LayerNorm and Linear layer weights.

        This phase applies smooth quantization optimization to reduce quantization error.

        Args:
            collect_feature_map: If True, also collect feature map info during this pass
                                 to avoid a second traversal.
        """
        layer_inputs = self._prepare_calibration_data()

        if collect_feature_map:
            # Re-initialize act_dict for fresh collection
            self.act_dict = [defaultdict(dict) for _ in range(len(self.modules))]

        print(f"OmniQuant: Starting weight optimization (Epochs={self.epochs})...")

        for idx in tqdm(range(len(self.modules)), desc="OmniQuant: Optimize Weights"):
            block = self.modules[idx]
            self.to_device(block, self.best_device)

            attn_inputs_list = []
            mlp_inputs_list = []
            next_layer_outputs = []

            def hook_attn_input(m, i, o):
                if isinstance(i, tuple) and len(i) > 0:
                    inp = i[0]
                else:
                    inp = i
                attn_inputs_list.append(inp.detach().view(-1, inp.shape[-1]))

            def hook_mlp_input(m, i, o):
                if isinstance(i, tuple) and len(i) > 0:
                    inp = i[0]
                else:
                    inp = i
                mlp_inputs_list.append(inp.detach().view(-1, inp.shape[-1]))

            h1 = block.self_attn.q_proj.register_forward_hook(hook_attn_input)
            h2 = block.mlp.gate_proj.register_forward_hook(hook_mlp_input)

            # Pre-compute sanitized kwargs once for this block
            sample_kw_gpu = {}
            for k, v in layer_inputs[0][1].items():
                if isinstance(v, torch.Tensor):
                    sample_kw_gpu[k] = v.to(self.best_device)
                else:
                    sample_kw_gpu[k] = v
            sanitized_kw_template = self._sanitize_kwargs(sample_kw_gpu, block)

            # Single forward pass: collect hooks AND compute outputs
            with torch.no_grad():
                for inp, kw in layer_inputs:
                    inp_gpu = inp.to(self.best_device)

                    # Reuse sanitized keys, only update tensor values
                    kw_gpu = {}
                    for k, v in kw.items():
                        if k in sanitized_kw_template:
                            if isinstance(v, torch.Tensor):
                                kw_gpu[k] = v.to(self.best_device)
                            else:
                                kw_gpu[k] = v

                    out = self._safe_forward(inp_gpu, block, kw_gpu)
                    # Store output for next layer
                    next_layer_outputs.append((out.detach().to("cpu"), kw))

                    del inp_gpu, kw_gpu, out

            h1.remove()
            h2.remove()

            # Process collected attention inputs
            if len(attn_inputs_list) > 0:
                # Concatenate on GPU, then move to CPU once
                total_attn_in = torch.cat(attn_inputs_list, dim=0).to("cpu")
                del attn_inputs_list

                qkv = [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]
                ln_attn = block.input_layernorm
                robust_max_attn = self._get_robust_act_max(total_attn_in)
                self._run_optimization(total_attn_in, qkv, ln_attn, robust_max_attn)
                del total_attn_in, robust_max_attn

            # Process collected MLP inputs
            if len(mlp_inputs_list) > 0:
                # Concatenate on GPU, then move to CPU once
                total_mlp_in = torch.cat(mlp_inputs_list, dim=0).to("cpu")
                del mlp_inputs_list

                fcs_mlp = [block.mlp.gate_proj, block.mlp.up_proj]
                ln_mlp = block.post_attention_layernorm
                robust_max_mlp = self._get_robust_act_max(total_mlp_in)
                self._run_optimization(total_mlp_in, fcs_mlp, ln_mlp, robust_max_mlp)
                del total_mlp_in, robust_max_mlp

            self.clear_memory()

            # Outputs already computed in the single forward pass above
            layer_inputs = next_layer_outputs
            del next_layer_outputs

            if "cpu" != self.best_device:
                self.to_device(block, "cpu")
            self.clear_memory()

        print("OmniQuant: Weight optimization completed.")

        # Save final layer outputs for potential reuse by collect_feature_map_info
        self._final_layer_outputs = layer_inputs

        for idx in range(len(self.modules)):
            self.to_device(self.modules[idx], "cpu")
        self.clear_memory()

        # If collect_feature_map is requested, do it now using optimized weights
        if collect_feature_map:
            self._collect_feature_map_optimized()

    def _collect_lm_head_info(self, calib_inputs):
        """Collect lm_head layer activation info for NPU."""
        if not self.generate_for_npu:
            return

        lm_head_idx = len(self.modules)
        self.act_dict.append(defaultdict(dict))

        if hasattr(self.model, 'lm') and hasattr(self.model.lm, 'lm'):
            lm_head = self.model.lm.lm
        elif hasattr(self.model, 'lm_head'):
            lm_head = self.model.lm_head
        else:
            lm_head = None
            print("Warning: lm_head not found in model, skipping lm_head calibration.")

        if lm_head is not None:
            lm_head.to(self.best_device)
            if hasattr(self.model, 'final_layernorm'):
                self.model.final_layernorm.to(self.best_device)

            lm_head_ops = {'lm_head': lm_head}

            for inp, kw in calib_inputs:
                inp_gpu = inp.to(self.best_device)
                with torch.no_grad():
                    if hasattr(self.model, 'final_layernorm'):
                        hidden_states = self.model.final_layernorm(inp_gpu)
                    else:
                        hidden_states = inp_gpu

                self._get_all_static_scales_safe(lm_head_idx, lm_head, lm_head_ops, hidden_states, {})
                del inp_gpu, hidden_states

            lm_head.to("cpu")
            if hasattr(self.model, 'final_layernorm'):
                self.model.final_layernorm.to("cpu")
            self.clear_memory()

    def _collect_feature_map_optimized(self):
        """Optimized feature map collection that reuses embedding computation.

        This uses cached layer outputs from optimize_weights() to avoid
        re-computing embeddings through all layers.
        """
        print("OmniQuant: Collecting static activation scales (optimized)...")
        gc.collect()
        torch.cuda.empty_cache()

        # Re-initialize act_dict
        self.act_dict = [defaultdict(dict) for _ in range(len(self.modules))]

        # Use cached initial inputs
        calib_inputs = self._prepare_calibration_data()

        for idx in tqdm(range(len(self.modules)), desc="Collecting Feature Map Info"):
            block = self.modules[idx]
            self.to_device(block, self.best_device)

            target_ops = SmoothQuantizer.get_all_leaf_modules(block)
            next_calib_inputs = []

            # Batch process samples for better GPU utilization
            batch_size = min(8, len(calib_inputs))  # Process multiple samples together

            for batch_start in range(0, len(calib_inputs), batch_size):
                batch_end = min(batch_start + batch_size, len(calib_inputs))
                batch_items = calib_inputs[batch_start:batch_end]

                for inp, kw in batch_items:
                    inp_gpu = inp.to(self.best_device)
                    kw_gpu = {k: (v.to(self.best_device) if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}

                    self._get_all_static_scales_safe(idx, block, target_ops, inp_gpu, kw_gpu)

                    sanitized_kw = self._sanitize_kwargs(kw_gpu, block)
                    with torch.no_grad():
                        out = self._safe_forward(inp_gpu, block, sanitized_kw)

                    next_calib_inputs.append((out.cpu(), kw))
                    del inp_gpu, kw_gpu, out

            calib_inputs = next_calib_inputs

            if "cpu" != self.best_device:
                self.to_device(block, "cpu")
            self.clear_memory()

        # Collect lm_head info if needed
        self._collect_lm_head_info(calib_inputs)

        del calib_inputs
        self.clear_memory()

        self._extract_static_scales()

        for idx in range(len(self.modules)):
            self.to_device(self.modules[idx], "cpu")
        self.clear_memory()

        print("OmniQuant: Feature map info collection completed.")

    def quantize(self, collect_feature_map=False):
        """Run the full OmniQuant quantization pipeline.

        Args:
            collect_feature_map: If True, collect feature map info after weight optimization.
                                 If False, only perform weight optimization.
        """
        # Run weight optimization, optionally collecting feature map info in the same pass
        self.optimize_weights(collect_feature_map=collect_feature_map)

    def clear_cache(self):
        """Clear cached calibration data to free memory."""
        if hasattr(self, '_cached_layer_inputs'):
            del self._cached_layer_inputs
            self._cached_layer_inputs = None
        if hasattr(self, '_final_layer_outputs'):
            del self._final_layer_outputs
            self._final_layer_outputs = None
        self.clear_memory()

    def _find_match_in_dict(self, mnn_op_name, layer_act_dict):
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
        import copy

        PASS_THROUGH_OPS = [
            'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten',
            'Transpose', 'Permute', 'ConvertTensor', 'Cast',
            'Slice', 'StridedSlice', 'Split', 'Concat', 'Pack'
        ]

        DATA_SELECT_OPS = ['Gather', 'GatherV2', 'GatherND']

        print("Start propagating quantization parameters...")
        changed = True
        pass_round = 0

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

                if op_type in PASS_THROUGH_OPS:
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

                elif op_type in DATA_SELECT_OPS:
                    data_idx = inputs[0]
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

                elif op_type == 'BinaryOp':
                    out_idx = outputs[0]

                    if out_idx in quant_info_dict:
                        target_info = quant_info_dict[out_idx]
                        for inp_idx in inputs:
                            if inp_idx not in quant_info_dict:
                                quant_info_dict[inp_idx] = copy.deepcopy(target_info)
                                quant_info_dict[inp_idx]['index'] = inp_idx
                                changed = True
                                update_count += 1

                    else:
                        scales = []
                        valid_inputs = []
                        for inp_idx in inputs:
                            if inp_idx in quant_info_dict:
                                scales.append(quant_info_dict[inp_idx]['quantInfo']['scale'])
                                valid_inputs.append(inp_idx)

                        if len(valid_inputs) > 0:
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

            should_process = False
            if not self.generate_for_npu:
                should_process = (op_type == 'Convolution')
            else:
                should_process = (op_type not in npu_ignore_types)

            # Handle lm_head separately using the dedicated index
            if 'lm_head' in op_name:
                if self.generate_for_npu and should_process:
                    lm_head_idx = len(self.modules)  # lm_head is stored at this index
                    if lm_head_idx < len(self.act_dict) and len(self.act_dict[lm_head_idx]) > 0:
                        # lm_head stats are stored with key 'lm_head'
                        if 'lm_head' in self.act_dict[lm_head_idx]:
                            stats = self.act_dict[lm_head_idx]['lm_head']
                            print("Quantize lm head for QNN")

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
                continue

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