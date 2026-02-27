import torch
import numpy as np


def get_available_memory(device):
    """Get available GPU memory in bytes."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return free
    else:
        # return a default number: 16 GB
        return 16 * 1024 * 1024 * 1024


def estimate_hqq_memory(num_elements, compute_dtype=torch.float32, symmetric=False):
    """
    Estimate the memory needed for HQQ quantization optimization.

    The optimization process creates multiple temporary tensors:
    - W_f: copy of weights in compute_dtype
    - W_q: quantized weights
    - W_r: reconstructed weights
    - W_e: error tensor
    - W_prime: additional tensor for symmetric quantization
    - Additional temporary tensors in _shrink_lp_op

    Returns estimated memory in bytes.
    """
    dtype_size = 4 if compute_dtype == torch.float32 else 2

    # Main working tensors
    num_tensors = 4 if not symmetric else 5
    main_memory = num_elements * dtype_size * num_tensors

    # Temporary tensors during _shrink_lp_op (can create 2-3 additional tensors)
    temp_memory = num_elements * dtype_size * 3

    # Scale and zero tensors (smaller, proportional to num_groups)
    # Assuming group_size = 64, num_groups = num_elements / 64
    scale_zero_memory = (num_elements // 64) * dtype_size * 2

    # Add 20% safety margin for PyTorch memory allocator overhead
    total_memory = (main_memory + temp_memory + scale_zero_memory) * 1.2

    return int(total_memory)


class HQQQuantizer:
    def __init__(self,
                 weight,
                 bit,
                 group_size,
                 sym=False,
                 compute_dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu"),
                 quant_config: dict = None,
                 auto_chunk: bool = True,
                 memory_safety_factor: float = 0.7):
        self.weight = weight
        self.bit = bit
        self.group_size = group_size
        self.sym = sym
        self.compute_dtype = compute_dtype
        self.device = device
        self.auto_chunk = auto_chunk
        self.memory_safety_factor = memory_safety_factor

    def quant(self):
        if self.auto_chunk and self.device.type in ('cuda', 'mps'):
            self._quantize_chunked()
        else:
            self._quantize()

    def _quantize_chunked(self):
        """Quantize with automatic chunking to avoid OOM."""
        num_elements = self.weight.numel()
        estimated_memory = estimate_hqq_memory(num_elements, self.compute_dtype, self.sym)
        available_memory = get_available_memory(self.device)
        safe_memory = available_memory * self.memory_safety_factor

        if estimated_memory <= safe_memory:
            # No chunking needed
            self._quantize()
            return

        # Calculate number of chunks needed
        num_chunks = int(np.ceil(estimated_memory / safe_memory))
        original_shape = self.weight.shape

        # Split along the first dimension (output channels)
        chunk_size = max(1, original_shape[0] // num_chunks)
        num_chunks = (original_shape[0] + chunk_size - 1) // chunk_size

        # Collect results
        W_q_list = []
        scale_list = []
        zero_list = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, original_shape[0])
            chunk_weight = self.weight[start_idx:end_idx].contiguous()

            # Quantize this chunk
            chunk_quantizer = HQQQuantizer(
                chunk_weight,
                self.bit,
                self.group_size,
                self.sym,
                self.compute_dtype,
                self.device,
                auto_chunk=False  # Disable recursive chunking
            )
            chunk_quantizer._quantize()

            W_q_list.append(chunk_quantizer.W_q)
            scale_list.append(chunk_quantizer.meta['scale'])
            if not self.sym:
                zero_list.append(chunk_quantizer.meta['zero'])

            # Clean up chunk memory
            del chunk_weight, chunk_quantizer
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()

        # Concatenate results
        self.W_q = torch.cat(W_q_list, dim=0)
        self.meta = {
            "nbits": self.bit,
            "group_size": self.group_size,
            "shape": original_shape,
            "scale": torch.cat(scale_list, dim=0),
            "zero": torch.cat(zero_list, dim=0) if not self.sym else None,
            "axis": 1,
        }

        # Clean up
        del W_q_list, scale_list, zero_list
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()

    @torch.inference_mode()
    def _quantize(
        self,
        channel_wise: bool = True,
        axis: int = 1,
    ) -> tuple:

        if self.group_size is not None:
            assert self.weight.numel() % self.group_size == 0, (
                "group_size should be divisble by the total tensor dimensions. shape: "
                + str(self.weight.shape)
                + ", group_size: "
                + str(self.group_size)
            )

        W = self.weight.to(self.compute_dtype).float()
        shape = W.shape

        # Reshape for grouping
        if (self.group_size is not None) and channel_wise:
            W = (
                W.reshape([-1, self.group_size])
                if (axis == 1)
                else W.reshape([self.group_size, -1])
            )

        # Get min/max values
        if not channel_wise:
            _min, _max = W.min(), W.max()
        else:
            _min = W.min(axis=axis, keepdim=True)[0]
            _max = W.max(axis=axis, keepdim=True)[0]

        if self.sym:
            max_v = 2**(self.bit-1) - 1    # 4bit: 7
            min_v = -2**(self.bit-1)       # 4bit: -8
            min_max = [min_v, max_v]       # [-8, 7]

            max_abs = torch.max(torch.abs(_min), torch.abs(_max))
            scale = max_v / max_abs
            scale = torch.where(max_abs <= 1e-4, torch.full_like(scale, 1.0), scale)
            scale = scale.clamp(max=2e4)
            zero = None
        else:
            max_v = round(2**self.bit - 1)  # 4bit: 15
            min_v = 0                       # 4bit: 0
            min_max = [min_v, max_v]        # [0, 15]

            denom = (_max - _min)
            scale = (max_v / denom)
            scale = torch.where(denom.abs() <= 1e-4, torch.full_like(scale, 1.0), scale)
            scale = scale.clamp(max=2e4)
            zero = -_min * scale
            zero = torch.round(zero)

        W_q, scale, zero = self._optimize_weights(
            W,
            scale,
            zero,
            min_max=min_max,
            axis=axis,
        )
        #W_q = (W * scale).round_().clamp_(min_max[0], min_max[1])
        # cleanup
        del W, _min, _max

        # Store meta-data (we invert the scale for dequantization)
        scale = 1.0 / scale
        meta = {
            "nbits": self.bit,
            "group_size": self.group_size,
            "shape": shape,
            "scale": scale,
            "zero": zero,
            "axis": axis,
        }

        W_q = W_q.to(self.weight.dtype)

        if self.device == torch.device('cuda'):
            torch.cuda.empty_cache()
        elif self.device == torch.device('mps'):
            torch.mps.empty_cache()

        self.W_q = W_q
        self.meta = meta

    @torch.inference_mode()
    def _optimize_weights(
        self,
        W: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        min_max: list,
        axis: int = 0,
        opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
        verbose: bool = False,
    ) -> tuple:
        lp_norm, beta, kappa, iters = (
            opt_params["lp_norm"],
            opt_params["beta"],
            opt_params["kappa"],
            opt_params["iters"],
        )

        dtype = torch.float32
        W_f   = W.to(dtype=dtype, device=self.device)
        scale = scale.to(dtype=dtype, device=self.device)
        if not self.sym:
            zero  = zero.to(dtype=dtype, device=self.device)

        best_error = torch.tensor(torch.inf, dtype=torch.float32, device=self.device)
        W_q = torch.empty_like(W_f)
        W_r = torch.empty_like(W_f)
        W_e = torch.empty_like(W_f)
        W_prime = torch.empty_like(W_f) if self.sym else None
        for i in range(iters):
            if not self.sym:
                self._optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta, lp_norm, axis, W_q, W_r, W_e)
            else:
                self._optimize_weights_proximal_scale_only(W_f, scale, min_max, beta, lp_norm, axis, W_q, W_r, W_e, W_prime)
            current_error = torch.abs(W_f - W_r).mean().float()
            if verbose:
                print(i, current_error.cpu())

            if current_error < best_error:
                best_error = current_error
            else:
                break

        scale = scale.to(W.device)
        if not self.sym:
            zero = zero.to(W.device)
        del W_f, W_q, W_r
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
        if not self.sym:
            W_q = torch.round(W * scale + zero).clamp_(min_max[0], min_max[1])
        else:
            W_q = torch.round(W * scale).clamp_(min_max[0], min_max[1])
        return W_q, scale, zero

    @torch.inference_mode()
    def _optimize_weights_proximal_legacy_step(self, W_f, scale, zero, min_max, beta, lp_norm, axis, W_q, W_r, W_e):
        torch.mul(W_f, scale, out=W_q)
        torch.add(W_q, zero, out=W_q)
        torch.round(W_q, out=W_q).clamp_(min_max[0], min_max[1])
        torch.sub(W_q, zero, out=W_r)
        torch.div(W_r, scale, out=W_r)
        torch.sub(W_f, W_r, out=W_e)
        self._shrink_lp_op(W_e, beta, lp_norm, out=W_e)
        torch.sub(W_f, W_e, out=W_r)
        torch.mul(W_r, scale, out=W_r)
        torch.sub(W_q, W_r, out=W_r)
        torch.mean(W_r, axis=axis, keepdim=True, out=zero)

    @torch.inference_mode()
    def _optimize_weights_proximal_scale_only(self, W_f, scale, min_max, beta, lp_norm, axis, W_q, W_r, W_e, W_prime, eps=1e-8):
        torch.mul(W_f, scale, out=W_q)
        torch.round(W_q, out=W_q).clamp_(min_max[0], min_max[1])
        torch.div(W_q, scale, out=W_r)
        torch.sub(W_f, W_r, out=W_e)
        self._shrink_lp_op(W_e, beta, lp_norm, out=W_e)
        torch.sub(W_f, W_e, out=W_prime)
        w_prime_dot_w_q = torch.sum(W_prime * W_q, axis=axis, keepdim=True)
        w_q_norm_sq = torch.sum(W_q**2, axis=axis, keepdim=True)
        torch.add(w_prime_dot_w_q, eps, out=w_prime_dot_w_q)
        torch.div(w_q_norm_sq, w_prime_dot_w_q, out=scale)

    # Shrinking operator
    @torch.inference_mode()
    def _shrink_lp_op(self, x: torch.Tensor, beta: float, lp_norm: float, out: torch.Tensor) -> torch.Tensor:
        if lp_norm == 1:
            #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
            torch.abs(x, out=out)
            out.sub_(1.0 / beta).clamp_min_(0.0)
            out.mul_(torch.sign(x))
            return out
        else:
            # Original formula: sign(x) * relu(|x| - (1/beta) * |x|^(lp_norm-1))
            # Note: This formula inherently requires a temporary tensor for lp_norm != 1
            # because we need both |x| and |x|^(lp_norm-1) simultaneously
            torch.abs(x, out=out)
            temp = out.pow(lp_norm - 1)
            temp.mul_(1.0 / beta)
            out.sub_(temp).clamp_min_(0.0)
            out.mul_(torch.sign(x))
            del temp
            return out