import torch
import numpy as np


class HQQQuantizer:
    def __init__(self, 
                 weight, 
                 bit,
                 group_size,  
                 sym=False,
                 compute_dtype: torch.dtype = torch.float32, 
                 device: torch.device = torch.device("cpu"),
                 quant_config: dict = None):
        self.weight = weight
        self.bit = bit
        self.group_size = group_size
        self.sym = sym
        self.compute_dtype = compute_dtype
        self.device = device
    
    def quant(self):
        self._quantize()

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
        for i in range(iters):
            if not self.sym:
                W_r, W_q, zero, scale = self._optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta, lp_norm, axis)
            else:
                W_r, W_q, scale = self._optimize_weights_proximal_scale_only(W_f, scale, min_max, beta, lp_norm, axis)
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
    
    def _optimize_weights_proximal_legacy_step(self, W_f, scale, zero, min_max, beta, lp_norm, axis):
        W_q = torch.round(W_f * scale + zero).clamp_(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = self._shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        return W_r, W_q, zero, scale
    
    def _optimize_weights_proximal_scale_only(self, W_f, scale, min_max, beta, lp_norm, axis, eps=1e-8):
        W_q = torch.round(W_f * scale).clamp_(min_max[0], min_max[1])
        W_r = W_q / scale
        W_e = self._shrink_lp_op(W_f - W_r, beta, lp_norm)
        W_prime = W_f - W_e
        # dot(W', W_q)
        w_prime_dot_w_q = torch.sum(W_prime * W_q, axis=axis, keepdim=True)
        # ||W_q||²
        w_q_norm_sq = torch.sum(W_q**2, axis=axis, keepdim=True)
        new_scale = w_q_norm_sq / (w_prime_dot_w_q + eps)
        return W_r, W_q, new_scale
        # Shrinking operator
    def _shrink_lp_op(self, x: torch.Tensor, beta: float, lp_norm: float) -> torch.Tensor:
        if lp_norm == 1: 
            #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
            out = torch.abs(x)
            out.sub_(1.0 / beta).clamp_min_(0.0)
            out.mul_(torch.sign(x))
            return out
        else:
            #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1))
            out = torch.abs(x)
            out.sub_((1.0 / beta) * out.pow(lp_norm - 1)).clamp_min_(0.0)
            out.mul_(torch.sign(x))
            return out


        
        
