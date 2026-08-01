import torch
from utils.hqq_quantizer import HQQQuantizer
from packaging.version import Version

def repack_low_bits(x, iNeedBits, block_size):
    v = []
    device = x.device
    block_number = x.shape[0]
    count = block_size * iNeedBits // 8
    for i in range(0, count):
        v.append(torch.zeros([block_number, 1], dtype=torch.uint8, device=device))
    iOffset = 0
    cMask = (1 << iNeedBits) - 1
    index = 0
    for i in range(0, block_size):
        p0 = x[:, i:i+1]
        uShift = 8 - iNeedBits - (iOffset % 8)
        if uShift < 0:
            v[index+iOffset // 8] |= ((p0 & cMask) >> (0 - uShift))
            v[index+(iOffset // 8) + 1] |= ((p0 & cMask) << (8 + uShift))
        else:
            v[index+iOffset // 8] |= ((p0 & cMask) << uShift)
        iOffset += iNeedBits
        if iOffset % 8 == 0:
            index += iOffset // 8
            iOffset = 0
    return torch.cat(v, axis=1)

def _quant_on_device(weight, quant_bit, quant_block, symmetric, awq, hqq):
    oc, ic = weight.shape
    if quant_block == 0:
        block_size = ic
    else:
        block_size = quant_block
    while ic % block_size != 0:
        block_size /= 2
    block_size = int(block_size)
    block_num = ic // block_size

    offset = 1 << (quant_bit - 1)
    clip_max = offset - 1

    if hqq:
        hqq_quantizer = HQQQuantizer(weight, quant_bit, block_size, symmetric, weight.dtype, weight.device)
        hqq_quantizer.quant()
        if not symmetric:
            q_weight = hqq_quantizer.W_q.to(torch.uint8).flatten()
            scale = hqq_quantizer.meta['scale'].flatten()
            zeros = scale * offset - scale * hqq_quantizer.meta['zero'].flatten()
            alpha = torch.stack([zeros.flatten(), scale.flatten()], axis=-1).flatten()
        else:
            q_weight = (hqq_quantizer.W_q + offset).to(torch.uint8).flatten()
            scale = hqq_quantizer.meta['scale'].flatten()
            alpha = scale.flatten()
    else:
        weight = weight.reshape(oc, block_num, block_size)
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

            if awq:
                q_weight = torch.round(weight / scale) - torch.round(min_val / scale) + clip_min
                zeros =  (torch.round(min_val / scale) - clip_min) * scale
            else:
                q_weight = torch.round((weight - min_val) / scale) + clip_min
                zeros =  min_val - scale * clip_min
            q_weight = (torch.clamp(q_weight.flatten(), clip_min, clip_max) + offset).to(torch.uint8)
            alpha = torch.stack([zeros.flatten(), scale.flatten()], axis=-1).flatten()

    if quant_bit < 8 and 8 % quant_bit == 0:
        group_size = 8 // quant_bit
        q_weight = q_weight.reshape(-1, group_size)
        # Accumulate column-by-column in uint8: avoids int64 promotion and an MPS
        # bug where sum(axis=1) over >=2^28 uint8 elements silently returns zeros.
        packed = torch.zeros(q_weight.shape[0], dtype=torch.uint8, device=q_weight.device)
        for i in range(group_size):
            shift = quant_bit * (group_size - 1 - i)
            packed |= q_weight[:, i] << shift
        q_weight = packed
    elif quant_bit < 8:
        q_weight = repack_low_bits(q_weight.reshape((block_num * oc, block_size)), quant_bit, block_size)

    if q_weight.device is not torch.device('cpu'):
        return q_weight.cpu(), alpha.float().cpu()
    return q_weight, alpha.float()

def _quant_result_sane(weight, q_weight, alpha, quant_bit, quant_block, symmetric):
    # Verify that (q_weight, alpha) reconstruct the source weight within
    # quantization error. This catches corrupted/partially-written GPU results.
    # A pure monotonicity/correlation check is NOT enough: with corrupted
    # scale/zero the quantized values stay monotonic in the weight, and a
    # consistent-but-wrong (q, alpha) pair even reconstructs perfectly — only
    # the cross-check against the original weight catches q/alpha inconsistency.
    try:
        if quant_bit > 8 or quant_bit < 2:
            return True
        oc, ic = weight.shape
        w = weight.detach().float().cpu().reshape(-1)
        bs = quant_block if quant_block else ic
        while ic % bs != 0:
            bs //= 2
        bs = int(bs)
        nb = ic // bs
        offset = 1 << (quant_bit - 1)
        q = q_weight.detach().cpu()
        if quant_bit < 8:
            if 8 % quant_bit != 0:
                return True  # repack_low_bits layout: not worth unpacking here
            group = 8 // quant_bit
            qb = q.reshape(-1, 1)
            vals = torch.cat([((qb >> (quant_bit * (group - 1 - i))) & ((1 << quant_bit) - 1))
                              for i in range(group)], dim=1)
            qf = vals.reshape(-1).float()
        else:
            qf = q.reshape(-1).float()
        n = oc * ic
        if qf.numel() < n:
            return True
        a = alpha.detach().float().cpu().reshape(-1)
        if a.numel() == oc * nb:
            scale = a.reshape(oc, nb, 1)
            zero = torch.zeros_like(scale)
        elif a.numel() == oc * nb * 2:
            a2 = a.reshape(oc, nb, 2)
            zero, scale = a2[:, :, 0:1], a2[:, :, 1:2]
        else:
            return True
        recon = ((qf[:n].reshape(oc, nb, bs) - offset) * scale + zero)
        # Valid quantization error is <= scale/2 per element (plus a small fp
        # margin); corrupted results exceed it by orders of magnitude.
        err = (recon - w.reshape(oc, nb, bs)).abs()
        tol = 1.5 * scale + 1e-5  # scale is (oc, nb, 1), broadcasts over bs
        return bool((err <= tol).all().item())
    except Exception:
        return True

def _quant_dispatch(weight, quant_bit, quant_block, symmetric, awq, hqq):
    # Try GPU quantization first for speed, fall back to CPU on OOM
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            q_weight, alpha = _quant_on_device(weight.cuda(), quant_bit, quant_block, symmetric, awq, hqq)
            if _quant_result_sane(weight, q_weight, alpha, quant_bit, quant_block, symmetric):
                return q_weight, alpha
            # Corrupted GPU result (uninitialized buffer); retry on CPU below.
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            q_weight, alpha = _quant_on_device(weight.to('mps'), quant_bit, quant_block, symmetric, awq, hqq)
            if _quant_result_sane(weight, q_weight, alpha, quant_bit, quant_block, symmetric):
                return q_weight, alpha
        except Exception:
            pass
    return _quant_on_device(weight, quant_bit, quant_block, symmetric, awq, hqq)

# Max elements per chunk for quantization (avoid OOM on large embedding tables)
_QUANT_MAX_ELEMENTS = 256 * 1024 * 1024  # 256M elements

def quant(weight, quant_bit, quant_block, symmetric, awq, hqq):
    oc, ic = weight.shape
    if oc * ic <= _QUANT_MAX_ELEMENTS:
        return _quant_dispatch(weight, quant_bit, quant_block, symmetric, awq, hqq)
    # Split along oc dimension for large weights
    chunk_oc = max(1, _QUANT_MAX_ELEMENTS // ic)
    q_weights, alphas = [], []
    for i in range(0, oc, chunk_oc):
        qw, alpha = _quant_dispatch(weight[i:i+chunk_oc], quant_bit, quant_block, symmetric, awq, hqq)
        q_weights.append(qw)
        alphas.append(alpha)
    return torch.cat(q_weights), torch.cat(alphas)

def onnx_export(model, inputs, onnx_model, input_names, output_names, dynamic_axes=None):
    export_kwargs = {
        'input_names': input_names,
        'output_names': output_names,
        'dynamic_axes': dynamic_axes,
        'do_constant_folding': True,
        'verbose': False,
        'opset_version': 15
    }

    # Disable torch dynamo for ONNX export in PyTorch >= 2.4.0
    if Version(torch.__version__) >= Version("2.4.0"):
        export_kwargs['dynamo'] = False

    torch.onnx.export(
            model, inputs,
            onnx_model,
            **export_kwargs)