import torch
from contextlib import nullcontext
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

def _quant_dispatch(weight, quant_bit, quant_block, symmetric, awq, hqq):
    # Try GPU quantization first for speed, fall back to CPU on OOM
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            return _quant_on_device(weight.cuda(), quant_bit, quant_block, symmetric, awq, hqq)
        except RuntimeError as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and 'out of memory' not in str(error).lower():
                raise
            torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            return _quant_on_device(weight.to('mps'), quant_bit, quant_block, symmetric, awq, hqq)
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
    from .qat import active_qat_loader
    qat_loader = active_qat_loader.get()
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

    with qat_loader.export_weights(model, onnx_model) if qat_loader is not None else nullcontext():
        torch.onnx.export(
                model, inputs,
                onnx_model,
                **export_kwargs)

class EncoderExport(torch.nn.Module):
    """Adapt native media tensors to a model and return token-major features."""
    def __init__(self, model, embedding_scale=1.0, output_axes=(1, 0, 2), squeeze_dim=None, input_transform=None):
        super().__init__()
        self.model = model
        self.embedding_scale = embedding_scale
        self.output_axes = output_axes
        self.squeeze_dim = squeeze_dim
        self.input_transform = input_transform

    def forward(self, *inputs):
        if self.input_transform is not None:
            inputs = self.input_transform(*inputs)
        outputs = self.model(*inputs)
        features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if self.squeeze_dim is not None:
            features = features.squeeze(self.squeeze_dim)
        if self.output_axes is not None:
            features = features.permute(*self.output_axes)
        if self.embedding_scale != 1.0:
            features = features / self.embedding_scale
        return (features, *outputs[1:]) if isinstance(outputs, (tuple, list)) else features


def embedding_scale(base):
    scale = getattr(base, 'scale_emb', None)
    if scale is None:
        return 1.0
    return float(scale.detach().cpu()) if isinstance(scale, torch.Tensor) else float(scale)
