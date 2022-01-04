from _mnncengine.cv import *
import MNN.numpy as _np

def __to_int(x):
    dtype = x.dtype
    if dtype == _np.int32:
        return x
    return x.astype(_np.int32)
def copyTo(src, mask=None, dst=None):
    if mask is None: return src.copy()
    origin_dtype = src.dtype
    # mask need cast to int
    src = __to_int(src)
    if dst is None: dst = _np.zeros_like(src)
    else: dst = __to_int(dst)
    # mask [h, w] -> [h, w, c]
    mask = _np.expand_dims(mask, -1)
    repeat = [1 for i in range(mask.ndim)]
    repeat[-1] = src.shape[-1]
    mask = _np.tile(mask, repeat)
    if mask.shape != src.shape:
        raise ValueError('mask [height, width] must equal to src [height, width].')
    mask = __to_int(mask)
    # select
    return _np.where(mask, src, dst).astype(origin_dtype)
def bitwise_and(src1, src2, dst=None, mask=None):
    origin_dtype = src1.dtype
    src1 = __to_int(src1)
    src2 = __to_int(src2)
    res = _np.bitwise_and(src1, src2)
    return copyTo(res, mask, dst).astype(origin_dtype)
def bitwise_or(src1, src2, dst=None, mask=None):
    origin_dtype = src1.dtype
    src1 = __to_int(src1)
    src2 = __to_int(src2)
    res = _np.bitwise_or(src1, src2).astype(origin_dtype)
    return copyTo(res, mask, dst)
def bitwise_xor(src1, src2, dst=None, mask=None):
    origin_dtype = src1.dtype
    src1 = __to_int(src1)
    src2 = __to_int(src2)
    res = _np.bitwise_xor(src1, src2)
    return copyTo(res, mask, dst).astype(origin_dtype)
def hconcat(src):
    return _np.concatenate(src, 1)
def vconcat(src):
    return _np.concatenate(src, 0)
