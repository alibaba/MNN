from _mnncengine.cv import *
import _mnncengine.cv as _F
import MNN.numpy as _np
import MNN

def __to_int(x):
    dtype = x.dtype
    if dtype == _np.int32:
        return x
    return x.astype(_np.int32)
def resize(src, dsize=None, fx=None, fy=None, interpolation=INTER_LINEAR, code = None, mean=[], norm=[]):
    if dsize is None and  fx is None and fy is None:
        raise ValueError('reisze must set dsize or fx,fy.')
    if dsize is None: dsize = [0, 0]
    if fx is None: fx = 0
    if fy is None: fy = 0
    if code is None: code = -1
    else: code = hash(code)
    return _F.resize(src, dsize, fx, fy, interpolation, code, mean, norm)
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
def mean(src, mask=None):
    if mask is not None:
        src = copyTo(src, mask)
    res = _np.mean(src, [0, 1])
    if res.ndim == 0: size = 0
    else: size = res.shape[0]
    if size < 4:
        res = _np.pad(res, [0, 4 - size])
    return res
def flip(src, flipCode):
    h, w, c = src.shape
    m = MNN.CVMatrix()
    if flipCode < 0:
        m.write([-1., 0., w-1., 0., -1., h-1.])
    elif flipCode == 0:
        m.write([1., 0., 0., 0., -1., h-1.])
    else:
        m.write([-1., 0., w-1., 0., 1., 0.])
    return warpAffine(src, m, [w, h])
ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2
def rotate(src, rotateMode):
    if rotateMode == ROTATE_90_CLOCKWISE:
        return flip(src.transpose([1, 0, 2]), 1)
    if rotateMode == ROTATE_180:
        return flip(src, -1)
    if rotateMode == ROTATE_90_COUNTERCLOCKWISE:
        return flip(src.transpose([1, 0, 2]), 0)
    return src
