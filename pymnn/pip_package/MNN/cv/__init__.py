from _mnncengine.cv import *
import _mnncengine.cv as _F
import MNN.expr as _expr
import MNN.numpy as _np
import MNN

# Enum Types
# ColorConversionCodes
COLOR_BGR2BGRA = 0
COLOR_RGB2RGBA = COLOR_BGR2BGRA
COLOR_BGRA2BGR = 1
COLOR_RGBA2RGB = COLOR_BGRA2BGR
COLOR_BGR2RGBA = 2
COLOR_RGB2BGRA = COLOR_BGR2RGBA
COLOR_RGBA2BGR = 3
COLOR_BGRA2RGB = COLOR_RGBA2BGR
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = COLOR_BGR2RGB
COLOR_BGRA2RGBA = 5
COLOR_RGBA2BGRA = COLOR_BGRA2RGBA
COLOR_BGR2GRAY = 6
COLOR_RGB2GRAY = 7
COLOR_GRAY2BGR = 8
COLOR_GRAY2RGB = COLOR_GRAY2BGR
COLOR_GRAY2BGRA = 9
COLOR_GRAY2RGBA = COLOR_GRAY2BGRA
COLOR_BGRA2GRAY = 10
COLOR_RGBA2GRAY = 11
COLOR_BGR2BGR565 = 12
COLOR_RGB2BGR565 = 13
COLOR_BGR5652BGR = 14
COLOR_BGR5652RGB = 15
COLOR_BGRA2BGR565 = 16
COLOR_RGBA2BGR565 = 17
COLOR_BGR5652BGRA = 18
COLOR_BGR5652RGBA = 19
COLOR_GRAY2BGR565 = 20
COLOR_BGR5652GRAY = 21
COLOR_BGR2BGR555 = 22
COLOR_RGB2BGR555 = 23
COLOR_BGR5552BGR = 24
COLOR_BGR5552RGB = 25
COLOR_BGRA2BGR555 = 26
COLOR_RGBA2BGR555 = 27
COLOR_BGR5552BGRA = 28
COLOR_BGR5552RGBA = 29
COLOR_GRAY2BGR555 = 30
COLOR_BGR5552GRAY = 31
COLOR_BGR2XYZ = 32
COLOR_RGB2XYZ = 33
COLOR_XYZ2BGR = 34
COLOR_XYZ2RGB = 35
COLOR_BGR2YCrCb = 36
COLOR_RGB2YCrCb = 37
COLOR_YCrCb2BGR = 38
COLOR_YCrCb2RGB = 39
COLOR_BGR2HSV = 40
COLOR_RGB2HSV = 41
COLOR_BGR2Lab = 44
COLOR_RGB2Lab = 45
COLOR_BGR2Luv = 50
COLOR_RGB2Luv = 51
COLOR_BGR2HLS = 52
COLOR_RGB2HLS = 53
COLOR_HSV2BGR = 54
COLOR_HSV2RGB = 55
COLOR_Lab2BGR = 56
COLOR_Lab2RGB = 57
COLOR_Luv2BGR = 58
COLOR_Luv2RGB = 59
COLOR_HLS2BGR = 60
COLOR_HLS2RGB = 61
COLOR_BGR2HSV_FULL = 66
COLOR_RGB2HSV_FULL = 67
COLOR_BGR2HLS_FULL = 68
COLOR_RGB2HLS_FULL = 69
COLOR_HSV2BGR_FULL = 70
COLOR_HSV2RGB_FULL = 71
COLOR_HLS2BGR_FULL = 72
COLOR_HLS2RGB_FULL = 73
COLOR_LBGR2Lab = 74
COLOR_LRGB2Lab = 75
COLOR_LBGR2Luv = 76
COLOR_LRGB2Luv = 77
COLOR_Lab2LBGR = 78
COLOR_Lab2LRGB = 79
COLOR_Luv2LBGR = 80
COLOR_Luv2LRGB = 81
COLOR_BGR2YUV = 82
COLOR_RGB2YUV = 83
COLOR_YUV2BGR = 84
COLOR_YUV2RGB = 85
COLOR_YUV2RGB_NV12 = 90
COLOR_YUV2BGR_NV12 = 91
COLOR_YUV2RGB_NV21 = 92
COLOR_YUV2BGR_NV21 = 93
COLOR_YUV420sp2RGB = COLOR_YUV2RGB_NV21
COLOR_YUV420sp2BGR = COLOR_YUV2BGR_NV21
COLOR_YUV2RGBA_NV12 = 94
COLOR_YUV2BGRA_NV12 = 95
COLOR_YUV2RGBA_NV21 = 96
COLOR_YUV2BGRA_NV21 = 97
COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21
COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21
COLOR_YUV2RGB_YV12 = 98
COLOR_YUV2BGR_YV12 = 99
COLOR_YUV2RGB_IYUV = 100
COLOR_YUV2BGR_IYUV = 101
COLOR_YUV2RGB_I420 = COLOR_YUV2RGB_IYUV
COLOR_YUV2BGR_I420 = COLOR_YUV2BGR_IYUV
COLOR_YUV420p2RGB = COLOR_YUV2RGB_YV12
COLOR_YUV420p2BGR = COLOR_YUV2BGR_YV12
COLOR_YUV2RGBA_YV12 = 102
COLOR_YUV2BGRA_YV12 = 103
COLOR_YUV2RGBA_IYUV = 104
COLOR_YUV2BGRA_IYUV = 105
COLOR_YUV2RGBA_I420 = COLOR_YUV2RGBA_IYUV
COLOR_YUV2BGRA_I420 = COLOR_YUV2BGRA_IYUV
COLOR_YUV420p2RGBA = COLOR_YUV2RGBA_YV12
COLOR_YUV420p2BGRA = COLOR_YUV2BGRA_YV12
COLOR_YUV2GRAY_420 = 106
COLOR_YUV2GRAY_NV21 = COLOR_YUV2GRAY_420
COLOR_YUV2GRAY_NV12 = COLOR_YUV2GRAY_420
COLOR_YUV2GRAY_YV12 = COLOR_YUV2GRAY_420
COLOR_YUV2GRAY_IYUV = COLOR_YUV2GRAY_420
COLOR_YUV2GRAY_I420 = COLOR_YUV2GRAY_420
COLOR_YUV420sp2GRAY = COLOR_YUV2GRAY_420
COLOR_YUV420p2GRAY = COLOR_YUV2GRAY_420
COLOR_YUV2RGB_UYVY = 107
COLOR_YUV2BGR_UYVY = 108
COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY
COLOR_YUV2BGR_Y422 = COLOR_YUV2BGR_UYVY
COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY
COLOR_YUV2BGR_UYNV = COLOR_YUV2BGR_UYVY
COLOR_YUV2RGBA_UYVY = 111
COLOR_YUV2BGRA_UYVY = 112
COLOR_YUV2RGBA_Y422 = COLOR_YUV2RGBA_UYVY
COLOR_YUV2BGRA_Y422 = COLOR_YUV2BGRA_UYVY
COLOR_YUV2RGBA_UYNV = COLOR_YUV2RGBA_UYVY
COLOR_YUV2BGRA_UYNV = COLOR_YUV2BGRA_UYVY
COLOR_YUV2RGB_YUY2 = 115
COLOR_YUV2BGR_YUY2 = 116
COLOR_YUV2RGB_YVYU = 117
COLOR_YUV2BGR_YVYU = 118
COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2
COLOR_YUV2BGR_YUYV = COLOR_YUV2BGR_YUY2
COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2
COLOR_YUV2BGR_YUNV = COLOR_YUV2BGR_YUY2
COLOR_YUV2RGBA_YUY2 = 119
COLOR_YUV2BGRA_YUY2 = 120
COLOR_YUV2RGBA_YVYU = 121
COLOR_YUV2BGRA_YVYU = 122
COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2
COLOR_YUV2BGRA_YUYV = COLOR_YUV2BGRA_YUY2
COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2
COLOR_YUV2BGRA_YUNV = COLOR_YUV2BGRA_YUY2
COLOR_YUV2GRAY_UYVY = 123
COLOR_YUV2GRAY_YUY2 = 124
COLOR_YUV2GRAY_Y422 = COLOR_YUV2GRAY_UYVY
COLOR_YUV2GRAY_UYNV = COLOR_YUV2GRAY_UYVY
COLOR_YUV2GRAY_YVYU = COLOR_YUV2GRAY_YUY2
COLOR_YUV2GRAY_YUYV = COLOR_YUV2GRAY_YUY2
COLOR_YUV2GRAY_YUNV = COLOR_YUV2GRAY_YUY2
COLOR_RGBA2mRGBA = 125
COLOR_mRGBA2RGBA = 126
COLOR_RGB2YUV_I420 = 127
COLOR_BGR2YUV_I420 = 128
COLOR_RGB2YUV_IYUV = COLOR_RGB2YUV_I420
COLOR_BGR2YUV_IYUV = COLOR_BGR2YUV_I420
COLOR_RGBA2YUV_I420 = 129
COLOR_BGRA2YUV_I420 = 130
COLOR_RGBA2YUV_IYUV = COLOR_RGBA2YUV_I420
COLOR_BGRA2YUV_IYUV = COLOR_BGRA2YUV_I420
COLOR_RGB2YUV_YV12 = 131
COLOR_BGR2YUV_YV12 = 132
COLOR_RGBA2YUV_YV12 = 133
COLOR_BGRA2YUV_YV12 = 134
COLOR_COLORCVT_MAX = 143
# InterpolationFlags
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3
INTER_LANCZOS4 = 4
INTER_LINEAR_EXACT = 5
INTER_NEAREST_EXACT = 6
INTER_MAX = 7
WARP_FILL_OUTLIERS = 8
WARP_INVERSE_MAP = 16
# BorderTypes
BORDER_CONSTANT = 0
BORDER_REFLECT_101 = 1
BORDER_REFLECT = 2
BORDER_REFLECT101 = BORDER_REFLECT_101
BORDER_DEFAULT = BORDER_REFLECT_101
'''
# not support padtype
BORDER_REPLICATE = 1
BORDER_WRAP = 3
BORDER_TRANSPARENT = 5
BORDER_ISOLATED = 16
'''
# ThresholdTypes
THRESH_BINARY = 0
THRESH_BINARY_INV = 1
THRESH_TRUNC = 2
THRESH_TOZERO = 3
THRESH_TOZERO_INV = 4
THRESH_MASK = 7
THRESH_OTSU = 8
THRESH_TRIANGLE = 16
# RetrievalModes
RETR_EXTERNAL = 0
RETR_LIST = 1
RETR_CCOMP = 2
RETR_TREE = 3
RETR_FLOODFILL = 4
# ContourApproximationModes
CHAIN_APPROX_NONE = 1
CHAIN_APPROX_SIMPLE = 2
CHAIN_APPROX_TC89_L1 = 3
CHAIN_APPROX_TC89_KCOS = 4
# LineTypes
FILLED = -1
LINE_4 = 4
LINE_8 = 8
LINE_AA = 16
# ImreadModes
IMREAD_GRAYSCALE = 0
IMREAD_COLOR = 1
IMREAD_ANYDEPTH = 4
# rotateMode
ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2
# solvePnP
SOLVEPNP_ITERATIVE = 0
SOLVEPNP_SQPNP = 8
# decomp types
DECOMP_LU = 0
DECOMP_SVD = 1
DECOMP_EIG = 2
DECOMP_CHOLESKY = 3
DECOMP_QR = 4
DECOMP_NORMAL = 16
# norm tyes
NORM_INF = 1
NORM_L1 = 2
NORM_L2 = 4
NORM_MINMAX = 32
# adaptive threshold types
ADAPTIVE_THRESH_MEAN_C = 0
ADAPTIVE_THRESH_GAUSSIAN_C = 1
# helper functions
def __to_int(x):
    dtype = x.dtype
    if dtype == _np.int32:
        return x
    return x.astype(_np.int32)
# geometric
def resize(src, dsize=None, fx=None, fy=None, interpolation=INTER_LINEAR, code = None, mean=[], norm=[]):
    if dsize is None and  fx is None and fy is None:
        raise ValueError('reisze must set dsize or fx,fy.')
    if dsize is None: dsize = [0, 0]
    if fx is None: fx = 0
    if fy is None: fy = 0
    if code is None: code = -1
    else: code = hash(code)
    return _F.resize(src, dsize, fx, fy, interpolation, code, mean, norm)
def warpAffine(src, M, dsize, flag=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=0, code=None, mean=[], norm=[]):
    if code is None: code = -1
    else: code = hash(code)
    return _F.warpAffine(src, M, dsize, flag, borderMode, borderValue, code, mean, norm)
def copyTo(src, mask=None, dst=None):
    if mask is None: return src.copy()
    origin_dtype = src.dtype
    if dst is None: dst = _np.zeros_like(src)
    else: dst = __to_int(dst)
    if src.ndim > mask.ndim:
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
def solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags=SOLVEPNP_ITERATIVE):
    rv, tv = _F.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
    return (True, rv, tv)
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
        res = _expr.reduce_sum(src, [0, 1]) / _np.sum(mask)
    else:
        res = _np.mean(src, [0, 1])
    if res.ndim == 0:
        res = _np.expand_dims(res, 0)
    size = res.shape[0]
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
def rotate(src, rotateMode):
    if rotateMode == ROTATE_90_CLOCKWISE:
        return flip(src.transpose([1, 0, 2]), 1)
    if rotateMode == ROTATE_180:
        return flip(src, -1)
    if rotateMode == ROTATE_90_COUNTERCLOCKWISE:
        return flip(src.transpose([1, 0, 2]), 0)
    return src
def normalize(src, dst, alpha=None, beta=None, norm_type=None, dtype=None, mask=None):
    dtype = src.dtype
    if norm_type == NORM_INF:
        dst = src / _expr.reduce_max(src)
    if norm_type == NORM_L1:
        dst = src / _expr.reduce_sum(src)
    if norm_type == NORM_L2:
        dst = src / _expr.sqrt(_expr.reduce_sum(src * src))
    if norm_type == NORM_MINMAX:
        dmin, dmax = min(alpha, beta), max(alpha, beta)
        smin, smax = _expr.reduce_min(src), _expr.reduce_max(src)
        scale = (dmax - dmin) * 1./(smax - smin)
        shift = dmin - smin * scale
        dst = src * scale + shift
    if dtype == _np.uint8:
        dst = _expr.minimum(_expr.maximum(_expr.round(dst), 0), 255)
    return _expr.cast(dst, dtype)
def merge(mv):
    shape = mv[0].shape
    if len(shape) > 3:
        return _np.stack(mv, -1)
    expand_dim = None
    if len(shape) == 1:
        expand_dim = [1, 2]
    elif len(shape) == 2:
        expand_dim = [2]
    if expand_dim is not None:
        mv = [_np.expand_dims(m, expand_dim) for m in mv]
    return _np.concatenate(mv, 2)
def split(m):
    shape = m.shape
    if len(shape) == 3:
        dst = _expr.split(m, [1]*shape[2], -1)
        dst = [_expr.squeeze(d, -1) for d in dst]
        return dst
    if len(shape) == 1:
        m = _expr.unsqueeze(m, 1)
    return (m,)
def addWeighted(src1, alpha, src2, beta, gamma):
    dtype = src1.dtype
    dst = src1 * alpha + src2 * beta + gamma
    return dst.astype(dtype)