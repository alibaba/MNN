import math
import MNN.expr as _F

# origin function
_Max = max
_All = all

# type alias
uint8 = _F.uint8
int32 = _F.int
int64 = _F.int64
float32 = _F.float
float64 = _F.double
# constant
pi = math.pi
# inf = math.inf
inf = float('inf')

# helper functions
def __not_impl(*args):
    raise NotImplementedError('MNN.numpy not implemet this function now.')
def __order_assert(order):
    if order is not None and order not in 'CK':
        raise RuntimeError("MNN.numpy just support order=\"C|K\"")
def __array_like_type(prototype, dtype, order, shape):
    if dtype is not None:
        dst_dtype = dtype
    else:
        dst_dtype = prototype.dtype
    if order != 'K' and shape is not None:
        dst_shape = shape
    else:
        dst_shape = prototype.shape
    return dst_dtype, dst_shape
def __normal_axis(axis, ndim):
    axis = axis if axis >= 0 else axis + ndim
    if axis < 0:
        raise ValueError('AxisError: axes_arg: axis %d is out of bounds for array of dimension %d'%{axis, ndim})
    return axis
def __normal_axes(axes, ndim):
    axes = _F._to_axis(axes)
    axes = tuple([__normal_axis(ax, ndim) for ax in axes])
    return axes
def __broadcast_shape(args):
    max_length = 0
    shapes = []
    # get max_length
    for arg in args:
        shape = arg.shape
        max_length = _Max(len(shape), max_length)
        shapes.append(shape)
    # padding 1 to max_length
    for shape in shapes:
        for i in range(max_length - len(shape)):
            shape.insert(i, 1)
    dst_shape = []
    # get every dim
    for i in range(max_length):
        dims = set()
        for shape in shapes:
            dims.add(shape[i])
        dims.remove(1)
        if len(dims) > 1:
            raise ValueError('shape mismatch: %d cannot be broadcast to %d'%{dims.pop(), dims.pop()})
        dst_shape.append(dims.pop() if len(dims) == 1 else 1)
    return dst_shape
def __override_operator(class_object, operator, func):
    import gc
    attr_dict = gc.get_referents(class_object.__dict__)[0]
    attr_dict[operator] = func

# Array creation routines
# 1. from shape or value
def empty(shape, dtype=float32, order='C'):
    __order_assert(order)
    return _F.placeholder(shape, _F.NCHW, dtype)
def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
    __order_assert(order)
    dst_dtype, dst_shape = __array_like_type(prototype, dtype, order, shape)
    return _F.placeholder(dst_shape, prototype.data_format, dst_dtype)
def eye(N, M=None, k=0, dtype=float32, order='C'):
    __order_assert(order)
    M = N if M is None else M
    x = _F.one_hot(arange(0 + k, N + k, 1), M)
    if dtype != float32:
        x = _F.cast(x, dtype)
    return x
def identity(n, dtype=float32):
    return eye(n, dtype=dtype)
def full(shape, fill_value, dtype=None, order='C'):
    __order_assert(order)
    return _F.fill(_F._to_var(shape), _F.scalar(fill_value, dtype))
def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    dst_dtype, dst_shape = __array_like_type(a, dtype, order, shape)
    return full(dst_shape, fill_value, dst_dtype)
def ones(shape, dtype=None, order='C'):
    return full(shape, 1, dtype, order)
def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    return full_like(a, 1, dtype, order, subok, shape)
def zeros(shape, dtype=None, order='C'):
    return full(shape, 0, dtype, order)
def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    return full_like(a, 0, dtype, order, subok, shape)
# 2. from existing data
def copy(a, order='K', subok=False):
    __order_assert(order)
    return _F.clone(a, True)
_Copy = copy
def array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0, like=None):
    __order_assert(order)
    if isinstance(object, _F.Var):
        if copy:
            x = _Copy(object)
        else:
            x = object
    else:
        if not isinstance(object, _F._Sequence):
            x = _F.scalar(object, dtype)
        else:
            # get shape and dtype of sequence
            dst_shape, item_type = _F._list_shape_type(object)
            x = _F.const(object, dst_shape, dtype=item_type)
    # if give dtype, cast to dtype
    if dtype is not None and dtype != x.dtype:
        x = _F.cast(x, dtype)
    # if give ndmin, unsqueeze
    if ndmin > 0:
        dim = x.ndim
        if ndmin > dim:
            x = _F.unsqueeze(x, [i for i in range(ndmin - dim)])
    return x
def asarray(a, dtype=None, order=None):
    return array(a, dtype, order=order)
def asanyarray(a, dtype=None, order=None):
    return array(a, dtype, order=order)
def ascontiguousarray(a, dtype=None, order=None):
    return array(a, dtype, order=order)
def asmatrix(data, dtype=None):
    return array(data, dtype, ndmin=2)
def frombuffer(buffer, dtype=float32, count=-1, offset=0):
    # TODO: impl 
    return array(buffer, dtype)
def fromfile(file, dtype=float32, count=-1, sep='', offset=0):
    __not_impl()
def fromfunction(function, shape, dtype=float32):
    __not_impl()
def fromiter(iter, dtype, count=-1):
    __not_impl()
def fromstring(string, dtype=float32, count=- 1, sep=''):
    __not_impl()
def loadtxt(fname, dtype=float32, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding='bytes', max_rows=None):
    __not_impl()
# 3. Numerical ranges
def __arange_3(start, stop, step=1, dtype=None):
    _type = type(stop)
    start = _F.scalar(_type(start), dtype)
    limit = _F.scalar(stop, dtype)
    delta = _F.scalar(_type(step), dtype)
    x = _F.range(start, limit, delta)
    if dtype is not None and x.dtype != dtype:
        x = _F.cast(x, dtype)
    return x
def __arange_1(stop, dtype=None):
    return __arange_3(0, stop, 1, dtype)
def arange(*args, **kargs):
    if 'dtype' in kargs: dtype=kargs['dtype']
    else: dtype = None
    if len(args) == 1:
        return __arange_1(args[0], dtype)
    if len(args) == 4:
        return __arange_3(*args)
    return __arange_3(*args, dtype=dtype)
def linspace(start, stop, num=50, endpoint=True,
             retstep=False, dtype=None, axis=0):
    step = (stop - start) / num
    x = arange(start, stop, step, dtype)
    if endpoint:
        # TODO
        raise RuntimeError("MNN.numpy arange not support endpoint")
    if retstep:
        x = (x, step)
    return x
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    pow = linspace(start, stop, num, endpoint, False, dtype, axis)
    return _F.pow(_F.scalar(base, dtype), pow)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    base = pow(stop / _F._Float(start), 1./ num)
    start = math.log(start, base)
    return logspace(start, _F._Float(num), num, endpoint, base, dtype, axis)
def meshgrid(xi, copy=True, sparse=False, indexing='xy'): __not_impl()
# 4. Building matrices
def diag(v, k=0):__not_impl()
def diagflat(v, k=0):__not_impl()
def tri(N, M=None, k=0, dtype=float32):__not_impl()
def tril(m, k=0):__not_impl()
def triu(m, k=0):__not_impl()
def vander(x, N=None, increasing=False):__not_impl()
# 5. The Matrix class
def mat(data, dtype=None):
    return asmatrix(data, dtype)
matrix = mat
def bmat(obj, ldict=None, gdict=None):__not_impl()
def repmat(a, m, n):
    row = tile(a, m)
    return reshape(tile(row, n), [m, -1])
# Array manipulation routines
# 1. Basic operations
def copyto(dst, src, casting='same_kind', where=True):
    dst = copy(src)
def shape(a):
    return tuple(a.shape)
# 2. Changing array shape
def reshape(a, newshape, order='C'):
    __order_assert(order)
    return _F.reshape(a, newshape)
def ravel(a, order='C'):
    return reshape(a, [-1], order)
# 3. Transpose-like operations
def moveaxis(a, source, destination):
    ndim = a.ndim
    assert(type(source) == type(destination))
    source = __normal_axes(source, ndim)
    destination = __normal_axes(destination, ndim)
    assert(len(source) == len(destination))
    axes = [n for n in range(ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        axes.insert(dest, src)
    return _F.transpose(a, axes)
def rollaxis(a, axis, start=0):
    ndim = a.ndim
    axis = __normal_axis(axis, ndim)
    start = __normal_axis(start, ndim)
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < ndim + 1):
        raise ValueError(msg % ('start', -ndim, 'start', ndim + 1, start))
    if axis < start:
        start -= 1
    if axis == start:
        return a
    axes = list(range(0, ndim))
    axes.remove(axis)
    axes.insert(start, axis)
    return _F.transpose(a, axes)
def swapaxes(a, axis1, axis2):
    axes = list(range(0, a.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return _F.transpose(a, axes)
def transpose(a, axes=None):
    if axes is None:
        axes = [i for i in reversed(range(a.ndim))]
    return _F.transpose(a, axes)
# 4. Changing number of dimensions
def __atleast_nd(n, *arys):
    res = []
    for ary in arys:
        ary = array(ary, copy=False, ndmin=n)
        res.append(ary)
    if len(res) == 1:
        return res[0]
    else:
        return res
def atleast_1d(*arys):
    return __atleast_nd(1, *arys)
def atleast_2d(*arys):
    return __atleast_nd(2, *arys)
def atleast_3d(*arys):
    res = []
    for ary in arys:
        ary = array(ary, copy=False, ndmin=1)
        if ary.ndim == 2:
            ary = expand_dims(ary, (-1))
        elif ary.ndim == 1:
            ary = expand_dims(ary, (0, -1))
        res.append(ary)
    if len(res) == 1:
        return res[0]
    else:
        return res
def broadcast(x, y):__not_impl()
def broadcast_to(array, shape):
    array = asarray(array)
    src_shape = array.shape
    if not _F._can_broadcast(src_shape, shape):
        raise ValueError('can\'t broadcast from ', src_shape, ' to ', shape, '.')
    return _F.broadcast_to(array, shape)
def broadcast_arrays(*args):
    args = [array(_m, copy=False) for _m in args]
    shape = __broadcast_shape(args)
    res = []
    for ary in args:
        res.append(broadcast_to(ary, shape))
    if len(res) == 1:
        return res[0]
    else:
        return res
def expand_dims(x, axis):
    if type(axis) not in (tuple, list):
        axis = [int(axis)]
    return _F.unsqueeze(x, axis)
def squeeze(a, axis=None):
    if axis is None:
        axis = []
    if type(axis) not in (tuple, list):
        axis = [int(axis)]
    return _F.squeeze(a, axis)
# 5. Changing kind of array
def asfarray(a, dtype=float64):
    return array(a, dtype=float32)
def asfortranarray(a, dtype=None):
    __not_impl()
def ascontiguousarray(a, dtype=None):
    return array(a, dtype=dtype)
def asarray_chkfinite(a, dtype=None):
    # TODO: check inf
    return array(a, dtype=dtype)
def asscalar(a):
    return a.read_as_tuple()[0]
def require(a, dtype=None, requirements=None):
    __not_impl()
# 6. Joining arrays
def concatenate(args, axis=0, out=None, dtype=None, casting="same_kind"):
    arys = []
    for arg in args:
        arg = array(arg, copy=False)
        if axis is None:
            arg = ravel(arg)
        arys.append(arg)
    return _F.concat(arys, axis)
def stack(arrays, axis=0, out=None):
    arrays = [array(_m, copy=False) for _m in arrays]
    return _F.stack(arrays, axis)
def block(arrays):
    __not_impl()
def vstack(tup):
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs)
def hstack(tup):
    arrs = atleast_1d(*tup)
    if arrs and arrs[0].ndim == 1:
        return concatenate(arrs, 0)
    else:
        return concatenate(arrs, 1)
def dstack(tup):
    arrs = atleast_3d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs, 2)
def column_stack(tup):
    arrays = []
    for v in tup:
        arr = asanyarray(v)
        if arr.ndim < 2:
            arr = transpose(array(arr, copy=False, subok=True, ndmin=2))
        arrays.append(arr)
    return concatenate(arrays, 1)
def row_stack(tup):
    return vstack(tup)
# 7. Splitting arrays
def split(ary, indices_or_sections, axis=0):
    size_splits = []
    idx_exceeds = False
    if type(indices_or_sections) not in (tuple, list):
        size_splits = [int(indices_or_sections)]
    else:
        size_splits.append(indices_or_sections[0])
        axis_length = ary.shape[axis]
        for i in range(1, len(indices_or_sections)):
            now_idx = indices_or_sections[i]
            if indices_or_sections[i] > axis_length:
                idx_exceeds = True
                now_idx = axis_length            
            size_splits.append(now_idx - indices_or_sections[i-1])
    res = _F.split(ary, size_splits, axis)
    if idx_exceeds:
        res.append(array([0])) # TODO: MNN not support empty Var
    return res
def array_split(ary, indices_or_sections, axis=0):
    msg = 'MNN.numpy.array_split not support %d divide the axis %d.'
    if ary.shape[axis] % indices_or_sections != 0:
        raise ValueError(msg%(indices_or_sections, ary.shape[axis]))
    return split(ary, indices_or_sections, axis)
def dsplit(ary, indices_or_sections):
    if ary.ndim < 3:
        raise ValueError('dsplit only works on arrays of 3 or more dimensions')
    return split(ary, indices_or_sections, 2)
def hsplit(ary, indices_or_sections):
    if ary.ndim == 0:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    if ary.ndim > 1:
        return split(ary, indices_or_sections, 1)
    else:
        return split(ary, indices_or_sections, 0)
def vsplit(ary, indices_or_sections):
    if ary.ndim < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    return split(ary, indices_or_sections, 0)
# 8. Tiling arrays
def tile(A, reps):
    return _F.tile(array(A, copy=False), array(reps, ndmin=1))
def repeat(a, repeats, axis=None):
    a = expand_dims(ravel(a), -1)
    if type(repeats) in (tuple, list):
        raise ValueError('MNN.numpy.repeat just support repeats:int')
    repeats = [1, int(repeats)]
    return ravel(tile(a, repeats))
# 9. Adding and removing elements
def delete(arr, obj, axis=None):__not_impl()
def insert(arr, obj, values, axis=None):__not_impl()
def append(arr, values, axis=None):
    __not_impl()
def resize(a, new_shape):__not_impl()
def trim_zeros(filt, trim='fb'):__not_impl()
def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):__not_impl()
# 10. Rearranging elements
def flip(m, axis=None):__not_impl()
def fliplr(m):__not_impl()
def flipud(m):__not_impl()
def roll(a, shift, axis=None):__not_impl()
def rot90(m, k=1, axes=(0, 1)):__not_impl()
# Binary operations
# 1. Elementwise bit operations [Not Impl]
bitwise_and = _F.bitwise_and
bitwise_or = _F.bitwise_or
bitwise_xor = _F.bitwise_xor
invert = left_shift = \
right_shift = packbits = unpackbits = binary_repr = base_repr = __not_impl
# String operations [Not Impl]
# Indexing routines
# 1. Generating index arrays
def where(condition, x, y):
    return _F.select(condition, x, y)
def indices(dimensions, dtype=int32, sparse=False):__not_impl()
def ix_(*args):__not_impl()
def ravel_multi_index(multi_index, dims, mode='raise', order='C'):__not_impl()
def unravel_index(indices, shape, order='C'):
    __order_assert(order)
    indices = array(indices)
    shape = array(shape)
    x = _F.unravel_index(indices, shape)
    return (squeeze(m_, 0) for m_ in split(x, shape.size))
def diag_indices(n, ndim=2):__not_impl()
def diag_indices_from(arr):__not_impl()
def mask_indices(n, mask_func, k=0):__not_impl()
def tril_indices(n, k=0, m=None):__not_impl()
def tril_indices_from(arr, k=0):__not_impl()
def triu_indices(n, k=0, m=None):__not_impl()
def triu_indices_from(arr, k=0):__not_impl()
# 2. Indexing-like
take = take_along_axis = choose = compress = diagonal = select = __not_impl
# 3. Inserting data into arrays
place = put = put_along_axis = putmask = fill_diagonal = __not_impl
# 4. Iterating over arrays
nditer = ndenumerate = ndindex = nested_iters = flatiter = __not_impl
# Input and output [Not Impl]
# Linear algebra
def dot(a, b, out=None):
    a = array(a)
    b = array(b)
    # cast to same type
    a, b = _F._match_dtype(a, b)
    ad = a.ndim
    bd = b.ndim
    def assert_dim(x, y):
        if x != y:
            raise ValueError('shapes not aligned: ', x, '!=', y, '.')
    # If either a or b is 0-D (scalar), it is equivalent to multiply.
    if ad == 0 or bd == 0:
        return asscalar(multiply(a, b))
    # If both a and b are 1-D arrays, it is inner product of vectors.
    if a.ndim == 1 and b.ndim == 1:
        assert_dim(a.shape[0], b.shape[0])
        return sum(multiply(a, b))
    # If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    if (a.ndim > 1 and b.ndim == 1) or (a.ndim == 1 and b.ndim > 1):
        assert_dim(a.shape[-1], b.shape[-1])
        return sum(multiply(a, b), [-1])
    # matmul just support float32
    origin_dtype = a.dtype
    a, b = _F._match_dtype(a, b, float32)
    # If both a and b are 2-D arrays, it is matrix multiplication.
    if a.ndim == 2 and b.ndim == 2:
        assert_dim(a.shape[1], b.shape[0])
        return _F.cast(_F.matmul(a, b), origin_dtype)
    # If a is an N-D array and b is an M-D array (where M>=2),
    # it is a sum product over the last axis of a and the second-to-last axis of b.
    if a.ndim > 2 and b.ndim > 1:
        reduce_dim = a.shape[-1]
        assert_dim(reduce_dim, b.shape[-2])
        a_shape = a.shape
        b_shape = b.shape
        a_shape.pop(-1)
        b_shape.pop(-2)
        dst_shape = a_shape + b_shape
        a = reshape(a, [-1, reduce_dim])
        b = reshape(moveaxis(b, -2, 0), [3, -1])
        res = _F.cast(_F.matmul(a, b), origin_dtype)
        return reshape(res, dst_shape)
def vdot(a, b):
    return dot(a, b)
def inner(a, b):
    return dot(a, b)
def outer(a, b):__not_impl()
def matmul(a, b):
    return dot(a, b)
def tensordot(a, b, axes=2):
    pass
# Logic functions
# 1. Truth value testing
def all(a, axis=None, out=None, keepdims=None):
    if axis is None:
        return bool(asscalar(_F.reduce_all(a)))
    keepdims = False if keepdims is None else keepdims
    return bool(asscalar(_F.reduce_all(a, axis, keepdims)))
def any(a, axis=None, out=None, keepdims=None):
    if axis is None:
        return bool(asscalar(_F.reduce_any(a)))
    keepdims = False if keepdims is None else keepdims
    return bool(asscalar(_F.reduce_any(a, axis, keepdims)))
def array_equal(a1, a2, equal_nan=False):
    return all(_F.equal(a1, a2))
def array_equiv(a1, a2):
    return array_equal(a1, a2)
greater = _F.greater
greater_equal = _F.greater_equal
less = _F.less
less_equal = _F.less_equal
equal = _F.equal
not_equal = _F.not_equal
# Mathematical functions
sin = _F.sin
cos = _F.cos
tan = _F.tan
arcsin = _F.asin
arccos = _F.acos
arctan = _F.atan
arctan2 = _F.atan2
sinh = _F.sinh
cosh = _F.cosh
tanh = _F.tanh
arcsinh = _F.asinh
arccosh = _F.acosh
arctanh = _F.atanh
around = _F.round
round_ = _F.round
rint = _F.round
fix = _F.round
floor = _F.floor
ceil = _F.ceil
trunc = _F.round
def prod(a, axis=None, out=None, keepdims=False):
    if axis is None: return asscalar(_F.reduce_prod(ravel(a)))
    res = _F.reduce_prod(a, axis, keepdims)
    if res.ndim == 0:
        return asscalar(res)
    return res
def sum(a, axis=None, out=None, keepdims=False):
    if axis is None: return asscalar(_F.reduce_sum(ravel(a)))
    res = _F.reduce_sum(a, axis, keepdims)
    if res.ndim == 0:
        return asscalar(res)
    return res
nanprod = prod
nansum = sum
sqrt = _F.sqrt
exp = _F.exp
expm1 = _F.expm1
log = _F.log
log1p = _F.log1p
sign = _F.sign
reciprocal = _F.reciprocal
positive = _F.clone
negative = _F.negative
multiply = _F.multiply
add = _F.add
divide = _F.divide
power = _F.pow
subtract = _F.subtract
true_divide = divide
floor_divide = _F.floordiv
float_power = power
mod = _F.mod
fmod = mod
remainder = mod
square = _F.square
abs = _F.abs
absolute = abs
fabs = abs
maximum = _F.maximum
minimum = _F.minimum
fmax = maximum
fmin = minimum
def hypot(x1, x2):
    return sqrt(x1*x1 + x2*x2)
def exp2(x):
    x = array(x)
    return _F.scalar(2, x.dtype) ** x
def log2(x):
    x = array(x)
    return log(x) / log(_F.scalar(2, x.dtype))
def log10(x):
    x = array(x)
    return log(x) / log(_F.scalar(10, x.dtype))
def logaddexp(x1, x2):
    x1 = array(x1)
    x2 = array(x2)
    return log(exp(x1) + exp(x2))
def logaddexp2(x1, x2):
    x1 = array(x1)
    x2 = array(x2)
    return log2(exp2(x1) + exp2(x2))
def sinc(x):
    x = array(x)
    _pi = _F.scalar(pi, x.dtype)
    return sin(_pi * x) / (_pi * x)
def signbit(x):
    x = array(x)
    return (sign(x) == _F.scalar(-1, x.dtype))
def copysign(x1, x2):
    return x1 * sign(x2)
def ldexp(x1, x2):
    return x1 * exp2(x2)
def divmod(x1, x2):
    return (floor_divide(x1, x2), mod(x1, x2))
def modf(x):
    x = array(x)
    out1, out2 = divmod(x, _F.scalar(1, x.dtype))
    return (out2, out1)
def clip(x, a_min, a_max):
    types = (list, tuple, type(x))
    if type(a_min) in types or type(a_max) in types:
        raise ValueError('MNN.numpy.clip just support scalar a_min/a_max.')
    dtype = x.dtype
    return _F.cast(_F.relu6(_F.cast(x), a_min, a_max), dtype)
def cbrt(x):
    x = array(x)
    return power(x, _F.scalar(1./3, x.dtype))
def degrees(x):__not_impl()
def radians(x):__not_impl()
def unwrap(p, discont=None, axis=- 1):__not_impl()
def deg2rad(x):__not_impl()
def rad2deg(x):__not_impl()
def cumprod(x):__not_impl()
def cumsum(x):__not_impl()
def nancumprod(x):__not_impl()
def nancumsum(x):__not_impl()
def diff(a, n=1, axis=-1):__not_impl()
def ediff1d(ary, to_end=None, to_begin=None):__not_impl()
def gradient(f, varargs, axis=None, edge_order=1):__not_impl()
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):__not_impl()
def trapz(y, x=None, dx=1.0, axis=-1):__not_impl()
def i0(x):__not_impl()
def frexp(x):__not_impl()
def nextafter(x1, x2):__not_impl()
def spacing(x):__not_impl()
def lcm(x1, x2):__not_impl()
def gcd(x1, x2):__not_impl()
def convolve(a, v):
    __not_impl()
    a = _F.convert(a, _F.NC4HW4)
    r = _F.conv2d(a, v, 0)
    return _F.convert(r, _F.NCHW)
def heaviside(x1, x2):__not_impl()
def nan_to_num(x):__not_impl()
def real_if_close(a, tol=100):__not_impl()
def interp(x, xp, fp):__not_impl()
# Padding Arrays
def pad(array, pad_width, mode='constant'):
    if mode == 'constant':
        mode = _F.CONSTANT
    elif mode == 'reflect':
        mode = _F.REFLECT
    elif mode == 'symmetric':
        mode = _F.SYMMETRIC
    else:
        raise TypeError('MNN.numpy.pad just support `mode` = (`constant`, `reflect`, `symmetric`).')
    array = asarray(array)
    pad_width = asarray(pad_width)
    if pad_width.dtype != int32:
        raise TypeError('`pad_width` must be of integral type.')
    pad_width = broadcast_to(pad_width, [array.ndim, 2])
    return _F.pad(array, pad_width, mode)
# Sorting, searching, and counting
# 1. Sorting
def sort(a, axis=- 1, kind=None, order=None):__not_impl()
def lexsort(keys, axis=-1):__not_impl()
def argsort(a, axis=-1, kind=None, order=None): __not_impl()
def msort(a): return sort(a, axis=0)
def sort_complex(a): __not_impl()
def partition(a, kth, axis=- 1, kind='introselect', order=None): __not_impl()
def argpartition(a, kth, axis=- 1, kind='introselect', order=None): __not_impl()
# 2. Searching
def argmax(a, axis=None, out=None):
    if axis is None:
        return asscalar(_F.argmax(ravel(a), 0))
    return _F.argmax(a, axis)
nanargmax = argmax
def argmin(a, axis=None, out=None):__not_impl()
nanargmin = argmin
def argwhere(a):
    mask = not_equal(a, _F.scalar(0, a.dtype))
    return _F.where(mask)
def nonzero(a):
    res = argwhere(a)
    if a.ndim == 1:
        return (ravel(res),)
    return tuple([ravel(m_) for m_ in split(transpose(res), 2)])
def flatnonzero(a):
    return nonzero(ravel(a))[0]
def searchsorted(a, v, side='left', sorter=None):__not_impl()
def extract(condition, arr):__not_impl()
# 3. Counting
def count_nonzero(a, axis=None, keepdims=False):
    mask = not_equal(a, _F.scalar(0, a.dtype))
    axis = _F._to_axis(axis)
    return sum(mask, axis, keepdims)
# Statistics
# 1. Order statistics
def amin(a, axis=None, out=None, keepdims=False):
    if axis is None: return asscalar(_F.reduce_min(ravel(a)))
    return _F.reduce_min(a, axis, keepdims)
def amax(a, axis=None, out=None, keepdims=False):
    if axis is None: return asscalar(_F.reduce_max(ravel(a)))
    return _F.reduce_max(a, axis, keepdims)
nanmin = amin
nanmax = amax
max = amax
min = amin
def ptp(a, axis=None):
    return amax(a, axis) - amin(a, axis)
def percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False): __not_impl()
nanpercentile = percentile
def quantile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False): __not_impl()
nanquantile = quantile
# 2. Averages and variances
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):__not_impl()
def mean(a, axis=None, dtype=float32, out=None, keepdims=False):
    a = _F.cast(a, dtype)
    if axis is None: return asscalar(_F.reduce_mean(ravel(a)))
    return _F.reduce_mean(a, axis, keepdims)
def average(a, axis=None, weights=None, returned=False):
    if weights is not None:
        raise ValueError('MNN.numpy average not support `weights`.')
    return mean(a, axis)
def var(a, axis=None, dtype=float32, out=None, ddof=0, keepdims=False):
    a = _F.cast(a, dtype)
    return mean(abs(square(a - mean(a, axis, keepdims=True))), axis)
def std(a, axis=None, dtype=float32, out=None, ddof=0, keepdims=False):
    res = sqrt(var(a, axis, dtype))
    if axis is None: return asscalar(res)
    return res
nanmedian = median
nanmean = mean
nanstd = std
nanvar = var
# Correlating
corrcoef = correlate = cov = __not_impl
# Histograms
histogram = histogram2d = histogramdd = bincount = histogram_bin_edges = digitize = __not_impl

# numpy ndarray functions
__override_operator(_F.Var, "all", all)
__override_operator(_F.Var, "any", any)
__override_operator(_F.Var, "argmax", argmax)
__override_operator(_F.Var, "argmin", argmin)
__override_operator(_F.Var, "argpartition", argpartition)
__override_operator(_F.Var, "argsort", argsort)
__override_operator(_F.Var, "astype", _F.cast)
__override_operator(_F.Var, "clip", clip)
__override_operator(_F.Var, "copy", copy)
__override_operator(_F.Var, "dot", dot)
def fill(self, fill_value): self.replace(full_like(self, fill_value))
__override_operator(_F.Var, "fill", fill)
__override_operator(_F.Var, "flatten", ravel)
__override_operator(_F.Var, "max", max)
__override_operator(_F.Var, "mean", mean)
__override_operator(_F.Var, "min", min)
__override_operator(_F.Var, "nonzero", nonzero)
__override_operator(_F.Var, "prod", prod)
__override_operator(_F.Var, "ptp", ptp)
__override_operator(_F.Var, "ravel", ravel)
__override_operator(_F.Var, "repeat", repeat)
__override_operator(_F.Var, "reshape", reshape)
__override_operator(_F.Var, "round", around)
__override_operator(_F.Var, "searchsorted", searchsorted)
__override_operator(_F.Var, "sort", sort)
__override_operator(_F.Var, "squeeze", squeeze)
__override_operator(_F.Var, "std", std)
__override_operator(_F.Var, "sum", sum)
__override_operator(_F.Var, "swapaxes", swapaxes)
__override_operator(_F.Var, "transpose", transpose)
__override_operator(_F.Var, "var", var)

from . import random
from . import linalg