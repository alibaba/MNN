_Int = int
_Float = float
_Range = range
_Sequence = (list, tuple, bytes, str)

from _mnncengine._expr import *
import _mnncengine._expr as _F

_numpy_supported = False
try:
    import numpy as np
    _numpy_supported = (type(np.arange(10)) == np.ndarray)
except Exception:
    print ("Numpy not found. Using MNN without numpy.")

def scalar(value, dtype=None):
    if dtype is not None:
        if dtype == _F.int or dtype == _F.uint8:
            value = _Int(value)
        elif dtype == _F.float:
            value = _Float(value)
        return _F.const([value], [], _F.NCHW, dtype)
    if type(value) == type(1):
        return _F.const([value], [], _F.NCHW, _F.int)
    elif type(value) == type(1.):
        return _F.const([value], [], _F.NCHW, _F.float)
    else:
        raise NotImplementedError("not supported data type for creating scalar variable")
def _list_shape_type(object, shape=()):
    if isinstance(object, _Sequence) and len(object) == 0:
        return [0], _F.float
    if not isinstance(object, _Sequence):
        if type(object) in (type(1), type(1<<64)):
            dst_type = _F.int
        else:
            dst_type = _F.float
        return list(shape), dst_type
    dst_type = _F.int
    if isinstance(object[0], _Sequence):
        l = len(object[0])
        if not all (len(item) == l for item in object):
            raise ValueError('not all lists have the same length')
    shape += (len(object), )
    # recurse
    shape, dst_type = _list_shape_type(object[0], shape)
    if isinstance(object, (bytes, str)):
        dst_type = _F.uint8
    return list(shape), dst_type
def _can_broadcast(src_shape, dst_shape):
    if len(src_shape) > len(dst_shape):
        return False
    for i in _Range(len(dst_shape) - len(src_shape)):
        src_shape.insert(i, 1)
    for i in _Range(len(dst_shape)):
        if dst_shape[i] % src_shape[i] != 0:
            return False
    return True
def _valid_dtype(x):
    if x.dtype == _F.uint8:
        return _F.int
    return x.dtype
def _match_dtype(x, y, dtype=None):
    def type_val(x):
        if x is None: return -1
        if x == _F.double: return 4
        if x == _F.float: return 3
        if x == _F.int64: return 2
        if x == _F.int: return 1
        if x == _F.uint8: return 0
    xtype = x.dtype
    ytype = y.dtype
    max_type = xtype if type_val(xtype) > type_val(ytype) else ytype
    if dtype is not None:
        x = _F.cast(x, dtype)
        y = _F.cast(y, dtype)
        return x, y
    if xtype != ytype:
        x = _F.cast(x, max_type)
        y = _F.cast(y, max_type)
    return x, y
def _to_var(x, dtype=None):
    # 1. scalar
    if isinstance(x, (_Int, _Float)):
        return scalar(x, dtype)
    # 2. numpy
    if _numpy_supported:
        try:
            if isinstance(x, np.ndarray): # convert numpy ndarray to MNN var
                if x.dtype.kind == 'i':
                    x = x.astype(np.int32)
                    x = _F.const(x, x.shape, dtype=_F.int)
                elif x.dtype.kind == 'f':
                    x = x.astype(np.float32)
                    x = _F.const(x, x.shape, dtype=_F.float)
                else:
                    raise ValueError('Just support i/f dtype numpy.')
        except:
            pass
    # 3. Sequence
    if isinstance(x, _Sequence) and x:
        dst_shape, item_type = _list_shape_type(x)
        x = _F.const(x, dst_shape, dtype=item_type)
    # 4. asssert
    if not isinstance(x, _F.Var):
        raise RuntimeError("parameter `x` must be var_like.")
    # 5. convert
    if dtype is not None and x.dtype != dtype:
        x = _F.cast(x, dtype)
    return x
def _to_axis(axis, shape=None):
    if type(axis) not in (tuple, list):
        try:
            axis = [_Int(axis)]
        except TypeError:
            raise ValueError('parameter `axis` must be axis_like.')
    if shape is not None and len(axis) != shape:
        raise ValueError('parameter `axis` shape (%d,) not match require shape (%d,)'%(len(axis), shape))
    return axis
def _to_int(x):
    try:
        x = _Int(x)
    except:
        raise ValueError('parameter `x` must be int_like.')
    return x
def _to_float(x):
    try:
        x = _Float(x)
    except:
        raise ValueError('parameter `x` must be float_like.')
    return x
def _to_c4(x):
    if x.data_format != _F.NC4HW4:
        raise ValueError('parameter `x` data_format must be NC4HW4.')
    return x
# wrapper for builtin functions start
def sign(x):
    '''
    sign(x)
    Returns an element-wise indication of the sign of a number.
    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The sign of `x`.

    Example:
    -------
    >>> expr.sign([-5., 4.5])
    var([-1., 1.])
    '''
    x = _to_var(x)
    return _F.sign(x)
def abs(x):
    '''
    abs(x)
    Calculate the absolute value element-wise.
    alias name: `absolute`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The abs of `x`.

    Example:
    -------
    >>> expr.abs([-5., 4.5])
    var([5., 4.5])
    '''
    x = _to_var(x)
    return _F.abs(x)
def negative(x):
    '''
    negative(x)
    Numerical negative, element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The `-x` of `x`.

    Example:
    -------
    >>> expr.negative([-5., 4.5])
    var([5., -4.5])
    '''
    x = _to_var(x)
    return _F.negative(x)
def floor(x):
    '''
    floor(x)
    Return the floor of the input, element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The floor of `x`.

    Example:
    -------
    >>> expr.floor([-5.1, 4.5])
    var([-6.,  4.])
    '''
    x = _to_var(x, _F.float)
    return _F.floor(x)
def round(x):
    '''
    round(x)
    Return the round of the input, element-wise.
    alias name: `around`, `round_`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The round of `x`.

    Example:
    -------
    >>> expr.round([-5.1, 4.5])
    var([-5.,  5.])
    '''
    x = _to_var(x, _F.float)
    return _F.round(x)
def ceil(x):
    '''
    ceil(x)
    Return the ceil of the input, element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The ceil of `x`.

    Example:
    -------
    >>> expr.ceil([-4.9, 4.5])
    var([-4.,  5.])
    '''
    x = _to_var(x, _F.float)
    return _F.ceil(x)
def square(x):
    '''
    square(x)
    Return the x*x, element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The `x*x` of `x`.

    Example:
    -------
    >>> expr.square([-5., 4.5])
    var([25., 20.25])
    '''
    x = _to_var(x, _F.float)
    return _F.square(x)
def sqrt(x):
    '''
    sqrt(x)
    Return the sqrt(x), element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The sqrt of `x`.

    Example:
    -------
    >>> expr.sqrt([9., 4.5])
    var([3., 2.1213202])
    '''
    x = _to_var(x, _F.float)
    return _F.sqrt(x)
def rsqrt(x):
    '''
    rsqrt(x)
    Return the rsqrt(x), element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The rsqrt of `x`.

    Example:
    -------
    >>> expr.rsqrt([9., 4.5])
    var([0.33333334, 0.47140455])
    '''
    x = _to_var(x, _F.float)
    return _F.rsqrt(x)
def exp(x):
    '''
    exp(x)
    Return the exp(x), element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The exp of `x`.

    Example:
    -------
    >>> expr.exp([9., 4.5])
    var([8102.449, 90.01698])
    '''
    x = _to_var(x, _F.float)
    return _F.exp(x)
def log(x):
    '''
    log(x)
    Return the log(x), element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The log of `x`.

    Example:
    -------
    >>> expr.log([9., 4.5])
    var([2.1972246, 1.5040774])
    '''
    x = _to_var(x, _F.float)
    return _F.log(x)
def sin(x):
    '''
    sin(x)
    Return the sin(x), element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The sin of `x`.

    Example:
    -------
    >>> expr.sin([9., 4.5])
    var([0.4121185, -0.9775301])
    '''
    x = _to_var(x, _F.float)
    return _F.sin(x)
def sinh(x):
    '''
    sinh(x)
    Return the sinh(x), element-wise.
    Equivalent to ``1/2 * (np.exp(x) - np.exp(-x))`` or ``-1j * np.sin(1j*x)``.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The sinh of `x`.

    Example:
    -------
    >>> expr.sinh([9., 4.5])
    var([4051.542, 45.00301])
    '''
    x = _to_var(x, _F.float)
    return _F.sinh(x)
def cos(x):
    '''
    cos(x)
    Return the cos(x), element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The cos of `x`.

    Example:
    -------
    >>> expr.cos([9., 4.5])
    var([-0.91113025, -0.2107958])
    '''
    x = _to_var(x, _F.float)
    return _F.cos(x)
def cosh(x):
    '''
    cosh(x)
    Return the cosh(x), element-wise.
    Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The cosh of `x`.

    Example:
    -------
    >>> expr.cosh([9., 4.5])
    var([4051.542, 45.014122])
    '''
    x = _to_var(x, _F.float)
    return _F.cosh(x)
def tan(x):
    '''
    tan(x)
    Return the tan(x), element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The tan of `x`.

    Example:
    -------
    >>> expr.tan([9., 4.5])
    var([-0.45231566, 4.637332])
    '''
    x = _to_var(x, _F.float)
    return _F.tan(x)
def tanh(x):
    '''
    tanh(x)
    Return the tanh(x), element-wise.
    Equivalent to ``np.sinh(x)/np.cosh(x)`` or ``-1j * np.tan(1j*x)``.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The tanh of `x`.

    Example:
    -------
    >>> expr.tanh([9., 4.5])
    var([1., 0.9997533])
    '''
    x = _to_var(x, _F.float)
    return _F.tanh(x)
def asin(x):
    '''
    asin(x)
    Return the arcsin(x), element-wise.
    alias name: `arcsin`.

    Parameters
    ----------
    x : var_like, input value, available range is [-1, 1].

    Returns
    -------
    y : Var. The arcsin of `x`.

    Example:
    -------
    >>> expr.asin([9., 0.5])
    var([nan, 0.5235988])
    '''
    x = _to_var(x, _F.float)
    return _F.asin(x)
def asinh(x):
    '''
    asinh(x)
    Return the arcsinh(x), element-wise.
    alias name: `arcsinh`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The arcsinh of `x`.

    Example:
    -------
    >>> expr.asinh([9., 0.5])
    var([2.893444, 0.4812118])
    '''
    x = _to_var(x, _F.float)
    return _F.asinh(x)
def acos(x):
    '''
    acos(x)
    Return the arccos(x), element-wise.
    alias name: `arccos`.

    Parameters
    ----------
    x : var_like, input value, available range is [-1, 1].

    Returns
    -------
    y : Var. The arccos of `x`.

    Example:
    -------
    >>> expr.asin([9., 0.5])
    var([nan, 1.0471975])
    '''
    x = _to_var(x, _F.float)
    return _F.acos(x)
def acosh(x):
    '''
    acosh(x)
    Return the arccosh(x), element-wise.
    alias name: `arccosh`.

    Parameters
    ----------
    x : var_like, input value, available range is [1, +inf).

    Returns
    -------
    y : Var. The arccosh of `x`.

    Example:
    -------
    >>> expr.acosh([9., 0.5])
    var([2.887271, nan])
    '''
    x = _to_var(x, _F.float)
    return _F.acosh(x)
def atan(x):
    '''
    atan(x)
    Return the arctan(x), element-wise.
    alias name: `arctan`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The arctan of `x`.

    Example:
    -------
    >>> expr.atan([9., 0.5])
    var([1.4601392, 0.4636476])
    '''
    x = _to_var(x, _F.float)
    return _F.atan(x)
def atanh(x):
    '''
    atanh(x)
    Return the arctanh(x), element-wise.
    alias name: `arctanh`.

    Parameters
    ----------
    x : var_like, input value, available range is [-1, 1].

    Returns
    -------
    y : Var. The arctanh of `x`.

    Example:
    -------
    >>> expr.atanh([9., 0.5])
    var([1.4601392, 0.4636476])
    '''
    x = _to_var(x, _F.float)
    return _F.atanh(x)
def reciprocal(x):
    '''
    reciprocal(x)
    Return the ``1/x``, element-wise.

    Parameters
    ----------
    x : var_like, input value, available range is [!=0].

    Returns
    -------
    y : Var. The ``1/x`` of `x`.

    Example:
    -------
    >>> expr.reciprocal([9., 0.5])
    var([0.11111111, 2.])
    '''
    x = _to_var(x, _F.float)
    return _F.reciprocal(x)
def log1p(x):
    '''
    log1p(x)
    Return the ``log(1 + x)``, element-wise.

    Parameters
    ----------
    x : var_like, input value, available range is (-1, +inf).

    Returns
    -------
    y : Var. The ``log(1 + x)`` of `x`.

    Example:
    -------
    >>> expr.log1p([9., 0.5])
    var([2.3025851, 0.4054651])
    '''
    x = _to_var(x, _F.float)
    return _F.log1p(x)
def gelu(x):
    '''
    gelu(x)
    Return the ``0.5x(1+tanh(sqrt(pi/2)*(0.044715*x^3)))``, element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The gelu of `x`.

    Example:
    -------
    >>> expr.gelu([9., 0.5])
    var([9., 0.345714])
    '''
    x = _to_var(x, _F.float)
    return _F.gelu(x)
def sigmoid(x):
    '''
    sigmoid(x)
    Return the ``1/(1+exp(-1))``, element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The sigmoid of `x`.

    Example:
    -------
    >>> expr.sigmoid([9., 0.5])
    var([0.9998766, 0.62246716])
    '''
    x = _to_var(x, _F.float)
    return _F.sigmoid(x)
def erf(x):
    x = _to_var(x, _F.float)
    return _F.erf(x)
def erfc(x):
    x = _to_var(x, _F.float)
    return _F.erfc(x)
def erfinv(x):
    x = _to_var(x, _F.float)
    return _F.erfinv(x)
def expm1(x):
    '''
    expm1(x)
    Return the ``exp(x) - 1``, element-wise.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    y : Var. The expm1 of `x`.

    Example:
    -------
    >>> expr.expm1([9., 0.5])
    var([8.1014492e+03, 6.4869785e-01])
    '''
    x = _to_var(x, _F.float)
    return _F.expm1(x)
def add(x, y):
    '''
    add(x, y)
    Return the ``x + y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x + y`` of `x` and `y`.

    Example:
    -------
    >>> expr.add([9., 0.5], [1.2, -3.0])
    var([10.2, -2.5])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.add(x, y)
def subtract(x, y):
    '''
    subtract(x, y)
    Return the ``x - y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x - y`` of `x` and `y`.

    Example:
    -------
    >>> expr.subtract([9., 0.5], [1.2, -3.0])
    var([7.8, 3.5])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.subtract(x, y)
def multiply(x, y):
    '''
    multiply(x, y)
    Return the ``x * y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x * y`` of `x` and `y`.

    Example:
    -------
    >>> expr.multiply([9., 0.5], [1.2, -3.0])
    var([10.8, -1.5])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.multiply(x, y)
def divide(x, y):
    '''
    divide(x, y)
    Return the ``x / y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x / y`` of `x` and `y`.

    Example:
    -------
    >>> expr.divide([9., 0.5], [1.2, -3.0])
    var([7.4999995, -0.16666667])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.divide(x, y)
def floordiv(x, y):
    '''
    floordiv(x, y)
    Return the ``x // y``, element-wise.
    alias name: `floor_divide`.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x // y`` of `x` and `y`.

    Example:
    -------
    >>> expr.floordiv([9., 0.5], [1.2, -3.0])
    var([7., -1.])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.floordiv(x, y)
def mod(x, y):
    '''
    mod(x, y)
    Return the ``x % y``, element-wise.
    alias name: `fmod`, `remainder`.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x % y`` of `x` and `y`.

    Example:
    -------
    >>> expr.mod([9., 0.5], [1.2, -3.0])
    var([0.59999967, 0.5])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.mod(x, y)
def floormod(x, y):
    '''
    floormod(x, y)
    Return the floormod, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The floormod of `x` and `y`.

    Example:
    -------
    >>> expr.floormod([9., 0.5], [1.2, -3.0])
    var([0.5999994, -2.5])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.floormod(x, y)
def pow(x, y):
    '''
    pow(x, y)
    Return the ``x ** y``, element-wise.
    alias name: `power`.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x ** y`` of `x` and `y`.

    Example:
    -------
    >>> expr.pow([9., 0.5], [1.2, -3.0])
    var([13.966612, 8.])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.pow(x, y)
def minimum(x, y):
    '''
    minimum(x, y)
    Return the minimum value, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The minimum value of `x` and `y`.

    Example:
    -------
    >>> expr.minimum([9., 0.5], [1.2, -3.0])
    var([1.2, -3.])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.minimum(x, y)
def maximum(x, y):
    '''
    maximum(x, y)
    Return the maximum value, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The maximum value of `x` and `y`.

    Example:
    -------
    >>> expr.maximum([9., 0.5], [1.2, -3.0])
    var([9., 0.5])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.maximum(x, y)
def equal(x, y):
    '''
    equal(x, y)
    Return the ``x == y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x == y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.equal([-9., 0.5], [1.2, 0.5])
    var([0, 1])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.equal(x, y)
def not_equal(x, y):
    '''
    not_equal(x, y)
    Return the ``x != y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x != y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.not_equal([-9., 0.5], [1.2, 0.5])
    var([1, 0])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.not_equal(x, y)
def greater(x, y):
    '''
    greater(x, y)
    Return the ``x > y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x > y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.greater([-9., 0.5], [1.2, -3.0])
    var([0, 1])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.greater(x, y)
def greater_equal(x, y):
    '''
    greater_equal(x, y)
    Return the ``x >= y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x >= y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.greater_equal([-9., 0.5], [1.2, -3.0])
    var([0, 1])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.greater_equal(x, y)
def less(x, y):
    '''
    less(x, y)
    Return the ``x < y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x < y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.less([-9., 0.5], [1.2, -3.0])
    var([1, 0])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.less(x, y)
def less_equal(x, y):
    '''
    less_equal(x, y)
    Return the ``x <= y``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``x <= y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.less_equal([-9., 0.5], [1.2, -3.0])
    var([1, 0])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.less_equal(x, y)
def squared_difference(x, y):
    '''
    squared_difference(x, y)
    Return the ``(x - y)^2``, element-wise.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``(x - y)^2`` of `x` and `y`.

    Example:
    -------
    >>> expr.squared_difference([-9., 0.5], [1.2, -3.0])
    var([104.03999, 12.25])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.squared_difference(x, y)
def atan2(x, y):
    '''
    atan2(x, y)
    Return the ``arctan(x / y)``, element-wise.
    alias name: `arctan2`.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    z : Var. The ``arctan(x / y)`` of `x` and `y`.

    Example:
    -------
    >>> expr.atan2([-9., 0.5], [1.2, -3.0])
    var([-1.4382448, -0.16514869])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    return _F.atan2(x, y)
def logical_or(x, y):
    '''
    logical_or(x, y)
    The dtype of x, y must be same.

    Parameters
    ----------
    x : var_like, input value, dtype just support int32.
    y : var_like, input value, dtype just support int32.

    Returns
    -------
    z : Var. The ``x or y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.logical_or([2, 1], [4, 2])
    var([1, 1])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    if x.dtype != _F.int or y.dtype != _F.int:
        raise ValueError('MNN.expr.logical_or just support int32')
    return _F.logical_or(x, y)
def bias_add(value, bias):
    '''
    bias_add(value, bias)

    Adds bias to value.
    This is (mostly) a special case of add where bias is restricted to 1-D.
    Broadcasting is supported, so value may have any number of dimensions.
    Unlike add, the type of bias is allowed to differ from value in the case where both types are quantized.

    Parameters
    ----------
    value: A variable with type dtype.float or dtype.int.
    bias: A 1-D variable with size matching the channel dimension of value.
          Must be the same type as value unless value is a quantized type, in which case a different quantized type may be used.

    Returns
    ----------
    A variable with the same type as value.
    
    Example
    ----------
    >>> expr.bias_add(np.eye(3,3), np.ones(3))
    var([[2., 1., 1.],
         [1., 2., 1.],
         [1., 1., 2.]], dtype=float32)
    '''
    value = _to_var(value)
    bias = _to_var(bias)
    value, bias = _match_dtype(value, bias)
    if len(bias.shape) != 1:
        raise RuntimeError("parameter bias must be 1-D in bias_add")
    if value.shape[-1] != bias.shape[-1]:
        raise RuntimeError("parameter bias's dim must match parameter value's dim in bias_add")
    return _F.bias_add(value, bias)
def bitwise_and(x, y):
    '''
    bitwise_and(x, y)
    The dtype of x, y must be same and be int32.

    Parameters
    ----------
    x : var_like, input value, dtype just support int32.
    y : var_like, input value, dtype just support int32.

    Returns
    -------
    z : Var. The ``x & y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.bitwise_and([1, 2], [3, 4])
    var([1, 0])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    if x.dtype != _F.int or y.dtype != _F.int:
        raise ValueError('MNN.expr.bitwise_and just support int32')
    return _F.bitwise_and(x, y)
def bitwise_or(x, y):
    '''
    bitwise_or(x, y)
    The dtype of x, y must be same and be int32.

    Parameters
    ----------
    x : var_like, input value, dtype just support int32.
    y : var_like, input value, dtype just support int32.

    Returns
    -------
    z : Var. The ``x | y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.bitwise_or([1, 2], [3, 4])
    var([3, 6])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    if x.dtype != _F.int or y.dtype != _F.int:
        raise ValueError('MNN.expr.bitwise_or just support int32')
    return _F.bitwise_or(x, y)
def bitwise_xor(x, y):
    '''
    bitwise_xor(x, y)
    The dtype of x, y must be same and be int32.

    Parameters
    ----------
    x : var_like, input value, dtype just support int32.
    y : var_like, input value, dtype just support int32.

    Returns
    -------
    z : Var. The ``x ^ y`` of `x` and `y`, dtype is int32.

    Example:
    -------
    >>> expr.bitwise_xor([1, 2], [3, 4])
    var([2, 6])
    '''
    x = _to_var(x)
    y = _to_var(y)
    x, y = _match_dtype(x, y)
    if x.dtype != _F.int or y.dtype != _F.int:
        raise ValueError('MNN.expr.bitwise_xor just support int32')
    return _F.bitwise_xor(x, y)
def reduce_sum(x, axis=[], keepdims=False):
    '''
    reduce_sum(x, axis=[], keepdims=False)
    Return the sum of all/axis.

    Parameters
    ----------
    x : var_like, input value.
    axis : axis_like, input value, just support int32. Default is [], reduce all.
    keepdims: bool, input value. Default is False.

    Returns
    -------
    z : Var. The sum of `x` along the `axis`.

    Example:
    -------
    >>> expr.reduce_sum([[1.,2.],[3.,4.]])
    var(10.)
    >>> expr.reduce_sum([[1.,2.],[3.,4.]], 0)
    var([4., 6.])
    '''
    x = _to_var(x, _valid_dtype(x))
    axis = _to_axis(axis)
    return _F.reduce_sum(x, axis, keepdims)
def reduce_mean(x, axis=[], keepdims=False):
    '''
    reduce_mean(x, axis=[], keepdims=False)
    Return the mean of all/axis.

    Parameters
    ----------
    x : var_like, input value.
    axis : axis_like, input value, just support int32. Default is [], reduce all.
    keepdims: bool, input value. Default is False.

    Returns
    -------
    z : Var. The mean of `x` along the `axis`.

    Example:
    -------
    >>> expr.reduce_mean([[1.,2.],[3.,4.]])
    var(2.5.)
    >>> expr.reduce_mean([[1.,2.],[3.,4.]], 0)
    var([2., 3.])
    '''
    x = _to_var(x, _valid_dtype(x))
    axis = _to_axis(axis)
    return _F.reduce_mean(x, axis, keepdims)
def reduce_max(x, axis=[], keepdims=False):
    '''
    reduce_max(x, axis=[], keepdims=False)
    Return the max of all/axis.

    Parameters
    ----------
    x : var_like, input value.
    axis : axis_like, input value, just support int32. Default is [], reduce all.
    keepdims: bool, input value. Default is False.

    Returns
    -------
    z : Var. The max of `x` along the `axis`.

    Example:
    -------
    >>> expr.reduce_max([[1.,2.],[3.,4.]])
    var(4.)
    >>> expr.reduce_max([[1.,2.],[3.,4.]], 0)
    var([3., 4.])
    '''
    x = _to_var(x, _valid_dtype(x))
    axis = _to_axis(axis)
    return _F.reduce_max(x, axis, keepdims)
def reduce_min(x, axis=[], keepdims=False):
    '''
    reduce_min(x, axis=[], keepdims=False)
    Return the min of all/axis.

    Parameters
    ----------
    x : var_like, input value.
    axis : axis_like, input value, just support int32. Default is [], reduce all.
    keepdims: bool, input value. Default is False.

    Returns
    -------
    z : Var. The min of `x` along the `axis`.

    Example:
    -------
    >>> expr.reduce_min([[1.,2.],[3.,4.]])
    var(1.)
    >>> expr.reduce_min([[1.,2.],[3.,4.]], 0)
    var([1., 2.])
    '''
    x = _to_var(x, _valid_dtype(x))
    axis = _to_axis(axis)
    return _F.reduce_min(x, axis, keepdims)
def reduce_prod(x, axis=[], keepdims=False):
    '''
    reduce_prod(x, axis=[], keepdims=False)
    Return the prod of all/axis.

    Parameters
    ----------
    x : var_like, input value.
    axis : axis_like, input value, just support int32. Default is [], reduce all.
    keepdims: bool, input value. Default is False.

    Returns
    -------
    z : Var. The prod of `x` along the `axis`.

    Example:
    -------
    >>> expr.reduce_prod([[1.,2.],[3.,4.]])
    var(24.)
    >>> expr.reduce_prod([[1.,2.],[3.,4.]], 0)
    var([3., 8.])
    '''
    x = _to_var(x, _valid_dtype(x))
    axis = _to_axis(axis)
    return _F.reduce_prod(x, axis, keepdims)
def reduce_any(x, axis=[], keepdims=False):
    '''
    reduce_any(x, axis=[], keepdims=False)
    Return the any of nonzero all/axis.

    Parameters
    ----------
    x : var_like, input value, dtype just support int32.
    axis : axis_like, input value, just support int32. Default is [], reduce all.
    keepdims: bool, input value. Default is False.

    Returns
    -------
    z : Var. The any of `x` along the `axis`.

    Example:
    -------
    >>> expr.reduce_any([[0,1],[0,3]])
    var(1)
    >>> expr.reduce_any([[0,1],[0,3]], 1)
    var([0, 1])
    '''
    x = _to_var(x)
    if x.dtype != _F.int:
        raise ValueError('MNN.expr.reduce_any just support int32')
    axis = _to_axis(axis)
    return _F.reduce_any(x, axis, keepdims)
def reduce_all(x, axis=[], keepdims=False):
    '''
    reduce_all(x, axis=[], keepdims=False)
    Return the all of nonzero of all/axis.

    Parameters
    ----------
    x : var_like, input value, dtype just support int32.
    axis : axis_like, input value, just support int32. Default is [], reduce all.
    keepdims: bool, input value. Default is False.

    Returns
    -------
    z : Var. The all of `x` along the `axis`.

    Example:
    -------
    >>> expr.reduce_all([[0,1],[0,3]])
    var(0)
    >>> expr.reduce_all([[0,1],[0,3]], 0)
    var([0, 1])
    '''
    x = _to_var(x)
    if x.dtype != _F.int:
        raise ValueError('MNN.expr.reduce_all just support int32')
    axis = _to_axis(axis)
    return _F.reduce_all(x, axis, keepdims)
# eltwise op skip
def cast(x, dtype=_F.float):
    '''
    cast(x, dtype=float)
    Return the dtype of x.

    Parameters
    ----------
    x : var_like, input value.
    dtype : dtype. Default is float.

    Returns
    -------
    z : Var. The dtype of `x`.

    Example:
    -------
    >>> expr.cast([[0,1],[0,3]], float)
    var([[0., 1.],
         [0., 3.]], dtype=float32)
    '''
    x = _to_var(x)
    return _F.cast(x, dtype)
def matmul(a, b, transposeA=False, transposeB=False):
    '''
    matmul(a, b, transposeA=False, transposeB=False)
    Return the ``a @ b``.

    Parameters
    ----------
    a : var_like, input value.
    b : var_like, input value.
    transposeA: bool, a is transpose or not.
    transposeB: bool, b is transpose or not.

    Returns
    -------
    c : Var. The result of ``x @ y``, dtype is float32.

    Example:
    -------
    >>> expr.matmul([[1,2],[3,4]], [[1,1],[2,2]])
    var([[0., 1.],
         [0., 3.]], dtype=float32)
    '''
    a = _to_var(a, _F.float)
    b = _to_var(b, _F.float)
    return _F.matmul(a, b, transposeA, transposeB)
def normalize(x, acrossSpatial, channelShared, eps, scale):
    '''
    normalize(x, acrossSpatial, channelShared, eps, scale)
    Return the normalize of x.

    Parameters
    ----------
    x : var_like, input value.
    acrossSpatial : int, input value.
    channelShared : int, input value.
    eps : float, input value.
    scale : [float], input value.

    Returns
    -------
    y : Var. The normalize of x.

    Example:
    -------
    >>> x = expr.const([-1.0, -2.0, 3.0, 4.0], [1, 2, 2, 1], expr.NCHW)
    >>> expr.normalize(x, 0, 0, 0.0, [0.5, 0.5])
    var([[[[-0.2236068],
           [-0.4472136]],
          [[ 0.3      ],
           [ 0.4      ]]]], dtype=float32)
    '''
    x = _to_var(x)
    return _F.normalize(x, acrossSpatial, channelShared, eps, scale)
def argmax(x, axis=0):
    '''
    argmax(x, axis=0)
    Return argmax of `x` along the `axis`.

    Parameters
    ----------
    x : var_like, input value.
    axis : int, input value. Default is 0.

    Returns
    -------
    c : Var. The argmax of `x` along the `axis`, dtype is int32.

    Example:
    -------
    >>> expr.argmax([[1,2],[3,4]])
    var([1, 1], dtype=int32)
    '''
    x = _to_var(x)
    axis = _to_int(axis)
    return _F.argmax(x, axis)
def argmin(x, axis=0):
    '''
    argmin(x, axis=0)
    Return argmin of `x` along the `axis`.

    Parameters
    ----------
    x : var_like, input value.
    axis : int, input value. Default is 0.

    Returns
    -------
    c : Var. The argmin of `x` along the `axis`, dtype is int32.

    Example:
    -------
    >>> expr.argmin([[1,2],[3,4]])
    var([0, 0], dtype=int32)
    '''
    x = _to_var(x)
    axis = _to_int(axis)
    return _F.argmin(x, axis)
def cumsum(x, axis=0):
    '''
    cumsum(x, axis=0)
    Return cumsum of `x` along the `axis`.

    Parameters
    ----------
    x : var_like, input value.
    axis : int, input value. Default is 0.

    Returns
    -------
    c : Var. The cumsum of `x` along the `axis`.

    Example:
    -------
    >>> expr.cumsum([[1,2],[3,4]])
    var([[1, 2],
         [4, 6]], dtype=int32)
    '''
    x = _to_var(x)
    axis = _to_int(axis)
    return _F.cumsum(x, axis)
def cumprod(x, axis=0):
    '''
    cumprod(x, axis=0)
    Return cumprod of `x` along the `axis`.

    Parameters
    ----------
    x : var_like, input value, dtype is float32.
    axis : int, input value. Default is 0.

    Returns
    -------
    c : Var. The cumprod of `x` along the `axis`.

    Example:
    -------
    >>> expr.cumprod([[1.,2.],[3.,4.]])
    var([[1., 2.],
         [3., 8.]], dtype=float32)
    '''
    x = _to_var(x, dtype=_F.float)
    axis = _to_int(axis)
    return _F.cumprod(x, axis)
def svd(x):
    '''
    svd(x)
    Return svd matrixs of `x`, `x` is a 2D Matrix.
    Return `w`, `u`, `vt` as ``a = u @ w @ vt``.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    w : Var, shape is (N).
    u : Var, shape is (M, N).
    vt : Var, shape is (N, N).

    Example:
    -------
    >>> expr.cumprod([[1.,2.],[3.,4.]])
    [var([5.464986  , 0.36596605], dtype=float32), var([[ 0.40455356,  0.91451436],
       [ 0.91451424, -0.40455365]], dtype=float32), var([[ 0.5760485 ,  0.81741554],
       [-0.81741554,  0.5760485 ]], dtype=float32)]
    '''
    x = _to_var(x, dtype=_F.float)
    return _F.svd(x)
def unravel_index(indices, shape):
    '''
    unravel_index(indices, shape)
    Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Parameters
    ----------
    indices : var_like, input value.
    shape : var_like, input value.

    Returns
    -------
    unraveled_coords : Var, dtype is int32.

    Example:
    -------
    >>> expr.unravel_index([22, 41, 37], (7,6))
    var([[3, 6, 6],
         [4, 5, 1]], dtype=int32)
    '''
    indices = _to_var(indices)
    shape = _to_var(shape)
    return _F.unravel_index(indices, shape)
def scatter_nd(indices, updates, shape):
    '''
    scatter_nd(indices, updates, shape)
    scatter value from updates by indices.

    Parameters
    ----------
    indices : var_like, input value.
    updates : var_like, input value.
    shape : var_like, input value.

    Returns
    -------
    scatter_val : Var.

    Example:
    -------
    >>> indices = expr.const([4, 3, 1, 7], [4, 1], expr.NHWC, expr.int)
    >>> expr.scatter_nd(indices, [9.0, 10.0, 11.0, 12.0], [8])
    var([ 0., 11.,  0., 10.,  9.,  0.,  0., 12.], dtype=float32)
    '''
    indices = _to_var(indices)
    updates = _to_var(updates)
    shape = _to_var(shape)
    return _F.scatter_nd(indices, updates, shape)
def one_hot(indices, depth, onValue=1.0, offValue=0.0, axis=-1):
    '''
    one_hot(indices, depth, onValue=1.0, offValue=0.0, axis=-1)
    get the one_hot value of indices.

    Parameters
    ----------
    indices : var_like, input value.
    depth : int, input value.
    onValue : float, input value. Default is 1.0.
    offValue : float, input value. Default is 0.0.
    axis : int, input value. Default is -1.

    Returns
    -------
    onehot_val : Var.

    Example:
    -------
    >>> expr.one_hot([0, 1, 2], 3)
    var([[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]], dtype=float32)
    '''
    indices = _to_var(indices)
    return _F.one_hot(indices, depth, onValue, offValue, axis)
def broadcast_to(input, shape):
    '''
    broadcast_to(input, shape)
    broadcast `input` to `shape`.

    Parameters
    ----------
    input : var_like, input value.
    shape : var_like, input value, dtype is int32.

    Returns
    -------
    broadcast_val : Var.

    Example:
    -------
    >>> expr.broadcast_to([1,2], 4)
    var([1, 2, 1571999095, -536868871], dtype=int32)
    '''
    input = _to_var(input)
    shape = _to_axis(shape)
    if not _can_broadcast(input.shape, shape):
        raise ValueError('MNN.expr.broadcast_to can\'t broadcast from', input.shape, ' to ', shape, '.')
    shape = _to_var(shape)
    if shape.dtype != _F.int:
        raise ValueError('MNN.expr.broadcast_to shape dtype just support int32.')
    return _F.broadcast_to(input, shape)
def placeholder(shape=[], format=_F.NCHW, dtype=_F.float):
    '''
    placeholder(shape=[], format=_F.NCHW, dtype=_F.float)
    build a placeholder var.

    Parameters
    ----------
    shape : [int], input value. Default is [].
    format : data_format, input value. Default is NCHW.
    dtype : dtype, input value. Default is float.

    Returns
    -------
    placeholder : Var. Before use this, need `write`. 

    Example:
    -------
    >>> expr.placeholder()
    var(1., dtype=float32)
    >>> expr.placeholder([2,2])
    var([[1.67e-43, 1.60e-43],
         [1.36e-43, 1.54e-43]], dtype=float32)
    '''
    return _F.placeholder(shape, format, dtype)
def clone(x, deepCopy=False):
    '''
    clone(x, deepCopy=False)
    clone a var.

    Parameters
    ----------
    x : var_like, input value.
    deepCopy : bool, input value. Default is False.

    Returns
    -------
    y : Var copy from x. 

    Example:
    -------
    >>> x = expr.const([1.,2.], [2])
    >>> expr.clone(x, True)
    array([1., 2.], dtype=float32)
    '''
    x = _to_var(x)
    return _F.clone(x, deepCopy)
# TODO: conv2d, conv2d_transpose, max_pool, avg_pool
def conv2d(input, weight, bias, stride=(1,1),
           padding=(0,0), dilate=(1,1), group=1, padding_mode=_F.VALID):
    '''
    conv2d(input, weight, bias, stride = (1,1),
           padding=(0,0), dilate=(1,1), group=1, padding_mode=_F.VALID)
    do convolution 2d.

    Parameters
    ----------
    input : var_like, input value, data_format is NC4HW4.
    weight : var_like, input value.
    bias : var_like, input value.
    stride : tuple of int, input value. Default is (1,1).
    padding : tuple of int, input value. Default is (0,0).
    dilate : tuple of int, input value. Default is (1,1).
    group : int, input value. Default is 1.
    padding_mode : Padding_Mode, input value. Default is VALID.

    Returns
    -------
    conv2d_res : Var, data_format is NC4HW4.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> w = expr.reshape(expr.range(0., 16., 1.), [2, 2, 2, 2])
    >>> b = expr.const([1.0, 1.0], [2])
    >>> expr.convert(expr.conv2d(x, w, b), expr.NCHW)
    var([[[[ 269.,  297.],
           [ 353.,  381.]],
          [[ 685.,  777.],
           [ 961., 1053.]]]], dtype=float32)
    '''
    input = _to_c4(_to_var(input))
    weight = _to_var(weight)
    bias = _to_var(bias)
    stride = _to_axis(stride, 2)
    padding = _to_axis(padding, 2)
    group = _to_int(group)
    return _F.conv2d(input, weight, bias, stride, padding,
                     dilate, group, padding_mode)
def conv2d_transpose(input, weight, bias, stride=(1,1), padding=(0,0),
                     dilate=(1,1), group=1, padding_mode=_F.VALID):
    '''
    conv2d_transpose(input, weight, bias, stride = (1,1), padding=(0,0),
                     dilate=(1,1), group=1, padding_mode=_F.VALID)
    do conv2d_transpose.

    Parameters
    ----------
    input : var_like, input value, data_format is NC4HW4.
    weight : var_like, input value.
    bias : var_like, input value.
    stride : tuple of int, input value. Default is (1,1).
    padding : tuple of int, input value. Default is (0,0).
    dilate : tuple of int, input value. Default is (1,1).
    group : int, input value. Default is 1.
    padding_mode : Padding_Mode, input value. Default is VALID.

    Returns
    -------
    conv2d_transpose_res : Var, data_format is NC4HW4.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> w = expr.reshape(expr.range(0., 16., 1.), [2, 2, 2, 2])
    >>> b = expr.const([1.0, 1.0], [2])
    >>> expr.convert(expr.conv2d_transpose(x, w, b), expr.NCHW)
    var([[[[ 73., 162., 180., 102.],
           [187., 417., 461., 259.],
           [247., 549., 593., 331.],
           [163., 358., 384., 212.]],
          [[109., 242., 276., 154.],
           [283., 625., 701., 387.],
           [391., 853., 929., 507.],
           [247., 534., 576., 312.]]]], dtype=float32)
    '''
    input = _to_c4(_to_var(input))
    weight = _to_var(weight)
    bias = _to_var(bias)
    stride = _to_axis(stride, 2)
    padding = _to_axis(padding, 2)
    group = _to_int(group)
    return _F.conv2d_transpose(input, weight, bias, stride, padding,
                               dilate, group, padding_mode)
def max_pool(x, kernel, stride, padding_mode=_F.VALID, pads=(0,0)):
    '''
    max_pool(x, kernel, stride, padding_mode=_F.VALID, pads=(0,0))
    do max pooling.

    Parameters
    ----------
    x : var_like, input value.
    kernel : var_like, input value.
    stride : tuple of int, input value. Default is (1,1).
    padding_mode : Padding_Mode, input value. Default is VALID.
    pads : tuple of int, input value. Default is (0,0).

    Returns
    -------
    maxpool_res : Var, data_format.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
    >>> expr.max_pool(x, [2,2], [1,1])
    var([[[[ 4.,  5.],
           [ 7.,  8.]],
          [[13., 14.],
           [16., 17.]]]], dtype=float32)
    '''
    x = _to_var(x)
    kernel = _to_axis(kernel, 2)
    stride = _to_axis(stride, 2)
    return _F.max_pool(x, kernel, stride, padding_mode, pads)
def avg_pool(x, kernel, stride, padding_mode=_F.VALID, pads=(0,0)):
    '''
    avg_pool(x, kernel, stride, padding_mode=_F.VALID, pads=(0,0))
    do avg pooling.

    Parameters
    ----------
    x : var_like, input value.
    kernel : var_like, input value.
    stride : tuple of int, input value. Default is (1,1).
    padding_mode : Padding_Mode, input value. Default is VALID.
    pads : tuple of int, input value. Default is (0,0).

    Returns
    -------
    avgpool_res : Var, data_format.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
    >>> expr.avg_pool(x, [2,2], [1,1])
    var([[[[ 2.,  3.],
           [ 5.,  6.]],
          [[11., 12.],
           [14., 15.]]]], dtype=float32)
    '''
    x = _to_var(x)
    kernel = _to_axis(kernel, 2)
    stride = _to_axis(stride, 2)
    return _F.avg_pool(x, kernel, stride, padding_mode, pads)
def reshape(x, shape, original_format=_F.NCHW):
    '''
    reshape(x, shape, original_format=NCHW)
    reshape the var `x` by `shape`.

    Parameters
    ----------
    x : var_like, input value.
    shape : axis_like, input value.
    original_format : data_format, input value. Default is NCHW.

    Returns
    -------
    reshape_res : Var whith `shape`.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
    >>> reshape(x, [3, 6])
    var([[ 0.,  1.,  2.,  3.,  4.,  5.],
         [ 6.,  7.,  8.,  9., 10., 11.],
         [12., 13., 14., 15., 16., 17.]], dtype=float32)
    '''
    x = _to_var(x)
    shape = _to_axis(shape)
    new_length = 1
    skip = False
    for value in shape:
        if value < 0:
            skip = True
        new_length *= value
    if new_length != x.size and not skip:
        raise RuntimeError("parameter shape is not valid")
    return _F.reshape(x, shape, original_format)
def scale(x, channels, scale, bias):
    '''
    scale(x, channels, scale, bias)
    get the ``x * scale + bias`` of `x`. 

    Parameters
    ----------
    x : var_like, input value, data_format is NC4HW4.
    channels : int, input value.
    scale : [float], input value.
    bias : [float], input value.

    Returns
    -------
    scale_res : Var, data_format is NC4HW4.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> y = expr.scale(x, 2, [2.0, 1.0], [3.0, 4.0])
    >>> expr.convert(y, expr.NCHW)
    var([[[[ 3.,  5.,  7.],
           [ 9., 11., 13.],
           [15., 17., 19.]],
          [[13., 14., 15.],
           [16., 17., 18.],
           [19., 20., 21.]]]], dtype=float32)
    '''
    x = _to_c4(_to_var(x))
    return _F.scale(x, channels, scale, bias)
def relu(x, slope=0.0):
    '''
    relu(x, slope=0.0)
    relu(slope*x if x < 0 else x) of `x`.

    Parameters
    ----------
    x : var_like, input value.
    slope : float, input value. Default is 0.0;

    Returns
    -------
    relu_res : Var.

    Example:
    -------
    >>> expr.relu([-1.0, 0.0, 2.0])
    var[0., 0., 2.], dtype=float32)
    '''
    x = _to_var(x)
    slope = _to_float(slope)
    return _F.relu(x, slope)
def relu6(x, min=0.0, max=6.0):
    '''
    relu6(x, min=0.0, max=6.0)
    `max(min(x, max), min)` of `x`.

    Parameters
    ----------
    x : var_like, input value.
    min : float, input value. Default is 0.0;
    max : float, input value. Default is 6.0;

    Returns
    -------
    relu6_res : Var.

    Example:
    -------
    >>> expr.relu6([-1.0, 7.0, 2.0])
    var[0., 6., 2.], dtype=float32)
    '''
    x = _to_var(x)
    min = _to_float(min)
    max = _to_float(max)
    return _F.relu6(x, min, max)
def prelu(x, slopes):
    '''
    prelu(x, slopes)
    prelu(slope*x if x < 0 else x) of `x`.

    Parameters
    ----------
    x : var_like, input value, data_format is NC4HW4.
    slopes : [float], input value.

    Returns
    -------
    prelu_res : Var, data_format is NC4HW4.

    Example:
    -------
    >>> x = expr.reshape(expr.range(-4., 4., 1.), [1, 2, 2, 2])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> expr.convert(expr.prelu(x, [0.5, 0.6]), expr.NCHW)
    var([[[[-2. , -1.5],
           [-1. , -0.5]],
          [[ 0. ,  1. ],
           [ 2. ,  3. ]]]], dtype=float32)
    '''
    x = _to_c4(_to_var(x))
    return _F.prelu(x, slopes)
def softmax(x, axis=-1):
    '''
    softmax(x, axis=-1)
    ``exp(x)/sum(exp(x), axis)`` of `x`.

    Parameters
    ----------
    x : var_like, input value.
    axis : int, input value. Default is -1.

    Returns
    -------
    softmax_res : Var.

    Example:
    -------
    >>> expr.softmax([[1., 2.], [3., 4.]], 0)
    var([[0.1191897 , 0.1191897 ],
         [0.88081026, 0.88081026]], dtype=float32)
    '''
    x = _to_var(x)
    axis = _to_int(axis)
    return _F.softmax(x, axis)
def softplus(x):
    '''
    softplus(x)
    ``log(exp(x) + 1)`` of `x`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    softplus_res : Var.

    Example:
    -------
    >>> expr.softplus([[1., 2.], [3., 4.]])
    var([[1.313261 , 2.1268892],
         [3.048587 , 4.01813  ]], dtype=float32)
    '''
    features = _to_var(x)
    return _F.softplus(x)
def softsign(x):
    '''
    softsign(x)
    ``x / (abs(x) + 1)`` of `x`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    softsign_res : Var.

    Example:
    -------
    >>> expr.softsign([[1., 2.], [3., 4.]])
    var([[0.5      , 0.6666667],
         [0.75     , 0.8      ]], dtype=float32)
    '''
    x = _to_var(x)
    return _F.softsign(x)
def slice(x, starts, sizes):
    '''
    slice(x, starts, sizes)
    ``x[starts[0]:starts[0]+sizes[0], ..., starts[-1]:starts[-1]+sizes[-1]]`` of `x`.

    Parameters
    ----------
    x : var_like, input value.
    starts : var_like, input value, dtype is int.
    sizes : var_like, input value, dtype is int.

    Returns
    -------
    slice_res : Var.

    Example:
    -------
    >>> expr.slice([[1., 2., 3.], [4., 5., 6.]], [0, 1], [1, 2])
    var([[2., 3.]], dtype=float32)
    '''
    x = _to_var(x)
    starts = _to_var(starts)
    sizes = _to_var(sizes)
    if starts.dtype != _F.int or sizes.dtype != _F.int:
        raise RuntimeError("parameter starts/sizes must be int type")
    return _F.slice(x, starts, sizes)
def split(x, size_splits, axis):
    '''
    split(x, size_splits, axis)
    ``x[starts[0]:starts[0]+sizes[0], ..., starts[-1]:starts[-1]+sizes[-1]]`` of `x`.

    Parameters
    ----------
    x : var_like, input value.
    size_splits : var_like, input value, dtype is int.
    axis : int, input value.

    Returns
    -------
    split_res : [Var].

    Example:
    -------
    >>> expr.split([[1., 2., 3.], [4., 5., 6.]], [1, 1], 0)
    [var([[1., 2., 3.]], dtype=float32), var([[4., 5., 6.]], dtype=float32)]
    '''
    x = _to_var(x)
    size_splits = _to_axis(size_splits)
    axis = _to_int(axis)
    return _F.split(x, size_splits, axis)
def strided_slice(x, begin, end, strides, begin_mask=0, end_mask=0,
                  ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    '''
    strided_slice(x, begin, end, strides, begin_mask=0, end_mask=0,
                  ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0)
    ``x[begin[0]:strides[0]:end[0], ..., begin[-1]:strides[-1]:end[-1]]`` of `x`.
    begin_mask:
    end_mask:
    ellipsis_mask:
    new_axis_mask:
    shrink_axis_mask:
    Parameters
    ----------
    x : var_like, input value.
    begin : var_like, input value, dtype is int.
    end : var_like, input value, dtype is int.
    strides : var_like, input value, dtype is int.
    begin_mask : int, input value. Default is 0.
    end_mask : int, input value. Default is 0.
    ellipsis_mask : int, input value. Default is 0.
    new_axis_mask : int, input value. Default is 0.
    shrink_axis_mask : int, input value. Default is 0.

    Returns
    -------
    strided_slice_res : Var.

    Example:
    -------
    >>> expr.strided_slice([[1., 2., 3.], [4., 5., 6.]], [0, 0], [1, 2], [1, 2])
    array([[1.]], dtype=float32)
    '''
    x = _to_var(x)
    begin = _to_var(begin)
    end = _to_var(end)
    strides = _to_var(strides)
    if begin.dtype != _F.int or end.dtype != _F.int or strides.dtype != _F.int:
        raise RuntimeError("parameter begin/end/strides must be int type")
    return _F.strided_slice(x, begin, end, strides, begin_mask, end_mask,
                            ellipsis_mask, new_axis_mask, shrink_axis_mask)
def concat(values, axis):
    '''
    concat(values, axis)
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    values : [var_like], input value.
    axis : int, input value.

    Returns
    -------
    concat_res : Var.

    Example:
    -------
    >>> expr.concat([[1., 2., 3.], [4., 5., 6.]], 0)
    var([1., 2., 3., 4., 5., 6.], dtype=float32)
    '''
    if type(values) not in (tuple, list):
        raise ValueError('parameter values must be [var_like]')
    values = [_to_var(value) for value in values]
    axis = _to_int(axis)
    return _F.concat(values, axis)
def where(x):
    '''
    where(x)
    return the index of `x > 0`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    where_res : Var, dtype is int.

    Example:
    -------
    >>> expr.where([0., 1., -2., 3.3])
    var([[1],[3]], dtype=int32)
    '''
    x = _to_var(x)
    return _F.where(x)
def convert(x, format):
    '''
    convert(x, format)
    connert data_format of `x` to `format`.

    Parameters
    ----------
    x : var_like, input value.
    format : data_format, input value, [NCHW, NHWC, NC4HW4].

    Returns
    -------
    convert_res : Var, data_format is format.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 8., 1.), [1, 2, 2, 2]) # data_format is NCHW
    >>> expr.convert(x, expr.NHWC)
    var([[[[0., 4.], [1., 5.]],
          [[2., 6.], [3., 7.]]]], dtype=float32)
    '''
    x = _to_var(x)
    return _F.convert(x, format)
def transpose(x, perm):
    '''
    transpose(x, perm)
    Permute the axes of an array; returns the modified array.

    Parameters
    ----------
    x : var_like, input value.
    perm : [int] or var_like, input value.

    Returns
    -------
    transpose_res : Var.

    Example:
    -------
    >>> expr.transpose([[1.,2.,3.],[4.,5.,6.]], [1,0])
    var([[1., 4.],
         [2., 5.],
         [3., 6.]], dtype=float32)
    '''
    x = _to_var(x)
    return _F.transpose(x, perm)
def channel_shuffle(x, axis):
    '''
    channel_shuffle(x, axis)
    do below operations:
    ``
    x = _Convert(x, NHWC);
    x = _Reshape(x, {0, 0, 0, group, -1}, NHWC);
    x = _Transpose(x, {0, 1, 2, 4, 3});
    x = _Reshape(x, {0, 0, 0, -1}, NHWC);
    channel_shuffle_res = _Convert(x, NC4HW4);
    ``

    Parameters
    ----------
    x : var_like, input value.
    axis : int, input value.

    Returns
    -------
    channel_shuffle_res : Var.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 8., 1.), [1, 4, 1, 2])
    >>> expr.convert(expr.channel_shuffle(x, 2), expr.NCHW)
    var([[[[0., 1.]],
          [[4., 5.]],
          [[2., 3.]],
          [[6., 7.]]]], dtype=float32)
    '''
    x = _to_var(x)
    axis = _to_int(axis)
    return _F.channel_shuffle(x, axis)
def reverse_sequence(x, y, batch_dim, seq_dim):
    '''
    reverse_sequence(x, y, batch_dim, seq_dim)
    1. Slice x along the dimension batch_dim;
    2. Reverses the y[i] elements along the dimension seq_dim.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.
    batch_dim : int, input value.
    seq_dim : int, input value.

    Returns
    -------
    transpose_res : Var.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 8., 1.), [2, 4])
    >>> expr.reverse_sequence(x, [1, 1], 0, 1)
    var([[0., 1., 2., 3.],
         [4., 5., 6., 7.]], dtype=float32)
    '''
    x = _to_var(x)
    y = _to_var(y)
    if y.dtype != _F.int or len(y.shape) != 1:
        raise ValueError("parameter y must be 1-D int type")
    if batch_dim < 0 or batch_dim >= len(x.shape):
        raise ValueError("parameter batch_dim must be in range of the number of dimensions of parameter x")
    if seq_dim < 0 or seq_dim >= len(x.shape):
        raise ValueError("parameter seq_dim must be in range of the number of dimensions of parameter x")
    if y.shape[0] != x.shape[batch_dim]:
        raise ValueError("parameter y must be shape [x.shape[batch.dim]]")
    return _F.reverse_sequence(x, y, batch_dim, seq_dim)
def crop(x, size, axis, offset):
    '''
    crop(x, size, axis, offset)
    Crop the `x` to `size` alone the dimension `axis`,
    and the start point is `offset`.

    Parameters
    ----------
    x : var_like, input value, data_format is NC4HW4.
    size : var_like, input value, dtype is int.
    axis : int, input value.
    offset : [int], input value.

    Returns
    -------
    crop_res : Var, data_format is NC4HW4.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 16., 1.), [1, 1, 4, 4])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> size = expr.const([0.0, 0.0, 0.0, 0.0], [1, 1, 2, 2])
    >>> expr.convert(expr.crop(x, size, 2, [1, 1]), expr.NCHW)
    var([[[[ 5.,  6.],
           [ 9., 10.]]]], dtype=float32)
    '''
    x = _to_var(x)
    size = _to_var(size)
    if len(x.shape) != 4 or x.data_format != _F.NC4HW4:
        raise RuntimeError("parameter x must be 4-D NC4HW4 format")
    if len(size.shape) != 4:
        raise RuntimeError("parameter size must be 4-D")
    if axis != 2 and axis != 3:
        raise RuntimeError("parameter axis must be 2 or 3, if 2 you may change both h/w, if 3 only w")
    if axis == 2:
        if len(offset) != 1 and len(offset) !=2:
            raise RuntimeError("parameter offset must be at most 2 if you want to change h/w")
    if axis == 3:
        if len(offset) != 1:
            raise RuntimeError("parameter offset must be at most 1 if you want to change w only")
    return _F.crop(x, size, axis, offset)
def resize(x, x_scale, y_scale):
    '''
    resize(x, x_scale, y_scale)
    Resize the (height,width) `x` to (y_scale * height, x_scale * width).

    Parameters
    ----------
    x : var_like, input value, data_format is NC4HW4.
    x_scale : float, input value, dtype is int.
    x_scale : float, input value.

    Returns
    -------
    resize_res : Var, data_format is NC4HW4.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 4., 1.), [1, 1, 2, 2])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> expr.convert(expr.resize(x, size, 2, 2), expr.NCHW)
    var([[[[0. , 0.5, 1. , 1. ],
           [1. , 1.5, 2. , 2. ],
           [2. , 2.5, 3. , 3. ],
           [2. , 2.5, 3. , 3. ]]]], dtype=float32)
    '''
    x = _to_var(x)
    x_scale = _to_float(x_scale)
    y_scale = _to_float(y_scale)
    if len(x.shape) != 4 or x.data_format != _F.NC4HW4:
        raise RuntimeError("parameter x must be 4-D NC4HW4 format")
    return _F.resize(x, x_scale, y_scale)
def crop_and_resize(x, boxes, box_ind, crop_size, method=_F.BILINEAR, extrapolation_value=0.):
    '''
    crop_and_resize(x, boxes, box_ind, crop_size,=_F.BILINEAR, extrapolation_value=0.)
    Extracts crops from the `x` and resizes them using `method` to output size `crop_size`. 

    Parameters
    ----------
    x : var_like, input value, data_format is NHWC.
    boxes : var_like, input value.
    box_ind : var_like, input value.
    crop_size : var_like, input value.
    method : Interp_Method, input value, [NEAREST, BILINEAR]. Default is BILINEAR.
    extrapolation_value : float, input value. Default is 0.

    Returns
    -------
    crop_and_resize_res : Var, data_format is NC4HW4.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 16., 1.), [1, 1, 4, 4])
    >>> x = expr.convert(x, expr.NHWC)
    >>> boxes = expr.const([0.2, 0.3, 0.4, 0.5], [1, 4])
    >>> expr.convert(expr.crop_and_resize(x, boxes, [0], [2, 2]), expr.NHWC)
    var([[[[3.3000002],
           [3.9      ]],
           [[5.7000003],
           [6.3      ]]]], dtype=float32)
    ''' 
    x = _to_var(x)
    boxes = _to_var(boxes)
    box_ind = _to_var(box_ind)
    crop_size = _to_var(crop_size)
    if len(x.shape) != 4:
        raise RuntimeError("parameter x must be 4-D format")
    if box_ind.dtype != _F.int or crop_size.dtype != _F.int:
        raise RuntimeError("parameter box_ind/crop_size must be int type")
    if len(boxes.shape) != 2 or boxes.shape[-1] !=4:
        raise RuntimeError("parameter boxes must be 2-D w/ shape [num_boxes, 4]")
    if len(box_ind.shape) != 1 or box_ind.shape[-1] != boxes.shape[0]:
        raise RuntimeError("parameter boxes must be 1-D w/ shape [num_boxes]")
    if len(crop_size.shape) != 1 or crop_size.shape[0] != 2:
        raise RuntimeError("parameter boxes must be 1-D w/ shape [2]")
    return _F.crop_and_resize(x, boxes, box_ind, crop_size, method, extrapolation_value)
def pad(x, paddings, mode=_F.CONSTANT):
    '''
    pad(x, paddings, mode=_F.CONSTANT)
    Padding the `x` whith `paddings` size.
 
    Parameters
    ----------
    x : var_like, input value.
    paddings : var_like, input value, shape is [x.ndim, 2], dtype is int.
    mode : PadValue_Mode, input value, [CONSTANT, REFLECT, SYMMETRIC]. Default is CONSTANT.

    Returns
    -------
    pad_res : Var.

    Example:
    -------
    >>> expr.pad([[1.,2.],[3.,4.]], [[0, 1], [1,2]])
    var([[0., 1., 2., 0., 0.],
         [0., 3., 4., 0., 0.],
         [0., 0., 0., 0., 0.]], dtype=float32)
    ''' 
    x = _to_var(x)
    paddings = _to_var(paddings)
    if paddings.dtype != _F.int:
        raise RuntimeError("parameter perm must be int type")
    if len(paddings.shape) != 2 or paddings.shape[-1] != 2 or paddings.shape[0] != len(x.shape):
        raise RuntimeError("parameter paddings must be 2-D w/ shape[n, 2], and n is the number of dimensions of parameter x")
    return _F.pad(x, paddings, mode)
def randomuniform(shape, dtype=_F.float, low=0.0, high=1.0, seed0=0, seed1=1):
    '''
    randomuniform(shape, dtype=_F.float, low=0.0, high=1.0, seed0=0, seed1=1)
    Generate `dtype` uniform random values of `shape`.
    The random value is ``low < value < high``.
    The generation seed is using `seed0` and `seed1`.
 
    Parameters
    ----------
    shape : axis_like, input value.
    dtype : dtype, input value, [float, double, int, int64, uint8]. Default is float.
    low : float, input value. Default is 0.0;
    high : float, input value. Default is 1.0;
    seed0 : int, input value. Default is 0;
    seed1 : int, input value. Default is 0;

    Returns
    -------
    pad_res : Var.

    Example:
    -------
    >>> expr.pad([[1.,2.],[3.,4.]], [[0, 1], [1,2]])
    var([[0., 1., 2., 0., 0.],
         [0., 3., 4., 0., 0.],
         [0., 0., 0., 0., 0.]], dtype=float32)
    ''' 
    shape = _to_axis(shape)
    low = _to_float(low)
    high = _to_float(high)
    seed0 = _to_int(seed0)
    seed1 = _to_int(seed1)
    return _F.randomuniform(shape, dtype, low, high, seed0, seed1)
def expand_dims(x, axis):
    '''
    expand_dims(x, axis)
    Add `1` to the dimension of `x` by `axis`.

    Parameters
    ----------
    x : var_like, input value.
    axis : int or var_like, input value, dtype is int.

    Returns
    -------
    expand_dims_res : Var.

    Example:
    -------
    >>> expr.expand_dims([1.,2.], 0)
    var([[1., 2.]], dtype=float32)
    ''' 
    x = _to_var(x)
    return _F.expand_dims(x, axis)
def rank(x):
    '''
    rank(x)
    Get the rank(ndim) of `x`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    rank_res : Var, dtype is int.

    Example:
    -------
    >>> expr.rank([[1.,2.],[3.,4.]])
    var(2, dtype=int32)
    ''' 
    x = _to_var(x)
    return _F.rank(x)
def size(x):
    '''
    size(x)
    Get the size of `x`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    size_res : Var, dtype is int.

    Example:
    -------
    >>> expr.size([[1.,2.],[3.,4.]])
    var(4, dtype=int32)
    ''' 
    x = _to_var(x)
    return _F.size(x)
def shape(x):
    '''
    shape(x)
    Get the shape of `x`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    shape_res : Var, dtype is int.

    Example:
    -------
    >>> expr.shape([[1.,2.],[3.,4.]])
    var([2, 2], dtype=int32)
    ''' 
    x = _to_var(x)
    return _F.shape(x)
def stack(values, axis=0):
    '''
    stack(values, axis=0)
    Stack the list of var `valuse` to a var along the `axis`.

    Parameters
    ----------
    values : [var_like], input value.
    axis : int, input value. Default is 0.

    Returns
    -------
    stack_res : Var.

    Example:
    -------
    >>> x = expr.const([1., 2.], [2])
    >>> y = expr.const([3., 4.], [2])
    >>> expr.stack([x, y])
    var([[1., 2.],
         [3., 4.]], dtype=float32)
    ''' 
    if not isinstance(values, (list, tuple)):
        raise RuntimeError("parameter values must be a list/tuple of MNN Var")
    if len(values) < 1:
        raise RuntimeError("parameter values must have at least one item")
    values = [_to_var(value) for value in values]
    for value in values:
        if value.shape != values[0].shape or value.dtype != values[0].dtype:
            raise RuntimeError("all items in parameter values must have same shape and dtype")
    axis = _to_int(axis)
    return _F.stack(values, axis)
def unstack(x, axis=0):
    '''
    unstack(x, axis=0)
    Unstack the list of var `x` to a var along the `axis`.

    Parameters
    ----------
    x : var_like, input value.
    axis : int, input value. Default is 0.

    Returns
    -------
    unstack_res : [Var].

    Example:
    -------
    >>> expr.unstack([[1., 2.], [3., 4.]])
    [var([1., 2.], dtype=float32), var([3., 4.], dtype=float32)]
    ''' 
    x = _to_var(x)
    axis = _to_int(axis)
    return _F.unstack(x, axis)
def fill(shape, value):
    '''
    fill(shape, value)
    Fill the `value` to a var with `shape`.

    Parameters
    ----------
    shape : var_like, input value, dtype is int.
    value : var_like, input value.

    Returns
    -------
    fill_res : Var.

    Example:
    -------
    >>> expr.fill([2,3], 1.0)
    var([[1., 1., 1.],
         [1., 1., 1.]], dtype=float32)
    ''' 
    shape = _to_var(shape)
    value = _to_var(value)
    if shape.dtype != _F.int:
        raise RuntimeError("parameter dims must be int type")
    if len(value.shape) != 0:
        raise RuntimeError("parameter value must be 0-D")
    return _F.fill(shape, value)
def tile(x, multiples):
    '''
    tile(x, multiples)
    Repeat `x` `multiples` times.

    Parameters
    ----------
    x : var_like, input value.
    multiples : var_like, input value, dtype is int.

    Returns
    -------
    tile_res : Var.

    Example:
    -------
    >>> expr.tile([2,3], [3])
    var([2, 3, 2, 3, 2, 3], dtype=int32)
    ''' 
    x = _to_var(x)
    multiples = _to_var(multiples)
    if multiples.dtype != _F.int or len(multiples.shape) != 1:
        raise RuntimeError("parameter multiples must be 1-D int type")
    if len(x.shape) != multiples.shape[-1]:
        raise RuntimeError("parameter multiples's length must match w/ number of dimensions of input")
    return _F.tile(x, multiples)
def gather(x, index):
    '''
    gather(x, index)
    Gether `x` by the `index`, means ``x[index]``.

    Parameters
    ----------
    x : var_like, input value.
    index : var_like, input value, dtype is int.

    Returns
    -------
    gather_res : Var.

    Example:
    -------
    >>> expr.gather([[1.,2.],[3.,4.]], [1])
    var([[3., 4.]], dtype=float32)
    '''
    x = _to_var(x)
    index = _to_var(index)
    if index.dtype != _F.int:
        raise RuntimeError("parameter index must be int type")
    return _F.gather(x, index)
def gather_nd(x, indices):
    '''
    gather_nd(x, indices)
    Gether `x` by the `indices`, means ``x[indices[0], ..., indices[-1]]``.

    Parameters
    ----------
    x : var_like, input value.
    indices : var_like, input value, dtype is int.

    Returns
    -------
    gather_nd_res : Var.

    Example:
    -------
    >>> expr.gather_nd([[1.,2.],[3.,4.]], [1, 0])
    var(3., dtype=float32)
    '''
    x = _to_var(x)
    indices = _to_var(indices)
    if indices.dtype != _F.int:
        raise RuntimeError("parameter indices must be int type")
    return _F.gather_nd(x, indices)
def select(cond, x, y):
    '''
    select(cond, x, y)
    Return elements chosen from `x` or `y` depending on `cond`.

    Parameters
    ----------
    cond : var_like, input value.
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    select_res : Var.

    Example:
    -------
    >>> expr.select([1., 0., 2.], [-1., -2., -3.], [1., 2., 3.])
    var([-1.,  2., -3.], dtype=float32)
    '''
    cond = _to_var(cond)
    x = _to_var(x)
    y = _to_var(y)
    return _F.select(cond, x, y)
def squeeze(x, axes=[]):
    '''
    squeeze(x, axes=[])
    Remove the `1` from the dimensions of `x` by `axes`.

    Parameters
    ----------
    x : var_like, input value.
    axes : axis_like, input value. Default is [].

    Returns
    -------
    squeeze_res : Var.

    Example:
    -------
    >>> expr.squeeze([[[1.0, 2.0]]], [0, 1])
    var([1., 2.], dtype=float32)
    '''
    x = _to_var(x)
    axes = _to_axis(axes)
    return _F.squeeze(x, axes)
def unsqueeze(x, axes=[]):
    '''
    unsqueeze(x, axes=[])
    Add the `1` from the dimensions of `x` by `axes`.

    Parameters
    ----------
    x : var_like, input value.
    axes : axis_like, input value.  Default is [].

    Returns
    -------
    unsqueeze_res : Var.

    Example:
    -------
    >>> expr.unsqueeze([1.0, 2.0], [0, 1])
    var([[[1., 2.]]], dtype=float32)
    '''
    x = _to_var(x)
    axes = _to_axis(axes)
    return _F.unsqueeze(x, axes)
def depth_to_space(x, axis):
    '''
    depth_to_space(x, axis)
    It is the reverse transformation of SpaceToDepth.

    Parameters
    ----------
    x : var_like, input value, data_format is NHWC.
    axis : int, input value.

    Returns
    -------
    depth_to_space_res : Var.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 12., 1.), [1, 4, 1, 3])
    >>> expr.depth_to_space(x, 2)
    var([[[[ 0.,  3.,  1.,  4.,  2.,  5.],
           [ 6.,  9.,  7., 10.,  8., 11.]]]], dtype=float32)
    '''
    x = _to_var(x)
    axis = _to_int(axis)
    return _F.depth_to_space(x, axis)
def space_to_depth(x, axis):
    '''
    space_to_depth(x, axis)
    Rearranges blocks of spatial data, into depth.

    Parameters
    ----------
    x : var_like, input value, data_format is NHWC.
    axis : int, input value.

    Returns
    -------
    space_to_depth_res : Var.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 16., 1.), [1, 1, 4, 4])
    >>> x = expr.convert(x, expr.NHWC)
    >>> expr.space_to_depth(x, 2)
    var([[[[ 0.,  1.,  4.,  5.],
           [ 2.,  3.,  6.,  7.]],
          [[ 8.,  9., 12., 13.],
           [10., 11., 14., 15.]]]], dtype=float32)
    '''
    x = _to_var(x)
    axis = _to_int(axis)
    return _F.space_to_depth(x, axis)
def batch_to_space_nd(x, block_shape, crops):
    '''
    batch_to_space_nd(x, block_shape, crops)
    Reshapes the batch dimension 0 into M + 1 dimensions of shape block_shape + [batch].
    Interleaves these blocks back into the grid defined by the spatial dimensions [1, ..., M], 
    to obtain a result with the same rank as the input.
    The spatial dimensions of this intermediate result are then optionally cropped according
    to crops to produce the output.

    Parameters
    ----------
    x : var_like, input value, data_format is NC4HW4.
    block_shape : var_like, input value, dtype is int.
    crops : var_like, input value, dtype is int.

    Returns
    -------
    batch_to_space_nd_res : Var.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 12., 1.), [4, 1, 1, 3])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> crops = expr.const([0, 0, 0, 0], [2, 2], expr.NCHW, expr.int)
    >>> expr.convert(expr.batch_to_space_nd(x, [2, 2], crops), expr.NHWC)
    var([[[[ 0.],
           [ 3.],
           [ 1.],
           [ 4.],
           [ 2.],
           [ 5.]],
          [[ 6.],
           [ 9.],
           [ 7.],
           [10.],
           [ 8.],
           [11.]]]], dtype=float32)
    '''
    x = _to_var(x)
    block_shape = _to_var(block_shape)
    crops = _to_var(crops)
    if len(x.shape) != 4 or x.data_format != _F.NC4HW4:
        raise RuntimeError("parameter x must be 4-D w/ NC4HW4 format")
    if block_shape.dtype != _F.int or crops.dtype != _F.int:
        raise RuntimeError("parameter block_shape/crops must be int type")
    if len(block_shape.shape) != 1:
        raise RuntimeError("parameter block_shape must be 1-D w/ shape [M]")
    if len(crops.shape) != 2 or crops.shape[-1] != 2 or crops.shape[0] != block_shape.shape[0]:
        raise RuntimeError("parameter crops must be 2-D w/ shape [M, 2]")
    return _F.batch_to_space_nd(x, block_shape, crops)
def space_to_batch_nd(x, block_shape, paddings):
    '''
    space_to_batch_nd(x, block_shape, crops)
    Divides "spatial" dimensions [1, ..., M] of the input into a grid of blocks of shape block_shape. 
    And interleaves these blocks with the "batch" dimension.

    Parameters
    ----------
    x : var_like, input value, data_format is NC4HW4.
    block_shape : var_like, input value, dtype is int.
    paddings : var_like, input value, dtype is int.

    Returns
    -------
    space_to_batch_nd_res : Var.

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 12., 1.), [3, 1, 2, 2])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> paddings = expr.const([0, 0, 0, 0], [2, 2], expr.NCHW, expr.int)
    >>> expr.convert(expr.space_to_batch_nd(x, [2, 2], paddings), expr.NHWC)
    var([[[[ 0.]]],
         [[[ 4.]]],
         [[[ 8.]]],
         [[[ 1.]]],
         [[[ 5.]]],
         [[[ 9.]]],
         [[[ 2.]]],
         [[[ 6.]]],
         [[[10.]]],
         [[[ 3.]]],
         [[[ 7.]]],
         [[[11.]]]], dtype=float32)
    '''
    x = _to_var(x)
    block_shape = _to_var(block_shape)
    paddings = _to_var(paddings)
    if len(x.shape) != 4 or x.data_format != _F.NC4HW4:
        raise RuntimeError("parameter x must be 4-D w/ NC4HW4 format")
    if block_shape.dtype != _F.int or paddings.dtype != _F.int:
        raise RuntimeError("parameter block_shape/paddings must be int type")
    if len(block_shape.shape) != 1:
        raise RuntimeError("parameter block_shape must be 1-D w/ shape [M]")
    if len(paddings.shape) != 2 or paddings.shape[-1] != 2:
        raise RuntimeError("parameter paddings must be 2-D w/ shape [M, 2]")
    return _F.space_to_batch_nd(x, block_shape, paddings)
def elu(x, alpha):
    '''
    elu(x, alpha)
    ``select(x > 0, x, alpha * (exp(x) - 1))``

    Parameters
    ----------
    x : var_like, input value.
    alpha : float, input value.

    Returns
    -------
    elu_res : Var.

    Example:
    -------
    >>> expr.elu([-1.0, 2.0], 1.673263)
    array([-1.0577048,  2.], dtype=float32)
    '''
    x = _to_var(x)
    alpha = _to_float(alpha)
    return _F.elu(x, alpha)
def selu(x, scale, alpha):
    '''
    selu(x, scale, alpha)
    ``scale * select(x >=0, x, alpha * elu(x))``
    Parameters
    ----------
    x : var_like, input value.
    scale : float, input value.
    alpha : float, input value.

    Returns
    -------
    selu_res : Var.

    Example:
    -------
    >>> expr.selu([-1.0, 2.0], 1.0507, 1.673263)
    var([-1.1113304, 2.1014], dtype=float32)
    '''
    x = _to_var(x)
    scale = _to_float(scale)
    alpha = _to_float(alpha)
    return _F.selu(x, scale, alpha)
def matrix_band_part(x, num_lower, num_upper):
    '''
    matrix_band_part(x, num_lower, num_upper)
    Copies a var setting everything outside a central band in each innermost matrix.
    num_lower: Number of subdiagonals to keep. If negative, keep entire lower triangle.
    num_upper: Number of superdiagonals to keep. If negative, keep entire upper triangle.

    Parameters
    ----------
    x : var_like, input value.
    num_lower : var_like, input value, dtype is int.
    num_upper : var_like, input value, dtype is int.

    Returns
    -------
    matrix_band_part_res : Var.

    Example:
    -------
    >>> expr.matrix_band_part([[-2., 1.], [-1., 2.]], 1, -1)
    var([[-2.,  1.],
         [-1.,  2.]], dtype=float32)
    ''' 
    x = _to_var(x)
    num_lower = _to_var(num_lower)
    num_upper = _to_var(num_upper)
    if len(num_lower.shape) != 0 or num_lower.dtype != _F.int:
        raise RuntimeError("parameter num_lower must be 0-D int")
    if len(num_upper.shape) != 0 or num_upper.dtype != _F.int:
        raise RuntimeError("parameter num_upper must be 0-D int")
    return _F.matrix_band_part(x, num_lower, num_upper)
def moments(x, axes=[2, 3], shift=None, keep_dims=True):
    '''
    moments(x, axes=[2, 3], shift=None, keep_dims=True)
    Calculates the mean and variance of x.

    Parameters
    ----------
    x : var_like, input value, data_format is NC4HW4.
    axes : axis_like, input value. Default is [2,3].
    shift : var_like, input value, dtype is int. Default is None.
    keep_dims : bool, input value. Default is True.

    Returns
    -------
    moments_res : [Var].

    Example:
    -------
    >>> x = expr.reshape(expr.range(0., 4., 1.), [1, 1, 2, 2])
    >>> x = expr.convert(x, expr.NC4HW4)
    >>> expr.moments(x)
    [var([[[[1.5]]]], dtype=float32), var([[[[1.25]]]], dtype=float32)]
    '''
    x = _to_var(x)
    if len(x.shape) != 4 or x.data_format != _F.NC4HW4:
        raise RuntimeError("parameter x must be 4-D w/ NC4HW4 format")
    if axes != [2, 3] and axes != (2, 3):
        raise RuntimeError("parameter axes must be [2, 3] in current implementation")
    if shift is None:
        shift = _F.const([0.], [1]) #though it's not used, it's preserved
    return _F.moments(x, axes, shift, keep_dims)
def setdiff1d(x, y):
    '''
    setdiff1d(x, y)
    Return ``set(x) - set(y)`` of `x`.

    Parameters
    ----------
    x : var_like, input value.
    y : var_like, input value.

    Returns
    -------
    setdiff1d_res : Var.

    Example:
    -------
    >>> expr.setdiff1d([1, 2, 3], [2, 3, 4])
    var([1], dtype=int32)
    '''
    x = _to_var(x)
    y = _to_var(y)
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise RuntimeError("parameter x/y must be 1-D")
    return _F.setdiff1d(x, y)
def zeros_like(x):
    '''
    zeros_like(x)
    Return all `0` var shape is same as `x`.

    Parameters
    ----------
    x : var_like, input value.

    Returns
    -------
    zeros_like_res : Var.

    Example:
    -------
    >>> expr.zeros_like([[1, 2], [3, 4]])
    var([[0, 0],
           [0, 0]], dtype=int32)
    '''
    x = _to_var(x)
    return _F.zeros_like(x)
def range(start, limit, delta):
    '''
    range(start, limit, delta)
    Return sequence of range ``[start:delta:limit]``.

    Parameters
    ----------
    start : var_like, input value, dtype is int.
    limit : var_like, input value, dtype is int.
    delta : var_like, input value, dtype is int.

    Returns
    -------
    range_res : Var.

    Example:
    -------
    >>> expr.range(1.0, 7.0, 2.0)
    var([1., 3., 5.], dtype=float32)
    '''
    start = _to_var(start)
    limit = _to_var(limit)
    delta = _to_var(delta)
    if limit.dtype != start.dtype or delta.dtype != start.dtype:
        raise RuntimeError("parameter start/limit/delta must use same data type, either all int or all float")
    return _F.range(start, limit, delta)
def sort(x, axis=-1, arg=False, descend=False):
    '''
    sort(x, axis=-1, arg=False, descend=False)
    Return the sorted array of ``x``.

    Parameters
    ----------
    x : var_like, input value.
    axis : int, sort by axis.
    arg : is ArgSort or not, default is False.
    descend : is descend or not, default is False.

    Returns
    -------
    sorted_res : Var.

    Example:
    -------
    >>> expr.sort([[5, 0], [1, 3]])
    var([[1, 0],
         [5, 3]], dtype=int32)
    '''
    x = _to_var(x)
    # sort will change the x
    x = clone(x, True)
    return _F.sort(x, axis, arg, descend)
def nms(boxes, scores, max_detections, iou_threshold=-1.0, score_threshold=-1.0):
    '''
    nms(boxes, scores, max_detections, iou_threshold=-1.0, score_threshold=-1.0)
    Return the nms array of ``boxes``.

    Parameters
    ----------
    boxes : var_like, input value, shape must be [num, 4].
    scores : var_like, input value, shape must be [num].
    max_detections : int.
    iou_threshold : float, default is 0.
    score_threshold : float, default is float_min.

    Returns
    -------
    nms_res : Var.

    Example:
    -------
    >>> expr.nms([[1, 1, 4, 4], [0, 0, 3, 3], [5, 5, 7, 7]], [0.9, 0.5, 0.1], 3, 0.1)
    var([0, 2], dtype=int32)
    '''
    boxes = _to_var(boxes, _F.float)
    scores = _to_var(scores, _F.float)
    max_detections = _to_int(max_detections)
    iou_threshold = _to_float(iou_threshold)
    score_threshold = _to_float(score_threshold)
    res = _F.nms(boxes, scores, max_detections, iou_threshold, score_threshold)
    idx = res >= 0
    idx.fix_as_const()
    if _F.reduce_any(idx).read_as_tuple()[0] == 0:
        return _F.const([], [0], NCHW, _F.int)
    return res[idx]
# TODO: detection_post_process
def roi_pooling(input, roi, pooled_height, pooled_width, spatial_scale, output_grad = False, backward_diff = None):
    '''
    roi_pooling(input, roi, pooled_height, pooled_width, spatial_scale, output_grad = False, backward_diff=None)
    Return the roi pooling result of ``input``.

    Parameters
    ----------
    input: val_like, input value, must be nc4hw4 format.
    roi: val_like, input value.
    pooled_height: int.
    pooled_width: int.
    spatial_scale: float.
    output_grad: bool, if true, use backward mode, output the grad of ``input`` with ``backward_diff``.
    backward_diff: val_like, the backward diff used in backward mode, must be nc4hw4 format.

    Returns
    -------
    roi_pooling result : Var, in nc4hw4 format.

    Example:
    -------
    >>> from MNN import expr as F
    >>> a = F.randomuniform((3, 9, 4, 4))
    >>> a = F.convert(a, F.NC4HW4)
    >>> spatialScale = 1.0 / 16
    >>> roi = np.array([[2, 1 / spatialScale, 2 / spatialScale, 3 / spatialScale, 3 / spatialScale],
                    [0, 0 / spatialScale, 2 / spatialScale, 2 / spatialScale, 3 / spatialScale]])
    >>> roi = F.const(roi.astype(np.float32), roi.shape, F.NCHW, F.float)
    >>> c = F.roi_pooling(a, roi, 3, 3, spatialScale)
    >>> print(F.convert(c, F.NCHW))
    '''

    input = _to_c4(_to_var(input))
    roi = _to_var(roi)
    pooled_height = _to_int(pooled_height)
    pooled_width = _to_int(pooled_width)
    spatial_scale = _to_float(spatial_scale)
    output_grad = _to_int(output_grad)
    if backward_diff is None:
        backward_diff = _F.convert(scalar(0.0), _F.NC4HW4)
    else:
        backward_diff = _to_c4(_to_var(backward_diff))
    return _F.roi_pooling(input, roi, pooled_height, pooled_width, spatial_scale, output_grad, backward_diff)
def roi_align(input, roi, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned, pooling_mode, output_grad = False, backward_diff = None):
    '''
    roi_align(input, roi, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned, pooling_mode, output_grad = False, backward_diff = None)
    Return the roi align result of ``input``.

    Parameters
    ----------
    input: val_like, input value, must be nc4hw4 format.
    roi: val_like, input value.
    pooled_height: int.
    pooled_width: int.
    spatial_scale: float.
    sampling_ratio: int.
    aligned: bool.
    pooling_mode: F.Pooling_Mode.AVEPOOL or F.Pooling_Mode.MAXPOOL.
    output_grad: bool, if true, use backward mode, output the grad of ``input`` with ``backward_diff``.
    backward_diff: val_like, the backward diff used in backward mode, must be nc4hw4 format.

    Returns
    -------
    roi_align result : Var, in nc4hw4 format.

    Example:
    -------
    >>> from MNN import expr as F
    >>> a = F.randomuniform((3, 9, 4, 4))
    >>> a = F.convert(a, F.NC4HW4)
    >>> spatialScale = 1.0 / 16
    >>> roi = np.array([[2, 1 / spatialScale, 2 / spatialScale, 3 / spatialScale, 3 / spatialScale],
                    [0, 0 / spatialScale, 2 / spatialScale, 2 / spatialScale, 3 / spatialScale]])
    >>> roi = F.const(roi.astype(np.float32), roi.shape, F.NCHW, F.float)
    >>> sampling_ratio = 2
    >>> aligned = True
    >>> pooling_mode = F.Pool_Mode.AVEPOOL
    >>> c = F.roi_align(a, roi, 3, 3, spatialScale, sampling_ratio, aligned, pooling_mode)
    >>> print(F.convert(c, F.NCHW))
    '''

    input = _to_c4(_to_var(input))
    roi = _to_var(roi)
    pooled_height = _to_int(pooled_height)
    pooled_width = _to_int(pooled_width)
    spatial_scale = _to_float(spatial_scale)
    sampling_ratio = _to_int(sampling_ratio)
    aligned = _to_int(aligned)
    output_grad = _to_int(output_grad)
    if backward_diff is None:
        backward_diff = _F.convert(scalar(0.0), _F.NC4HW4)
    else:
        backward_diff = _to_c4(_to_var(backward_diff))
    return _F.roi_align(input, roi, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned, pooling_mode, output_grad, backward_diff)
# wrapper for builtin functions end
