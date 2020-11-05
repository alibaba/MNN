_Int = int
_Float = float
from _mnncengine._expr import *
import _mnncengine._expr as _F
import numpy as np
def _to_var(x, to_float=True):
    if isinstance(x, np.ndarray):
        if to_float:
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            return _F.const(x, x.shape)
        if not to_float:
            if x.dtype != np.int32:
                x = x.astype(np.int32)
            return _F.const(x, x.shape, dtype=_F.int)
    elif isinstance(x, (list, tuple)) and x:
        x = np.array(x)
        if to_float:
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            return _F.const(x, x.shape)
        if not to_float:
            if x.dtype != np.int32:
                x = x.astype(np.int32)
            return _F.const(x, x.shape, dtype=_F.int)
    elif isinstance(x, _Int):
        return _F.const(x, [], dtype=_F.int)
    elif isinstance(x, _Float):
        return _F.const(x, [], dtype=_F.float)
    return x 
def scalar(value):
    if type(value) == type(1):
        res = _F.const([value], [], _F.NCHW, _F.int)
        return res
    elif type(value) == type(1.):
        res = _F.const([value], [], _F.NCHW, _F.float)
        return res
    else:
        raise NotImplementedError("not supported data type for creating scalar variable")
def sign(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.sign(x)
def floor(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.floor(x)
def ceil(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.ceil(x)
def square(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.square(x)  
def sqrt(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.sqrt(x)  
def rsqrt(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.rsqrt(x)  
def exp(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.exp(x)
def log(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.log(x)
def sin(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.sin(x)
def cos(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.cos(x)
def tan(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.tan(x)
def asin(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.asin(x)
def acos(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.acos(x) 
def atan(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.atan(x)
def log1p(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.log1p(x)
def tanh(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.tanh(x)
def sigmoid(x):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    return _F.sigmoid(x)
def minimum(x, y):
    x = _to_var(x)
    y = _to_var(y)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    if not isinstance(y, Var):
        raise RuntimeError("parameter y is not valid")
    return _F.minimum(x, y)
def maximum(x, y):
    x = _to_var(x)
    y = _to_var(y)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    if not isinstance(y, Var):
        raise RuntimeError("parameter y is not valid")
    return _F.maximum(x, y)
def bias_add(value, bias):
    """
    Adds bias to value.

    This is (mostly) a special case of add where bias is restricted to 1-D.
    Broadcasting is supported, so value may have any number of dimensions.
    Unlike add, the type of bias is allowed to differ from value in the case where both types are quantized.

    Example usage:
    >>> MNN.expr.bias_add(np.eye(3,3), np.ones(3))
    array([[2., 1., 1.],
       [1., 2., 1.],
       [1., 1., 2.]], dtype=float32)

    Args:
    value: A variable with type dtype.float or dtype.int.
    bias: A 1-D variable with size matching the channel dimension of value.
          Must be the same type as value unless value is a quantized type, in which case a different quantized type may be used.

    Returns:
    A variable with the same type as value.
  """
    value = _to_var(value)
    bias = _to_var(bias)
    if not isinstance(value, Var):
        raise RuntimeError("parameter value is not valid")
    if not isinstance(bias, Var):
        raise RuntimeError("parameter bias is not valid")
    if len(bias.shape) != 1:
        raise RuntimeError("parameter bias must be 1-D in bias_add")
    if value.shape[-1] != bias.shape[-1]:
        raise RuntimeError("parameter bias's dim must match parameter value's dim in bias_add")
    return _F.bias_add(value, bias)
def unravel_index(indices, dims):
    indices = _to_var(indices, to_float=False)
    dims = _to_var(dims, to_float=False)
    if not isinstance(indices, Var):
        raise RuntimeError("parameter indices is not valid")
    if not isinstance(dims, Var):
        raise RuntimeError("parameter dims is not valid")
    return _F.unravel_index(indices, dims)
def one_hot(indices, depth, on_value=1., off_value=0., axis=-1):
    indices = _to_var(indices, to_float=False)
    if not isinstance(indices, Var):
        raise RuntimeError("parameter indices is not valid")
    return _F.one_hot(indices, depth, on_value, off_value, axis)
def broadcast_to(input, shape):
    shape = _to_var(shape, to_float=False)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    if not isinstance(shape, Var):
        raise RuntimeError("parameter shape is not valid")
    return _F.broadcast_to(input, shape)
def zeros_like(input):
    input = _to_var(input)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    return _F.zeros_like(input)
def range(start, limit, delta):
    start = _to_var(start)
    limit = _to_var(limit)
    delta = _to_var(delta)
    if not isinstance(start, Var):
        raise RuntimeError("parameter start is not valid")
    if not isinstance(limit, Var):
        raise RuntimeError("parameter limit is not valid")
    if not isinstance(delta, Var):
        raise RuntimeError("parameter delta is not valid")
    if limit.dtype != start.dtype or delta.dtype != start.dtype:
        raise RuntimeError("parameter start/limit/delta must use same data type, either all int or all float")
    return _F.range(start, limit, delta)
def rank(input):
    input = _to_var(input)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    return _F.rank(input)
def space_to_batch_nd(input, block_shape, paddings):
    input = _to_var(input)
    block_shape = _to_var(block_shape, to_float=False)
    paddings = _to_var(paddings, to_float=False)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    if not isinstance(block_shape, Var):
        raise RuntimeError("parameter block_shape is not valid")
    if not isinstance(paddings, Var):
        raise RuntimeError("parameter paddings is not valid")
    if len(input.shape) != 4 or input.data_format != _F.NC4HW4:
        raise RuntimeError("parameter input must be 4-D w/ NC4HW4 format")
    if block_shape.dtype != _F.int or paddings.dtype != _F.int:
        raise RuntimeError("parameter block_shape/paddings must be int type")
    if len(block_shape.shape) != 1:
        raise RuntimeError("parameter block_shape must be 1-D w/ shape [M]")
    if len(paddings.shape) != 2 or paddings.shape[-1] != 2:
        raise RuntimeError("parameter paddings must be 2-D w/ shape [M, 2]") 
    return _F.space_to_batch_nd(input, block_shape, paddings)
def batch_to_space_nd(input, block_shape, crops):
    input = _to_var(input)
    block_shape = _to_var(block_shape, to_float=False)
    crops = _to_var(crops, to_float=False)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    if not isinstance(block_shape, Var):
        raise RuntimeError("parameter block_shape is not valid")
    if not isinstance(crops, Var):
        raise RuntimeError("parameter crops is not valid")
    if len(input.shape) != 4 or input.data_format != _F.NC4HW4:
        raise RuntimeError("parameter input must be 4-D w/ NC4HW4 format")
    if block_shape.dtype != _F.int or crops.dtype != _F.int:
        raise RuntimeError("parameter block_shape/crops must be int type")
    if len(block_shape.shape) != 1:
        raise RuntimeError("parameter block_shape must be 1-D w/ shape [M]")
    if len(crops.shape) != 2 or crops.shape[-1] != 2 or crops.shape[0] != block_shape.shape[0]:
        raise RuntimeError("parameter crops must be 2-D w/ shape [M, 2]")
    return _F.batch_to_space_nd(input, block_shape, crops)
def setdiff1d(x, y):
    x = _to_var(x)
    y = _to_var(y)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    if not isinstance(y, Var):
        raise RuntimeError("parameter y is not valid")
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise RuntimeError("parameter x/y must be 1-D")
    return _F.setdiff1d(x, y)
def moments(x, axes=[2, 3], shift=None, keep_dims=True):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    if len(x.shape) != 4 or x.data_format != _F.NC4HW4:
        raise RuntimeError("parameter x must be 4-D w/ NC4HW4 format")
    if axes != [2, 3] and axes != (2, 3):
        raise RuntimeError("parameter axes must be [2, 3] in current implementation")
    shift = _F.const([0.], [1]) #though it's not used, it's preserved
    return _F.moments(x, axes, shift, True)
def matrix_band_part(input, num_lower, num_upper):
    input = _to_var(input)
    num_lower = _to_var(num_lower)
    num_upper = _to_var(num_upper)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    if not isinstance(num_lower, Var):
        raise RuntimeError("parameter num_lower is not valid")
    if not isinstance(num_upper, Var):
        raise RuntimeError("parameter num_upper is not valid")
    if len(num_lower.shape) != 0 or num_lower.dtype != _F.int:
        raise RuntimeError("parameter num_lower must be 0-D int")
    if len(num_upper.shape) != 0 or num_upper.dtype != _F.int:
        raise RuntimeError("parameter num_upper must be 0-D int")
    return _F.matrix_band_part(input, num_lower, num_upper)
def gather_nd(params, indices):
    params = _to_var(params)
    indices = _to_var(indices, to_float=False)
    if not isinstance(params, Var):
        raise RuntimeError("parameter params is not valid")
    if not isinstance(indices, Var):
        raise RuntimeError("parameter indices is not valid")
    if indices.dtype != _F.int:
        raise RuntimeError("parameter indices must be int type")
    return _F.gather_nd(params, indices)
def gather(params, indices):
    params = _to_var(params)
    indices = _to_var(indices, to_float=False)
    if not isinstance(params, Var):
        raise RuntimeError("parameter params is not valid")
    if not isinstance(indices, Var):
        raise RuntimeError("parameter indices is not valid")
    if indices.dtype != _F.int:
        raise RuntimeError("parameter indices must be int type")
    return _F.gather(params, indices)
def fill(dims, value):
    dims = _to_var(dims, to_float=False)
    value = _to_var(value)
    if not isinstance(dims, Var):
        raise RuntimeError("parameter dims is not valid")
    if not isinstance(value, Var):
        raise RuntimeError("parameter value is not valid")
    if dims.dtype != _F.int:
        raise RuntimeError("parameter dims must be int type")
    if len(value.shape) != 0:
        raise RuntimeError("parameter value must be 0-D")
    return _F.fill(dims, value)
def tile(input, multiples):
    input = _to_var(input)
    multiples = _to_var(multiples, to_float=False)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    if not isinstance(multiples, Var):
        raise RuntimeError("parameter multiples is not valid")
    if multiples.dtype != _F.int or len(multiples.shape) != 1:
        raise RuntimeError("parameter multiples must be 1-D int type")
    if len(input.shape) != multiples.shape[-1]:
        raise RuntimeError("parameter multiples's length must match w/ number of dimensions of input")
    return _F.tile(input, multiples)
def shape(input):
    input = _to_var(input)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    return _F.shape(input)
def softplus(features):
    features = _to_var(features)
    if not isinstance(features, Var):
        raise RuntimeError("parameter features is not valid")
    return _F.softplus(features)
def softsign(features):
    features = _to_var(features)
    if not isinstance(features, Var):
        raise RuntimeError("parameter features is not valid")
    return _F.softsign(features)
def stack(values, axis=0):
    if not isinstance(values, (list, tuple)):
        raise RuntimeError("parameter values must be a list/tuple of MNN Var")
    if len(values) < 1:
        raise RuntimeError("parameter values must have at least one item")
    for value in values:
        if not isinstance(value, Var):
            raise RuntimeError("all items in parameter values must be MNN Var type")
        if value.shape != values[0].shape or value.dtype != values[0].dtype:
            raise RuntimeError("all items in parameter values must have same shape and dtype")   
    return _F.stack(values, axis)
def slice(input, starts, sizes):
    input = _to_var(input)
    starts = _to_var(starts, to_float=False)
    sizes = _to_var(sizes, to_float=False)
    if not isinstance(input, Var):
        raise RuntimeError("parameter input is not valid")
    if not isinstance(starts, Var):
        raise RuntimeError("parameter starts is not valid")
    if not isinstance(sizes, Var):
        raise RuntimeError("parameter sizes is not valid")
    if starts.dtype != _F.int or sizes.dtype != _F.int:
        raise RuntimeError("parameter starts/sizes must be int type")
    return _F.slice(input, starts, sizes)
def transpose(x, perm):
    x = _to_var(x)
    perm = _to_var(perm, to_float=False)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    if not isinstance(perm, Var):
        raise RuntimeError("parameter perm is not valid")
    if perm.dtype != _F.int:
        raise RuntimeError("parameter perm must be int type")
    if len(perm.shape) != 1 or perm.shape[-1] != len(x.shape):
        raise RuntimeError("parameter perm must be 1-D, and lenth match the number of dimensions of parameter x")
    return _F.transpose(x, perm)
def pad(x, paddings, mode=CONSTANT):
    x = _to_var(x)
    paddings = _to_var(paddings, to_float=False)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    if not isinstance(paddings, Var):
        raise RuntimeError("parameter paddings is not valid")
    if paddings.dtype != _F.int:
        raise RuntimeError("parameter perm must be int type")
    if len(paddings.shape) != 2 or paddings.shape[-1] != 2 or paddings.shape[0] != len(x.shape):
        raise RuntimeError("parameter paddings must be 2-D w/ shape[n, 2], and n is the number of dimensions of parameter x")
    return _F.pad(x, paddings, mode)
def resize(images, x_scale, y_scale):
    images = _to_var(images)
    if not isinstance(images, Var):
        raise RuntimeError("parameter images is not valid")
    if len(images.shape) != 4 or images.data_format != _F.NC4HW4:
        raise RuntimeError("parameter images must be 4-D NC4HW4 format")
    return _F.resize(images, x_scale, y_scale)
def crop(images, size, axis, offset):
    images = _to_var(images)
    size = _to_var(size)
    if not isinstance(images, Var):
        raise RuntimeError("parameter images is not valid")
    if not isinstance(size, Var):
        raise RuntimeError("parameter size is not valid")
    if len(images.shape) != 4 or images.data_format != _F.NC4HW4:
        raise RuntimeError("parameter images must be 4-D NC4HW4 format")
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
    return _F.crop(images, size, axis, offset)
def crop_and_resize(image, boxes, box_ind, crop_size, method=BILINEAR, extrapolation_value=0.):
    image = _to_var(image)
    boxes = _to_var(boxes, to_float=False)
    box_ind = _to_var(box_ind, to_float=False)
    crop_size = _to_var(crop_size, to_float=False)
    if not isinstance(image, Var):
        raise RuntimeError("parameter image is not valid")
    if not isinstance(boxes, Var):
        raise RuntimeError("parameter boxes is not valid")
    if not isinstance(box_ind, Var):
        raise RuntimeError("parameter box_ind is not valid")
    if not isinstance(crop_size, Var):
        raise RuntimeError("parameter crop_size is not valid")
    if len(image.shape) != 4:
        raise RuntimeError("parameter image must be 4-D format")
    if boxes.dtype != _F.int or box_ind.dtype != _F.int or crop_size.dtype != _F.int:
        raise RuntimeError("parameter boxes/box_ind/crop_size must be int type")
    if len(boxes.shape) != 2 or boxes.shape[-1] !=4:
        raise RuntimeError("parameter boxes must be 2-D w/ shape [num_boxes, 4]")
    if len(box_ind.shape) != 1 or box_ind.shape[-1] != boxes.shape[0]:
        raise RuntimeError("parameter boxes must be 1-D w/ shape [num_boxes]")
    if len(crop_size.shape) != 1 or crop_size.shape[0] != 2:
        raise RuntimeError("parameter boxes must be 1-D w/ shape [2]")
    return _F.crop_and_resize(image, boxes, box_ind, crop_size, method, extrapolation_value)
def reverse_sequence(x, y, batch_dim, seq_dim):
    x = _to_var(x)
    y = _to_var(y, to_float=False)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    if not isinstance(y, Var):
        raise RuntimeError("parameter y is not valid")
    if y.dtype != _F.int or len(y.shape) != 1:
        raise RuntimeError("parameter y must be 1-D int type")
    if batch_dim < 0 or batch_dim >= len(x.shape):
        raise RuntimeError("parameter batch_dim must be in range of the number of dimensions of parameter x")
    if seq_dim < 0 or seq_dim >= len(x.shape):
        raise RuntimeError("parameter seq_dim must be in range of the number of dimensions of parameter x")
    if y.shape[0] != x.shape[batch_dim]:
        raise RuntimeError("parameter y must be shape [x.shape[batch.dim]]")
    return _F.reverse_sequence(x, y, batch_dim, seq_dim)
def reshape(x, shape, original_format=NCHW):
    x = _to_var(x)
    if not isinstance(x, Var):
        raise RuntimeError("parameter x is not valid")
    if not isinstance(shape, (list, tuple)):
        raise RuntimeError("parameter shape is not valid")
    new_length = 1
    skip = False 
    for value in shape:
        if value < 0:
            skip = True
        new_length *= value
         
    if new_length != x.size and not skip:
        raise RuntimeError("parameter shape is not valid")
    return _F.reshape(x, shape, original_format) 
