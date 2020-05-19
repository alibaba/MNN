_Slice = slice
_Int = int
_newaxis = None
from _mnncengine import *
from . import data
from . import expr
from . import nn
from . import optim
from . import tools

def _check_index(idx):
    """Check if a given value is a valid index."""
    return isinstance(idx, _Int)
def _override_operator(class_object, operator, func):
    """Overrides operator on class_object to call func.

    Args:
    class_object: the class to override for; for example Var.
    operator: the string name of the operator to override.
    func: the function that replaces the overridden operator.

    Raises:
    ValueError: If operator has already been overwritten,
      or if operator is not allowed to be overwritten.
    """
    existing = getattr(class_object, operator, None)
    if existing is not None:
        # Check to see if this is a default method-wrapper or slot wrapper which
        # will be true for the comparison operators.
        if not isinstance(existing, type(object.__lt__)):
            raise ValueError("operator %s cannot be overwritten again on class %s." %(operator, class_object))
    setattr(class_object, operator, func)
def _slice_helper(input, slice_spec):
    if not isinstance(slice_spec, (list, tuple)):
        slice_spec = [slice_spec]

    begin, end, strides = [], [], []
    index = 0

    new_axis_mask, shrink_axis_mask = 0, 0
    begin_mask, end_mask = 0, 0
    ellipsis_mask = 0
    for s in slice_spec:
        if isinstance(s, _Slice):
            if s.start is not None:
                _check_index(s.start)
                begin.append(s.start)
            else:
                begin.append(0)
                begin_mask |= (1 << index)
            if s.stop is not None:
                _check_index(s.stop)
                end.append(s.stop)
            else:
                end.append(0)
                end_mask |= (1 << index)
            if s.step is not None:
                _check_index(s.step)
                strides.append(s.step)
            else:
                strides.append(1)
        elif s is Ellipsis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            ellipsis_mask |= (1 << index)
        elif s is _newaxis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            new_axis_mask |= (1 << index)
        else:
            _check_index(s)
            begin.append(s)
            end.append(s + 1)
            strides.append(1)
            shrink_axis_mask |= (1 << index)
        index += 1
    var_begin = expr.const(value_list=begin, dtype=expr.dtype.int, shape=[len(begin)])
    var_end = expr.const(value_list=end, dtype=expr.dtype.int, shape=[len(begin)])
    var_strides = expr.const(value_list=strides, dtype=expr.dtype.int, shape=[len(begin)])
    return expr.strided_slice(input, var_begin, var_end, var_strides,\
          begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
_override_operator(expr.Var, "__getitem__", _slice_helper)
