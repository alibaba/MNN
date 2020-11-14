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
        if not isinstance(existing, type(object.__lt__)) and not isinstance(existing, type(object.__repr__)):
            raise ValueError("operator %s cannot be overwritten again on class %s." %(operator, class_object))
    setattr(class_object, operator, func)
def _match_data(dtype, value):
    if dtype in {expr.dtype.double, expr.dtype.float}:
        return float(value)
    elif dtype in {expr.dtype.int, expr.dtype.int64, expr.dtype.uint8}:
        return int(value)
    else:
        raise RuntimeError("not supported type")
   
def _unify_other(self, other):
    """unify the self, other output to same format"""
    other_is_var = isinstance(other, expr.Var)
    if other_is_var:
       pass
    else:
       if not isinstance(other, int) and not isinstance(other, float) and not isinstance(other, bool):
           raise RuntimeError("not supported type")
       dtype = self.dtype
       other = _match_data(dtype, other)
       other = [other]
       other = expr.const(other, [], dtype=dtype)
    return other
def _add(self, other):
    other = _unify_other(self, other)
    return expr.add(self, other)
def _radd(self, other):
    other = _unify_other(self, other)
    return expr.add(other, self)
def _sub(self, other):
    other = _unify_other(self, other)
    return expr.subtract(self, other)
def _rsub(self, other):
    other = _unify_other(self, other)
    return expr.subtract(other, self)
def _multiply(self, other):
    other = _unify_other(self, other)
    return expr.multiply(self, other)
def _rmultiply(self, other):
    other = _unify_other(self, other)
    return expr.multiply(other, self)
def _truediv(self, other):
    other = _unify_other(self, other)
    return expr.divide(self, other)
def _rtruediv(self, other):
    other = _unify_other(self, other)
    return expr.divide(other, self)
def _floordiv(self, other):
    other = _unify_other(self, other)
    return expr.floordiv(self, other)
def _rfloordiv(self, other):
    other = _unify_other(self, other)
    return expr.floordiv(other, self)
def _floormod(self, other):
    other = _unify_other(self, other)
    return expr.floor_mod(self, other)
def _rfloormod(self, other):
    other = _unify_other(self, other)
    return expr.floor_mod(other, self)
def _pow(self, other):
    other = _unify_other(self, other)
    return expr.pow(self, other)
def _rpow(self, other):
    other = _unify_other(self, other)
    return expr.pow(other, self)
def _eq(self, other):
    other = _unify_other(self, other)
    return expr.equal(self, other)
def _req(self, other):
    other = _unify_other(self, other)
    return expr.equal(other, self)
def _ne(self, other):
    other = _unify_other(self, other)
    return expr.not_equal(self, other)
def _rne(self, other):
    other = _unify_other(self, other)
    return expr.not_equal(other, self)
def _ge(self, other):
    other = _unify_other(self, other)
    return expr.greater_equal(self, other)
def _rge(self, other):
    other = _unify_other(self, other)
    return expr.greater_equal(other, self)
def _gt(self, other):
    other = _unify_other(self, other)
    return expr.greater(self, other)
def _rgt(self, other):
    other = _unify_other(self, other)
    return expr.greater(other, self)
def _le(self, other):
    other = _unify_other(self, other)
    return expr.less_equal(self, other)
def _rle(self, other):
    other = _unify_other(self, other)
    return expr.less_equal(other, self)
def _lt(self, other):
    other = _unify_other(self, other)
    return expr.less(self, other)
def _rlt(self, other):
    other = _unify_other(self, other)
    return expr.less(other, self)
def _abs(self):
    return expr.abs(self)
def _neg(self):
    return expr.negative(self)
def _read(self):
    return self.read().__repr__() 
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
_override_operator(expr.Var, "__repr__", _read)
_override_operator(expr.Var, "__getitem__", _slice_helper)
_override_operator(expr.Var, "__add__", _add)
_override_operator(expr.Var, "__radd__", _radd)
_override_operator(expr.Var, "__sub__", _sub)
_override_operator(expr.Var, "__rsub__", _rsub)
_override_operator(expr.Var, "__mul__", _multiply)
_override_operator(expr.Var, "__rmul__", _rmultiply)
_override_operator(expr.Var, "__truediv__", _truediv)
_override_operator(expr.Var, "__rtruediv__", _rtruediv)
_override_operator(expr.Var, "__floordiv__", _floordiv)
_override_operator(expr.Var, "__rfloordiv__", _rfloordiv)
_override_operator(expr.Var, "__floormod__", _floormod)
_override_operator(expr.Var, "__rfloormod__", _rfloormod)
_override_operator(expr.Var, "__pow__", _pow)
_override_operator(expr.Var, "__rpow__", _rpow)
_override_operator(expr.Var, "__eq__", _eq)
_override_operator(expr.Var, "__req__", _req)
_override_operator(expr.Var, "__ne__", _ne)
_override_operator(expr.Var, "__rne__", _rne)
_override_operator(expr.Var, "__ge__", _ge)
_override_operator(expr.Var, "__rge__", _rge)
_override_operator(expr.Var, "__gt__", _gt)
_override_operator(expr.Var, "__rgt__", _rgt)
_override_operator(expr.Var, "__le__", _le)
_override_operator(expr.Var, "__rle__", _rle)
_override_operator(expr.Var, "__lt__", _lt)
_override_operator(expr.Var, "__rlt__", _rlt)
_override_operator(expr.Var, "__abs__", _abs)
_override_operator(expr.Var, "__neg__", _neg)
