from _mnncengine._expr import *

import _mnncengine._expr as _F

def scalar(value):
    if type(value) == type(1):
        res = _F.const([value], [], _F.NCHW, _F.int)
        return res
    elif type(value) == type(1.):
        res = _F.const([value], [], _F.NCHW, _F.float)
        return res
    else:
        raise NotImplementedError("not supported data type for creating scalar variable")
