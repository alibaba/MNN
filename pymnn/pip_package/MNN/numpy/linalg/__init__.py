import MNN.expr as _F

# Linear algebra
def norm(x, ord=None, axis=None, keepdims=False):
    from MNN import numpy as np
    x = np.array(x, dtype=np.float32)
    ndim = x.ndim
    if axis is None:
        if ((ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
            (ord == 2 and ndim == 1)):
            x = x.ravel()
            sqnorm = np.dot(x, x)
            ret = np.sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim*[1])
            return ret
    if axis is None:
        axis = tuple(range(ndim))
    else:
        axis = _F._to_axis(axis)
    if len(axis) == 1:
        if ord == np.inf:
            return np.max(np.abs(x), axis=axis, keepdims=keepdims)
        elif ord == -np.inf:
            return np.min(np.abs(x), axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            return np.sum(x, axis=axis, keepdims=keepdims)
        elif ord == 1:
            # special case for speedup
            return np.sum(np.abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            # s = (x.conj() * x).real
            return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
        # None of the str-type keywords for ord ('fro', 'nuc')
        # are valid for vectors
        elif isinstance(ord, str):
            raise ValueError('Invalid norm order for vectors')
        else:
            ord = float(ord)
            return np.sum(np.abs(x)**ord, axis=axis, keepdims=keepdims)**(1./ord)

def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    w, u, vt = _F.svd(a)
    return (u, w, vt)

def solve(a, b):
    import _mnncengine.cv as _cv
    return _cv.solve(a, b)[1]