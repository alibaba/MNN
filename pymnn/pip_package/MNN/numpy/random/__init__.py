import MNN.expr as _F
import MNN.numpy as np
# Random sampling
def random(shape):
    return _F.randomuniform(shape, np.float32)
def rand(*args):
    return random(args)
randn = rand
def randint(low, high=None, size=None, dtype=np.int32):
    if type(low) in (list, tuple) or type(high) in (list, tuple):
        raise ValueError('MNN.numpy randint just support low/high is int.')
    if high is None:
        high = low
        low = 0
    if size is None:
        size = [1]
    return _F.randomuniform(size, dtype, low, high)