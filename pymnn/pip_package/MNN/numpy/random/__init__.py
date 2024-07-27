import MNN.expr as _F

# Random sampling
def random(shape):
    return _F.randomuniform(shape, _F.float)
def rand(*args):
    return random(args)
randn = rand
def randint(low, high=None, size=None, dtype=_F.int):
    if type(low) in (list, tuple) or type(high) in (list, tuple):
        raise ValueError('MNN.numpy randint just support low/high is int.')
    if high is None:
        high = low
        low = 0
    if size is None:
        size = [1]
    return _F.cast(_F.randomuniform(size, _F.float, low, high), dtype)