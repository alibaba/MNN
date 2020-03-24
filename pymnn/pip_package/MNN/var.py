import _mnncengine.expr as F
def float(value):
    res = F.placeholder([], F.NCHW, F.float)
    res.write([value])
    res.fix(F.Const)
    return res
def int(value):
    res = F.placeholder([], F.NCHW, F.int)
    res.write([value])
    res.fix(F.Const)
    return res

