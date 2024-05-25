import MNN.numpy as np
import MNN
import sys
nn = MNN.nn
F = MNN.expr
F.lazy_eval(True)
F.set_lazy_mode(1)

opt = MNN.optim.Grad()

vars = F.load_as_dict(sys.argv[1])
output = vars['loss']
parameters = [vars['weight']]
rgbdiff = F.placeholder(output.shape, output.data_format, output.dtype)
rgbdiff.name = 'loss_diff'
rgbdiff.write([1.0])
rgbdiff.fix_as_const()

parameters, grad = opt.grad([output], [rgbdiff], parameters)
for i in range(0, len(parameters)):
    grad[i].name = 'grad::' + parameters[i].name
F.save(grad, sys.argv[2])
