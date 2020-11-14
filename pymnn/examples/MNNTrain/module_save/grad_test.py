import numpy as np
import MNN
nn = MNN.nn
F = MNN.expr

v0 = F.const([0.3,0.1, -0.3,0.4], [4])
v2 = F.const([0.3,0.1, -0.3,0.4], [4])
v1 = v0 * v0

outputDiff = F.const([0.05, 0.03, 0.02, 0.01], [4])

v0Grad = nn.grad(v1, [v0, v2], [outputDiff], "")
print(v0Grad)
print(v0Grad[0].read())
F.save(v0Grad, "temp.grad")
