import numpy as np
import MNN
nn = MNN.nn
F = MNN.expr

class Net(nn.Module):
    """construct a lenet 5 model"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.conv(1, 20, [5, 5])
        self.conv2 = nn.conv(20, 50, [5, 5])
        self.fc1 = nn.linear(800, 500)
        self.fc2 = nn.linear(500, 10)
        self.step = F.const([10], [], F.NCHW, F.int)
        self.lr = F.const([0.0004],[], F.NCHW, F.float)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.conv2(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.reshape(x, [0, -1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x


model = Net()
F.save(model.parameters, 'mnist.snapshot')


model2 = Net()
model2.load_parameters(F.load_as_list('mnist.snapshot'))

print(model2.lr.read())
print(model2.step.read())
