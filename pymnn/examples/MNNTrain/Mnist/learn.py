import MNN.train as train
import MNNPy.train
import MNN.train.cnn as nn
import MNN.expr as F
import time
import MNN.train.data as data

class Net(MNNPy.train.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv(1, 20, [5, 5])
        self.conv2 = nn.Conv(20, 50, [5, 5])
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.Relu(self.conv1(x))
        x = F.MaxPool(x, [2, 2], [2, 2])
        x = F.Relu(self.conv2(x))
        x = F.MaxPool(x, [2, 2], [2, 2])
        x = F.Convert(x, F.NCHW)
        x = F.Reshape(x, [0, -1])
        x = F.Relu(self.fc1(x))
        x = self.fc2(x)
        x = F.Softmax(x, 1)
        return x


def initFloat(value):
    res = F.Input([], F.NCHW, F.float)
    res.write([value])
    res.fix(F.Const)
    return res
def initInt(value):
    res = F.Input([], F.NCHW, F.int)
    res.write([value])
    res.fix(F.Const)
    return res

def testFunc(loader, net):
    loader.reset()
    net.train(False)
    iterNumber = loader.iterNumber()
    correct = 0
    for i in range(0, iterNumber):
        example = loader.next()[0]
        data = example[0][0]
        label = example[1][0]

        data = F.Multiply(F.Cast(data, F.float), initFloat(1.0/255.0))
        predict = net(data)
        predict = F.ArgMax(predict, 1)
        accu = F.ReduceSum(F.Equal(predict, F.Cast(label, F.int)), [], False)
        correct += accu.read()[0]
    print(correct * 1.0 / loader.size())


def trainFunc(loader, net, opt):
    loader.reset()
    net.train()
    t0 = time.time()
    iterNumber = loader.iterNumber()
    for i in range(0, iterNumber):
        example = loader.next()[0]
        data = example[0][0]
        label = example[1][0]

        data = F.Multiply(F.Cast(data, F.float), initFloat(1.0/255.0))
        predict = net(data)
        target = F.OneHot(F.Cast(label, F.int), initInt(10), initFloat(1.0), initFloat(0.0))
        loss = train.loss.CrossEntropy(predict, target)
        opt.step(loss)
        if i % 100 == 0:
            print(loss.read())
    t1 = time.time()
    cost = t1 - t0
    print("Epoch cost: %.3f" %cost)
    F.save(net.parameters(), "cache/temp.snapshot")


net = Net()
opt = train.SGD(0.01, 0.9)
net.loadParameters(F.load("cache/temp.snapshot"))
opt.append(net.parameters())

mnistDataset = data.mnist.create("/Users/jiangxiaotang/data/mnist", data.mnist.Train)
trainLoader = mnistDataset.createLoader(64, True, True, 0)
testmnistDataset = data.mnist.create("/Users/jiangxiaotang/data/mnist", data.mnist.Test)
testLoader = mnistDataset.createLoader(10, True, False, 0)

F.setThreadNumber(4)
for epoch in range(0, 10):
    trainFunc(trainLoader, net, opt)
    # Save Model
    fileName = 'cache/%d.mnist.mnn' %epoch
    net.train(False)
    predict = net.forward(F.Input([1, 1, 28, 28], F.NC4HW4))
    print("Save to " + fileName)
    F.save([predict], fileName)
    testFunc(testLoader, net)
