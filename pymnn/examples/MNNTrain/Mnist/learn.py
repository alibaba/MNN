import MNN
import MNN.var as var
c_train = MNN.c_train
nn = c_train.cnn
F = MNN.expr
data = c_train.data
import time

class Net(MNN.train.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.conv(1, 20, [5, 5])
        self.conv2 = nn.conv(20, 50, [5, 5])
        self.fc1 = nn.linear(800, 500)
        self.fc2 = nn.linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.conv2(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x



def testFunc(loader, net):
    loader.reset()
    net.train(False)
    iter_number = loader.iter_number()
    correct = 0
    for i in range(0, iter_number):
        example = loader.next()[0]
        data = example[0][0]
        label = example[1][0]

        data = F.cast(data, F.float) * var.float(1.0/255.0)
        predict = net(data)
        predict = F.argmax(predict, 1)
        accu = F.reduce_sum(F.equal(predict, F.cast(label, F.int)), [], False)
        correct += accu.read()[0]
    print("Accu: ", correct * 100.0 / loader.size(), "%")


def trainFunc(loader, net, opt):
    loader.reset()
    net.train()
    t0 = time.time()
    iter_number = loader.iter_number()
    for i in range(0, iter_number):
        example = loader.next()[0]
        data = example[0][0]
        label = example[1][0]

        data = F.cast(data, F.float) * var.float(1.0/255.0)
        predict = net(data)
        target = F.one_hot(F.cast(label, F.int), var.int(10), var.float(1.0), var.float(0.0))
        loss = c_train.loss.CrossEntropy(predict, target)
        opt.step(loss)
        if i % 100 == 0:
            print(loss.read())
    t1 = time.time()
    cost = t1 - t0
    print("Epoch cost: %.3f" %cost)
    F.save(net.parameters(), "cache/temp.snapshot")


net = Net()
opt = c_train.SGD(0.01, 0.9)
net.loadParameters(F.load("cache/temp.snapshot"))
opt.append(net.parameters())

import sys
mnistDataPath = sys.argv[1]

mnistDataset = data.mnist.create(mnistDataPath, data.mnist.Train)
trainLoader = mnistDataset.create_loader(64, True, True, 0)
testmnistDataset = data.mnist.create(mnistDataPath, data.mnist.Test)
testLoader = mnistDataset.create_loader(10, True, False, 0)

F.setThreadNumber(4)
for epoch in range(0, 10):
    trainFunc(trainLoader, net, opt)
    # Save Model
    fileName = 'cache/%d.mnist.mnn' %epoch
    net.train(False)
    predict = net.forward(F.placeholder([1, 1, 28, 28], F.NC4HW4))
    print("Save to " + fileName)
    F.save([predict], fileName)
    testFunc(testLoader, net)
