import MNN
import MNN.var as var
c_train = MNN.c_train
nn = c_train.cnn
F = MNN.expr
data = c_train.data
import time

import sys
class Net(MNN.train.Module):
    def __init__(self):
        super(Net, self).__init__()
        modelFile = sys.argv[1]
        print(modelFile)
        varMap = F.load_dict(modelFile)
        inputVar = varMap['input']
        outputVar = varMap['MobilenetV2/Logits/AvgPool']
        self.net = c_train.load_module([inputVar], [outputVar], True)
        self.fc = nn.conv(1280, 4, [1, 1])
    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        x = F.softmax(F.reshape(F.convert(x, F.NCHW), [0, -1]))
        return x

scale = [0.00784314, 0.00784314, 0.00784314, 0.00784314]
mean = [127.5, 127.5, 127.5, 0]

imageConfig = data.image.config(MNN.cv.BGR, 224, 224, scale, mean, [1.0, 1.0], False)
picturePath = sys.argv[2]
print(picturePath)
txtPath = sys.argv[3]
imageDataset = data.image.image_label(picturePath, txtPath, imageConfig, False)
imageLoader = imageDataset.create_loader(10, True, True, 0)

def trainFunc(loader, net, opt):
    loader.reset()
    net.train(True)
    t0 = time.time()
    iter_number = loader.iter_number()
    for i in range(0, iter_number):
        example = loader.next()[0]
        data = example[0][0]
        label = F.reshape(example[1][0], [-1])
        data = F.convert(data, F.NC4HW4)
        predict = net(data)
        target = F.one_hot(F.cast(label, F.int), var.int(4), var.float(1.0), var.float(0.0))
        loss = c_train.loss.CrossEntropy(predict, target)
        if i % 10 == 0:
            print(i, loss.read(), iter_number)
        opt.step(loss)
    t1 = time.time()
    cost = t1 - t0
    print("Epoch cost: %.3f" %cost)
    F.save(net.parameters(), "cache/temp.snapshot")

def testFunc(loader, net):
    loader.reset()
    net.train(False)
    iter_number = loader.iter_number()
    correct = 0
    for i in range(0, iter_number):
        example = loader.next()[0]
        data = example[0][0]
        label = F.reshape(example[1][0], [-1])
        data = F.convert(data, F.NC4HW4)
        predict = net(data)
        predict = F.argmax(predict, 1)
        accu = F.reduce_sum(F.equal(predict, F.cast(label, F.int)), [], False)
        correct += accu.read()[0]
    print("Accu: ", correct * 100.0 / loader.size(), "%")

net = Net()
net.loadParameters(F.load("cache/temp.snapshot"))
opt = c_train.SGD(0.0001, 0.9);
opt.append(net.parameters())
F.setThreadNumber(4)
testTxt = sys.argv[4]
testDataset = data.image.image_label(picturePath, testTxt, imageConfig, False)
testLoader = testDataset.create_loader(10, True, False, 0)

for epoch in range(0, 10):
    testFunc(testLoader, net)
    trainFunc(imageLoader, net, opt)

