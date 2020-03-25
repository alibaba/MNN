import MNN
import MNN.var as var
c_train = MNN.c_train
nn = c_train.cnn
F = MNN.expr
data = c_train.data
import time

import sys
modelFile = sys.argv[1]
print(modelFile)

varMap = F.load_dict(modelFile)
inputVar = varMap['sub_7']
outputVar = varMap['ResizeBilinear_3']
net = c_train.load_module([inputVar], [outputVar], True)
c_train.compress.quantize(net, 8,  c_train.compress.PerChannel, c_train.compress.MovingAverage)
checkNet = c_train.load_module([inputVar], [outputVar], False)

scale = [0.00784314, 0.00784314, 0.00784314, 0.00784314]
mean = [127.5, 127.5, 127.5, 0]

imageConfig = data.image.config(MNN.cv.BGR, 257, 257, scale, mean, [1.0, 1.0], False)
picturePath = sys.argv[2]
print(picturePath)
imageDataset = data.image.image_no_label(picturePath, imageConfig)
imageLoader = imageDataset.create_loader(5, True, True, 0)

def trainFunc(loader, net, checkNet, opt):
    loader.reset()
    net.train(True)
    t0 = time.time()
    iter_number = loader.iter_number()
    for i in range(0, iter_number):
        example = loader.next()[0]
        data = example[0][0]
        data = F.convert(data, F.NC4HW4)
        p0 = net(data)
        p1 = checkNet(data)
        p0 = F.reshape(F.convert(p0, F.NCHW), [0, -1])
        p1 = F.reshape(F.convert(p1, F.NCHW), [0, -1])
        loss = c_train.loss.MSE(p0, p1)
        opt.step(loss)
        if i % 10 == 0:
            print(loss.read())
    t1 = time.time()
    cost = t1 - t0
    print("Epoch cost: %.3f" %cost)
    F.save(net.parameters(), "cache/temp.snapshot")


opt = c_train.SGD(0.000000000, 0.9);
opt.append(net.parameters())

for epoch in range(0, 1):
    trainFunc(imageLoader, net, checkNet, opt)

net.train(False)
testInput = F.placeholder([1, 3, 257, 257], F.NC4HW4)
testInput.set_name("data")
testOutput = net(testInput)
testOutput.set_name("prob");
quanName = "temp.quan.mnn"
print("Save to " + quanName)
F.save([testOutput], quanName)
