import MNN
import MNN.var as var
c_train = MNN.c_train
nn = c_train.cnn
F = MNN.expr
data = c_train.data
import time

import sys
# Write your model name
modelFile = "segment.mnn"
print(modelFile)

varMap = F.load_dict(modelFile)
# Create quan module
inputVar = varMap['sub_7']
outputVar = varMap['ResizeBilinear_3']
net = c_train.load_module([inputVar], [outputVar], True)
c_train.compress.quantize(net, 8,  c_train.compress.PerChannel, c_train.compress.MovingAverage)

# Set config for image dataset
scale = [0.00784314, 0.00784314, 0.00784314, 0.00784314]
mean = [127.5, 127.5, 127.5, 0]

imageConfig = data.image.config(MNN.cv.BGR, 257, 257, scale, mean, [1.0, 1.0], False)
picturePath = "../../../data/val_500/imagenet_val_500/"
print(picturePath)
imageDataset = data.image.image_no_label(picturePath, imageConfig)
loader = imageDataset.create_loader(5, True, True, 0)

loader.reset()
net.train(True)
iter_number = loader.iter_number()
for i in range(0, iter_number):
    t0 = time.time()
    example = loader.next()[0]
    data = example[0][0]
    data = F.convert(data, F.NC4HW4)
    p0 = net(data)
    t1 = time.time()
    cost = t1 - t0
    print("Run ", i, " / ", iter_number,' cost:', cost, "s")


net.train(False)
# Set Input
testInput = F.placeholder([1, 3, 257, 257], F.NC4HW4)
testInput.set_name("sub_7")
testOutput = net(testInput)
# Set Output Name
testOutput.set_name("ResizeBilinear_3");
quanName = "temp.quan.mnn"
print("Save to " + quanName)
F.save([testOutput], quanName)
