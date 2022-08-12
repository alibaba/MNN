from __future__ import print_function
import time
import numpy as np
import MNN
from dataset import MnistDataset
nn = MNN.nn
F = MNN.expr

# open lazy evaluation for train
F.lazy_eval(True)

class Net(nn.Module):
    """construct a lenet 5 model"""
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
        # MNN use NC4HW4 format for convs, so we need to convert it to NCHW before entering other ops
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x


def test_func(net, test_dataloader):
    """test function"""
    net.train(False)
    test_dataloader.reset()

    correct = 0
    for i in range(test_dataloader.iter_number):
        example = test_dataloader.next()
        input_data = example[0]
        output_target = example[1]
        data = input_data[0]  # which input, model may have more than one inputs
        label = output_target[0]  # also, model may have more than one outputs

        predict = net.forward(data)
        predict = F.argmax(predict, 1)
        predict = np.array(predict.read())
        label = np.array(label.read())
        correct += (np.sum(label == predict))

    print("test acc: ", correct * 100.0 / test_dataloader.size, "%")

def train_func(net, train_dataloader, opt):
    """train function"""
    net.train(True)
    train_dataloader.reset()

    t0 = time.time()
    for i in range(train_dataloader.iter_number):
        example = train_dataloader.next()
        input_data = example[0]
        output_target = example[1]
        data = input_data[0]  # which input, model may have more than one inputs
        label = output_target[0]  # also, model may have more than one outputs

        predict = net.forward(data)
        target = F.one_hot(F.cast(label, F.int), 10, 1, 0)
        loss = nn.loss.cross_entropy(predict, target)
        opt.step(loss)
        if i % 100 == 0:
            print("train loss: ", loss.read())

    t1 = time.time()
    cost = t1 - t0
    print("Epoch cost: %.3f s." %cost)
    F.save(net.parameters, "temp.mnist.snapshot")


def learning_rate_scheduler(lr, epoch):
    if (epoch + 1) % 2 == 0:
        lr *= 0.1
    return lr


def demo():
    """
    demo for MNIST handwritten digits recognition
    """

    model = Net()

    train_dataset = MnistDataset(True)
    test_dataset = MnistDataset(False)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_dataloader = MNN.data.DataLoader(test_dataset, batch_size = 100, shuffle = False)

    opt = MNN.optim.SGD(model, 0.01, 0.9, 0.0005)

    F.set_thread_number(4)
    for epoch in range(0, 1):
        opt.learning_rate = learning_rate_scheduler(opt.learning_rate, epoch)
        train_func(model, train_dataloader, opt)
        
        # save model
        file_name = '%d.mnist.mnn' %epoch
        model.train(False)
        predict = model.forward(F.placeholder([1, 1, 28, 28], F.NC4HW4))
        print("Save to " + file_name)
        F.save([predict], file_name)
    
        test_func(model, test_dataloader)


if __name__ == "__main__":
    demo()
