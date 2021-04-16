from __future__ import print_function
import time
import argparse
import numpy as np
import MNN
from imagenet_dataset import ImagenetDataset
nn = MNN.nn
F = MNN.expr


def test_func(net, test_dataloader):
    net.train(False)
    test_dataloader.reset()

    correct = 0
    total = 0
    for i in range(test_dataloader.iter_number):
        example = test_dataloader.next()
        input_data = example[0]
        output_target = example[1]
        data = input_data[0]  # which input, model may have more than one inputs
        label = output_target[0]  # also, model may have more than one outputs

        predict = net(data)
        predict = F.argmax(predict, 1)
        predict = np.array(predict.read())
        label = np.array(label.read())
        correct += (np.sum(label == predict))
        total += label.size

        if (i+1) % 10 == 0:
            print("test iteration", i+1, ", accuracy: ", correct / total * 100, "%")

    print("test acc: ", correct * 100.0 / test_dataloader.size, "%")


def train_func(net, train_dataloader, opt, num_classes):
    net.train(True)
    train_dataloader.reset()

    t0 = time.time()
    # for i in range(train_dataloader.iter_number):
    for i in range(100): # actually, in our full experiment, we only need 3K images using ILSVRC2012 training dataset
        example = train_dataloader.next()
        input_data = example[0]
        output_target = example[1]
        data = input_data[0]  # which input, model may have more than one inputs
        label = output_target[0]  # also, model may have more than one outputs

        predict = net.forward(data)
        target = F.one_hot(F.cast(label, F.int), num_classes, 1, 0)
        loss = nn.loss.cross_entropy(predict, target)
        opt.step(loss)

        if i % 10 == 0:
            print("train loss: ", loss.read())

    t1 = time.time()
    cost = t1 - t0
    print("Epoch cost: %.3f s." % cost)
    F.save(net.parameters, "temp.mobilenet.snapshot")


def demo():
    '''
    demo for quant-aware-training using tf mobilenet v2.
    the dataset used is the ILSVRC2012 validation dataset which has 50000 images
    10000 for training (actually we only need 3K in our standard experiment using ILSVRC2012 training dataset)
    40000 for testing
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True,\
        help="mobilenet MNN model file")
    parser.add_argument("--val_image_path", type=str, required=True,\
        help="path to ILSVRC2012 val images")
    parser.add_argument("--val_txt", type=str, required=True,\
                        help="path to ILSVRC2012 val.txt")

    args = parser.parse_args()

    model_file = args.model_file
    image_path = args.val_image_path
    val_txt = args.val_txt

    train_dataset = ImagenetDataset(image_path, val_txt, True)
    test_dataset = ImagenetDataset(image_path, val_txt, False)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = MNN.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    m = F.load_as_dict(model_file)

    inputs_outputs = F.get_inputs_and_outputs(m)
    for key in inputs_outputs[0].keys():
        print('input names:\t', key)
    for key in inputs_outputs[1].keys():
        print('output names:\t', key)

    # get inputs and outputs
    inputs = [m['input']]
    outputs = [m['MobilenetV2/Predictions/Reshape_1']]

    net = nn.load_module(inputs, outputs, True)

    # turn net to quant-aware-training module
    nn.compress.train_quant(net, quant_bits=8)

    opt = MNN.optim.SGD(net, 1e-5, 0.9, 0.00004)

    num_classes = 1001

    for epoch in range(5):
        train_func(net, train_dataloader, opt, num_classes)

        # save model
        file_name = '%d.mobilenet.mnn' % epoch
        net.train(False)
        predict = net.forward(F.placeholder([1, 3, 224, 224], F.NC4HW4))
        print("Save to " + file_name)
        F.save([predict], file_name)

        test_func(net, test_dataloader)


if __name__ == "__main__":
    demo()
