from __future__ import print_function
import time
import argparse
import numpy as np
from finetune_dataset import FinetuneDataset
import MNN
nn = MNN.nn
F = MNN.expr


def load_feature_extractor(model_file):
    var_dict = F.load_as_dict(model_file)
    input_var = var_dict['input']
    output_var = var_dict['MobilenetV2/Logits/AvgPool']
    # 'False' means the parameters int this module will not update during training
    feature_extractor = nn.load_module([input_var], [output_var], False)
    feature_extractor = nn.FixModule(feature_extractor)  # fix feature extractor

    return feature_extractor


class Net(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(Net, self).__init__()
        self.feature_extractor = feature_extractor
        # use conv to implement fc
        self.fc = nn.conv(1280, num_classes, [1, 1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = F.softmax(x)
        return x


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

        predict = net.forward(F.convert(data, F.NC4HW4))
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
    for i in range(train_dataloader.iter_number):
        example = train_dataloader.next()
        input_data = example[0]
        output_target = example[1]
        data = input_data[0]  # which input, model may have more than one inputs
        label = output_target[0]  # also, model may have more than one outputs

        # need to convert data to NC4HW4, because the input format of feature extractor is NC4HW4
        predict = net.forward(F.convert(data, F.NC4HW4))
        target = F.one_hot(F.cast(label, F.int), num_classes, 1, 0)
        loss = nn.loss.cross_entropy(predict, target)
        opt.step(loss)

        if i % 10 == 0:
            print("train loss: ", loss.read())

    t1 = time.time()
    cost = t1 - t0
    print("Epoch cost: %.3f s." % cost)
    F.save(net.parameters, "temp.mobilenet_finetune.snapshot")


def demo():
    """
    demo for finetuning on your own dataset using MobilenetV2 feature extractor
    """
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True,\
                        help="mobilenet MNN model file")
    parser.add_argument("--train_image_folder", type=str, required=True,\
                        help="path to your train images")
    parser.add_argument("--train_txt", type=str, required=True,\
                        help="path to your train images txt, each line should have format 'image_name label'")
    parser.add_argument("--test_image_folder", type=str, required=True,\
                        help="path to your test images")
    parser.add_argument("--test_txt", type=str, required=True,\
                        help="path to your test images txt, each line should have format 'image_name label'")
    parser.add_argument("--num_classes", type=int, required=True,\
                        help="num classes of your dataset")

    args = parser.parse_args()

    model_file = args.model_file
    train_image_folder = args.train_image_folder
    train_txt = args.train_txt
    test_image_folder = args.test_image_folder
    test_txt = args.test_txt
    num_classes = args.num_classes

    train_dataset = FinetuneDataset(train_image_folder, train_txt)
    test_dataset = FinetuneDataset(test_image_folder, test_txt)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = MNN.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    feature_extractor = load_feature_extractor(model_file)

    net = Net(feature_extractor, num_classes)

    opt = MNN.optim.SGD(net, 1e-3, 0.9, 0.00004)

    for epoch in range(10):
        train_func(net, train_dataloader, opt, num_classes)

        # save model
        file_name = '%d.mobilenet_finetune.mnn' % epoch
        net.train(False)
        predict = net.forward(F.placeholder([1, 3, 224, 224], F.NC4HW4))
        print("Save to " + file_name)
        F.save([predict], file_name)

        test_func(net, test_dataloader)


if __name__ == "__main__":
    demo()
