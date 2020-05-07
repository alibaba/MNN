try:
    import mnist
except ImportError:
    print("please 'pip install mnist' before run this demo")

import numpy as np
import MNN
F = MNN.expr


class MnistDataset(MNN.data.Dataset):
    def __init__(self, training_dataset=True):
        super(MnistDataset, self).__init__()
        self.is_training_dataset = training_dataset
        if self.is_training_dataset:
            self.data = mnist.train_images() / 255.0
            self.labels = mnist.train_labels()
        else:
            self.data = mnist.test_images() / 255.0
            self.labels = mnist.test_labels()

    def __getitem__(self, index):
        dv = F.const(self.data[index].flatten().tolist(), [1, 28, 28], F.data_format.NCHW)
        dl = F.const([self.labels[index]], [], F.data_format.NCHW, F.dtype.uint8)
        # first for inputs, and may have many inputs, so it's a list
        # second for targets, also, there may be more than one targets
        return [dv], [dl]

    def __len__(self):
        # size of the dataset
        if self.is_training_dataset:
            return 60000
        else:
            return 10000
