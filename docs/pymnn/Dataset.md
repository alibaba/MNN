## data.Dataset

```python
class Dataset
```
Dataset是一个虚基类, 用户实现自己的Dataset需要继承基类并重写`__getitem__`和`__len__`方法

*注意到`__getitem__`方法需要返回两个变量,一个为输入,另一个为目标*

示例：

```python
try:
    import mnist
except ImportError:
    print("please 'pip install mnist' before run this demo")
import MNN
import MNN.expr as expr
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
        dv = expr.const(self.data[index].flatten().tolist(), [1, 28, 28], expr.NCHW)
        dl = expr.const([self.labels[index]], [], expr.NCHW, expr.uint8)
        # first for inputs, and may have many inputs, so it's a list
        # second for targets, also, there may be more than one targets
        return [dv], [dl]
    def __len__(self):
        # size of the dataset
        if self.is_training_dataset:
            return 60000
        else:
            return 10000
```