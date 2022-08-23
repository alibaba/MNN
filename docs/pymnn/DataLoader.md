## data.DataLoader
```python
class DataSet
```
DataLoader数据加载器，支持数据批处理和随机采样

---
### `DataLoader(dataset, batch_size, shuffle, num_workers)`
创建一个DataLoader

参数：
- `dataset:DataSet` 数据集实例
- `batch_size:int` 批处理大小
- `shuffle:bool` 打乱数据集标记，默认为True
- `num_workers:int` 线程数，默认为0

返回：数据加载器

返回类型：`DataLoader`

---
### `iter_number`

返回总迭代次数，当剩余的数据在一个批次大小中没有满仍然会被加载

属性类型：只读

类型：`int`

---
### `size`

获取数据集大小

属性类型：只读

类型：`int`

---
### `reset()`

重置数据加载器，数据加载器每次用完后都需要重置

返回：`None`

返回类型：`None`

---
### `next()`

在数据集中获取批量数据

返回：`([Var], [Var])` 两组数据，第一组为输入数据，第二组为结果数据

返回类型：`tuple`

示例：

```python
train_dataset = MnistDataset(True)
test_dataset = MnistDataset(False)
train_dataloader = data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_dataloader = data.DataLoader(test_dataset, batch_size = 100, shuffle = False)
...
# use in training
def train_func(net, train_dataloader, opt):
    """train function"""
    net.train(True)
    # need to reset when the data loader exhausted
    train_dataloader.reset()
    t0 = time.time()
    for i in range(train_dataloader.iter_number):
        example = train_dataloader.next()
        input_data = example[0]
        output_target = example[1]
        data = input_data[0]  # which input, model may have more than one inputs
        label = output_target[0]  # also, model may have more than one outputs
        predict = net.forward(data)
        target = expr.one_hot(expr.cast(label, expr.int), 10, 1, 0)
        loss = nn.loss.cross_entropy(predict, target)
        opt.step(loss)
        if i % 100 == 0:
            print("train loss: ", loss.read())
```