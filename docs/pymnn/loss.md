## loss

```python
module loss
```
loss模块是模型训练使用的模块，提供了多个损失函数

---
### `cross_entropy(predicts, onehot_targets)`
求交叉熵损失

参数：
- `predicts:Var` 输出层的预测值，`dtype=float`，`shape=(batch_size, num_classes)`
- `onehot_targets:Var` onehot编码的标签，`dtype=float`，`shape=(batch_size, num_classes)`

返回：交叉熵损失

返回类型：`Var`

示例

```python
>>> predict = np.random.random([2,3])
>>> onehot = np.array([[1., 0., 0.], [0., 1., 0.]])
>>> nn.loss.cross_entropy(predict, onehot)
array(4.9752955, dtype=float32)
```
---
### `kl(predicts, onehot_targets)`
求KL损失

参数：
- `predicts:Var` 输出层的预测值，`dtype=float`，`shape=(batch_size, num_classes)`
- `onehot_targets:Var` onehot编码的标签，`dtype=float`，`shape=(batch_size, num_classes)`

返回：KL损失

返回类型：`Var`

示例

```python
>>> predict = np.random.random([2,3])
>>> onehot = np.array([[1., 0., 0.], [0., 1., 0.]])
>>> nn.loss.kl(predict, onehot)
array(inf, dtype=float32)
```
---
### `mse(predicts, onehot_targets)`
求MSE损失

参数：
- `predicts:Var` 输出层的预测值，`dtype=float`，`shape=(batch_size, num_classes)`
- `onehot_targets:Var` onehot编码的标签，`dtype=float`，`shape=(batch_size, num_classes)`

返回：MSE损失

返回类型：`Var`

示例

```python
>>> predict = np.random.random([2,3])
>>> onehot = np.array([[1., 0., 0.], [0., 1., 0.]])
>>> nn.loss.mse(predict, onehot)
array(1.8694793, dtype=float32)
```
---
### `mae(predicts, onehot_targets)`
求MAE损失

参数：
- `predicts:Var` 输出层的预测值，`dtype=float`，`shape=(batch_size, num_classes)`
- `onehot_targets:Var` onehot编码的标签，`dtype=float`，`shape=(batch_size, num_classes)`

返回：MAE损失

返回类型：`Var`

示例

```python
>>> predict = np.random.random([2,3])
>>> onehot = np.array([[1., 0., 0.], [0., 1., 0.]])
>>> nn.loss.mae(predict, onehot)
array(2.1805272, dtype=float32)
```
---
### `hinge(predicts, onehot_targets)`
求Hinge损失

参数：
- `predicts:Var` 输出层的预测值，`dtype=float`，`shape=(batch_size, num_classes)`
- `onehot_targets:Var` onehot编码的标签，`dtype=float`，`shape=(batch_size, num_classes)`

返回：Hinge损失

返回类型：`Var`

示例

```python
>>> predict = np.random.random([2,3])
>>> onehot = np.array([[1., 0., 0.], [0., 1., 0.]])
>>> nn.loss.hinge(predict, onehot)
array(2.791432, dtype=float32)
```