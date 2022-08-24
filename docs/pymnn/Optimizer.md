## optim.Optimizer

```python
class Optimizer
```
Optimizer是一个优化器基类`Optimizer`，`SGD`和`ADAM`都是该类的具体实现

---
### `Optimizer()`
创建一个空Optimizer

*在实际使用中创建空Optimizer没有意义，请使用`optim.SGD`或`optim.ADAM`来创建优化器实例*

参数：
- `None`

返回：Optimizer对象

属性类型：读写

返回类型：`Optimizer`

---
### `learning_rate`

获取和设置优化器的学习率

属性类型：读写

类型：`float`

---
### `momentum`

获取和设置优化器的动量

属性类型：读写

类型：`float`

---
### `momentum2`

获取并设置优化器的第二个动量，*只有ADAM优化器才有该属性*

属性类型：读写

类型：`float`

---
### `weight_decay`

获取并设置优化器的权重衰减因子

属性类型：读写

类型：`float`

---
### `eps`

获取并设置优化器的eps系数，*只有ADAM优化器才有该属性*

属性类型：读写

类型：`float`

---
### `regularization_method`

获取并设置优化器的正则化方法

属性类型：读写

类型：`RegularizationMethod`

---
### `step(loss)`
反向传播以获得参数的梯度，并使用其相应的梯度更新参数

参数：
- `loss:Var` 当前的损失函数值

返回：是否更新成功

返回类型：`bool`

---
示例

请参考[`optim`](optim.md)