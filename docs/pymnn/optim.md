## optim

```python
module optim
```
optim时优化器模块，提供了一个优化器基类`Optimizer`，并提供了`SGD`和`ADAM`优化器实现；主要用于训练阶段迭代优化

---
### `optim Types`
- [Optimizer](Optimizer.md)

---
### `optim.Regularization_Method`
优化器的正则化方法，提供了L1和L2正则化方法
- 类型：`Enum`
- 枚举值：
  - `L1`
  - `L2`
  - `L1L2`

---
### `SGD(module, lr, momentum, weight_decay, regularization_method)`
创建一个SGD优化器

参数：
- `module:_Module` 模型实例
- `lr:float` 学习率
- `momentum:float` 动量，默认为0.9
- `weight_decay:float` 权重衰减，默认为0.0
- `regularization_method:RegularizationMethod` 正则化方法，默认为L2正则化

返回：SGD优化器实例

返回类型：`Optimizer`

示例：

```python
model = Net()
sgd = optim.SGD(model, 0.001, 0.9, 0.0005, optim.Regularization_Method.L2)
# feed some date to the model, then get the loss
loss = ...
sgd.step(loss) # backward and update parameters in the model
```
---
### `ADAM(module, lr, momentum, momentum2, weight_decay, eps, regularization_method)`
创建一个ADAM优化器

参数：
- `module:_Module` 模型实例
- `lr:float` 学习率
- `momentum:float` 动量，默认为0.9
- `momentum2:float` 动量2，默认为0.999
- `weight_decay:float` 权重衰减，默认为0.0
- `eps:float` 正则化阈值，默认为1e-8
- `regularization_method:RegularizationMethod` 正则化方法，默认为L2正则化

返回：ADAM优化器实例

返回类型：`Optimizer`

示例：

```python
model = Net()
sgd = optim.ADAM(model, 0.001)
# feed some date to the model, then get the loss
loss = ...
sgd.step(loss) # backward and update parameters in the model
```