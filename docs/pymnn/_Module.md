## nn._Module

```python
class _Module
```
_Module时V3 API中用来描述模型的基类，维护了具体的计算图，可以执行推理和训练等操作；具体的模型类都继承自_Module；
可以通过`nn.load_module_from_file`加载模型来推理或训练现有模型；也可以通过继承_Module来实现自定义模型并训练推理

---
### `_Module()`
创建一个空_Module

*在实际使用中创建空_Module没有意义，请使用`nn.load_module_from_file`或继承实现自己的_Module*

参数：
- `None`

返回：_Module对象

返回类型：`_Module`

---
### `parameters`

获取_Module的参数

属性类型：只读

类型：`[Var]`

---
### `name`

获取_Module的名称

属性类型：只读

类型：`str`

---
### `is_training`

获取_Module的是否为训练模式

属性类型：只读

类型：`bool`

---
### `forward(input)`

模块前向传播,返回一个结果变量

参数：
- `input:Var|[Var]` 前向传播输入变量

返回：前向传播输出变量

返回类型：`Var`

---
### `__call__(input)`
与`forward`相同

---
### `onForward(input)`

模块前向传播,返回多个结果变量

参数：
- `input:Var|[Var]` 前向传播输入变量

返回：前向传播输出变量

返回类型：`[Var]`

---
### `set_name(name)`
设置_Module的名称

参数：
- `name:str` 模块的名称

返回：`None`

返回类型：`None`

---
### `train(isTrain)`
设置_Module的训练状态

参数：
- `isTrain:bool` 是否为训练模式

返回：`None`

返回类型：`None`

---
### `load_parameters(parameters)`
加载现有的参数

参数：
- `parameters:[Var]` 参数值

返回：是否成功加载参数

返回类型：`bool`

---
### `clear_cache()`
清除_Module的缓存，并递归清除子模块的缓存

参数：
- `None`

返回：`None`

返回类型：`None`

---
### `_register_submodules(children)`
注册子模块

参数：
- `children:[_Module]` 子模块列表

返回：`None`

返回类型：`None`

---
### `_add_parameter(parameter)`
添加参数

参数：
- `parameter:Var` 参数值

返回：添加前的参数数量

返回类型：`int`

---
### `Example`

```python
import MNN
import MNN.nn as nn
import MNN.expr as expr

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.conv(1, 20, [5, 5])
        self.conv2 = nn.conv(20, 50, [5, 5])
        
        self.fc1 = nn.linear(800, 500)
        self.fc2 = nn.linear(500, 10)
    def forward(self, x):
        x = expr.relu(self.conv1(x))
        x = expr.max_pool(x, [2, 2], [2, 2])
        x = expr.relu(self.conv2(x))
        x = expr.max_pool(x, [2, 2], [2, 2])
        # some ops like conv, pool , resize using special data format `NC4HW4`
        # so we need to change their data format when fed into reshape
        # we can get the data format of a variable by its `data_format` attribute
        x = expr.convert(x, expr.NCHW)
        x = expr.reshape(x, [0, -1])
        x = expr.relu(self.fc1(x))
        x = self.fc2(x)
        x = expr.softmax(x, 1)
        return x
model = Net()
```