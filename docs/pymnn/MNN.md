## MNN

```python
module MNN
```
MNN是Pymnn中最基础的Module，其中包含了V2 API所需要数据结构与一些枚举类型；同时包含了一些基础函数。
其他子模块则也需要通过MNN模块引入，引入方式为`import MNN.{module} as {module}`

---
### `MNN submodules`
- [expr](expr.md)
- [numpy](numpy.md)
- [cv](cv.md)
- [nn](nn.md)
- [optim](optim.md)
- [data](data.md)

---
### `MNN Types`
- [Interpreter](Interpreter.md)
- [Session](Session.md)
- [OpInfo](OpInfo.md)
- [Tensor](Tensor.md)
- [CVImageProcess](CVImageProcess.md)
- [CVMatrix](CVMatrix.md)

---
### `version()`
获取当前MNN的版本号

参数：
- `None`

返回：版本号

返回类型：`str`

---
### `get_model_uuid(model_path)`
获取指定模型的uuid信息

*注意：该API仅在打开PYMNN_TRAIN_API时有效，移动端默认关闭*

参数：
- `model_path:str` 模型路径

返回：uuid信息

返回类型：`str`