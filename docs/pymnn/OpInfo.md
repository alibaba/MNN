## MNN.OpInfo

```python
class OpInfo
```
OpInfo是用来描述推理过程中一个执行层的具体信息的数据类型，其中包含了该层的Op类型，名称，计算量（flops)

用法参考[runSessionWithCallBackInfo](Interpreter.html#runsessionwithcallbackinfo-session-begin-callback-end-callback)

---
### `OpInfo()`
创建一个空OpInfo

*在实际使用中创建空OpInfo没有意义，仅在`runSessionWithCallBackInfo`函数中用作返回信息*

参数：
- `None`

返回：OpInfo对象

返回类型：`OpInfo`

---
### `getName()`

获取该层的名称

参数：
- `None`

返回：该层的名称

返回类型：`str`

---
### `getType()`

获取该层的算子类型

参数：
- `None`

返回：该层的算子类型

返回类型：`str`

---
### `getFlops()`

获取该层的计算量

参数：
- `None`

返回：该层的计算量

返回类型：`float`

---
### `Example`

```python
import MNN

interpreter = MNN.Interpreter("mobilenet_v1.mnn")
session = interpreter.createSession()
# set input
# ...
def begin_callback(tensors, opinfo):
    print('layer name = ', opinfo.getName())
    print('layer op = ', opinfo.getType())
    # print('layer flops = ', opinfo.getFlops())
    for tensor in tensors:  
        print(tensor.getShape())
    return True
def end_callback(tensors, opinfo):
    print('layer name = ', opinfo.getName())
    print('layer op = ', opinfo.getType())
    # print('layer flops = ', opinfo.getFlops())
    for tensor in tensors:  
        print(tensor.getShape())
    return True
interpreter.runSessionWithCallBackInfo(session, begin_callback, end_callback)
```