## nn.RuntimeManager

```python
class RuntimeManager
```
RuntimeManager持有运行时资源，在CPU时持有线程池，内存池等资源；在GPU时持有Kernal池等资源；
模型的执行需要使用RuntimeManager的资源，在同一个线程内RuntimeManager可以被共享使用，*注意：不可跨线程使用*

---
### `RuntimeManager()`
创建一个空Tensor

*在实际使用中创建空RuntimeManager没有意义，请使用`nn.create_runtime_manager`来创建RuntimeManager*

参数：
- `None`

返回：RuntimeManager对象

返回类型：`RuntimeManager`

---
### `set_cache(cache_path)`

设置缓存文件路径，在GPU情况下可以把kernel和Op-info缓存到该文件中

参考：[Interpreter.setCacheFile](Interpreter.html#setcachefile-cache-path)

参数：
- `cache_path:str`

返回：`None`

返回类型：`None`

---
### `set_external(path)`

设置额外数据文件路径，使用该文件中的数据作为权重或常量

参考：[Interpreter.setExternalFile](Interpreter.html#setexternalfile-path)

参数：
- `path:str`

返回：`None`

返回类型：`None`

---
### `update_cache()`

在执行推理之后，更新GPU的kernel信息到缓存文件；应该在每次推理结束后指定该函数

参考：[Interpreter.updateCacheFile](Interpreter.html#updatecachefile-session-flag)

参数：
- `None`

返回：`None`

返回类型：`None`

---
### `set_mode(mode)`

设置会话的执行模式

参考：[Interpreter.setSessionMode](Interpreter.html#setsessionmode-mode)

参数：
- `mode:int` 执行Session的模式，请参考[mode](Interpreter.html#setsessionmode-mode)

返回：`None`

返回类型：`None`

---
### `set_hint(mode, value)`

设置执行时的额外信息

参考：[Interpreter.setSessionMode](Interpreter.html#setsessionhint-mode-value)

参数：
- `mode:int` hint类型
- `value:int` hint值

返回：`None`

返回类型：`None`

---
### `Example`

```python
import MNN.nn as nn
import MNN.cv as cv
import MNN.numpy as np
import MNN.expr as expr

config = {}
config['precision'] = 'low'
# 使用GPU后端
config['backend'] = 3
config['numThread'] = 4

# 创建RuntimeManager
rt = nn.create_runtime_manager((config,))
rt.set_cache(".cachefile")
# mode = auto_backend
rt.set_mode(9)
# tune_num = 20 
rt.set_hint(0, 20)
# 加载模型并使用RuntimeManager
net = nn.load_module_from_file('mobilenet_v1.mnn', ['data'], ['prob'], runtime_manager=rt)
# cv读取bgr图片
image = cv.imread('cat.jpg')
# 转换为float32, 形状为[224,224,3]        
image = cv.resize(image, (224, 224), mean=[103.94, 116.78, 123.68], norm=[0.017, 0.017, 0.017])
# 增加batch HWC to NHWC
input_var = np.expand_dims(image, 0)
# NHWC to NC4HW4
input_var = expr.convert(input_var, expr.NC4HW4)
# 执行推理
output_var = net.forward(input_var)
# NC4HW4 to NHWC 
output_var = expr.convert(output_var, expr.NHWC)
# 打印出分类结果, 282为猫
print("output belong to class: {}".format(np.argmax(output_var)))
```