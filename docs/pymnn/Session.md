## MNN.Session *[deprecated]*

```python
class Session
```
Session是MNN V2接口中推理数据的持有者。Session通过[Interpreter](Interpreter.md)创建；多个推理可以共用同一个模型，即，多个Session可以共用一个Interpreter。

*不建议使用该接口，请使用[nn](nn.md)代替*

---
### `Session()`
创建一个空Tensor

*在实际使用中创建空Session没有意义，请使用`Interpreter.createSession`来创建Session*

参数：
- `None`

返回：Session对象

返回类型：`Session`

---
### `cache()`

将该Session存储到当前线程的缓存中，以便多次使

参数：
- `None`

返回：`None`

返回类型：`None`

---
### `removeCache()`

将该Session从当前线程的缓存中移除

参数：
- `None`

返回：`None`

返回类型：`None`

---
### `Example`
完整用法请参考[Interpreter的用法示例](Interpreter.html#example)

```python
import MNN

# 创建interpreter
interpreter = MNN.Interpreter("mobilenet_v1.mnn")
# 创建session
session = interpreter.createSession({
    'numThread': 2,
    'saveTensors': ('fc7',),
    'inputPaths': ('data',),
    'outputPaths': ('prob',)
})
session.cache()
# session1-3均使用session的cache
session1 = interpreter.createSession()
session2 = interpreter.createSession()
session3 = interpreter.createSession()
session.removeCache()
# session4不使用session的cache
session4 = interpreter.createSession()
```