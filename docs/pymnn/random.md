## random

```python
module random
```
random模块提供了[numpy.random](https://numpy.org/doc/stable/reference/random/index.html)模块的部分函数，提供了生成随机数的功能。

---
### `rand(*args)`
作用等同与 `numpy` 中的 [`np.random.rand`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html) 函数，生成指定形状的随机数。

参数：
- `args:[int]` 输出随机数的形状

返回：得到的随机变量 

返回类型：`Var` 

示例

```python
>>> np.random.rand([2, 3])
array([[4.1702199e-01, 9.9718481e-01, 7.2032452e-01],
       [9.3255734e-01, 1.1438108e-04, 1.2812445e-01]], dtype=float32)
```
---
### `randn(*args)`
作用等同与 `numpy` 中的 [`np.random.randn`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html) 函数，生成指定形状的随机数。

参数：
- `args:[int]` 输出随机数的形状

返回：得到的随机变量 

返回类型：`Var` 

示例

```python
>>> np.random.randn([2, 3])
array([[4.1702199e-01, 9.9718481e-01, 7.2032452e-01],
       [9.3255734e-01, 1.1438108e-04, 1.2812445e-01]], dtype=float32)
```
---
### `randint(low, high=None, size=None, dtype=_F.int)`
作用等同与 `numpy` 中的 [`np.random.randint`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html) 函数，生成指定形状,范围的随机数。

参数：
- `low:scalar` 输出随机数的最小值
- `high:scalar` 输出随机数的最大值
- `size:[int]` 输出随机数的形状
- `dtype:dtype，输出随机数的类型` 输出随机数的类型，默认为int

返回：得到的随机变量 

返回类型：`Var` 

示例

```python
>>> np.random.randint(0, 4, size=10, dtype=np.int32)
array([1, 3, 2, 3, 0, 0, 1, 3, 0, 0], dtype=int32)
```
---
### `random(shape)`
作用等同与 `numpy` 中的 [`np.random.random`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html) 函数，生成指定形状的随机数。

参数：
- `shape:[int]` 输出随机数的形状

返回：得到的随机变量 

返回类型：`Var` 

示例

```python
>>> np.random.random([2, 3])
array([[4.1702199e-01, 9.9718481e-01, 7.2032452e-01],
       [9.3255734e-01, 1.1438108e-04, 1.2812445e-01]], dtype=float32)
```