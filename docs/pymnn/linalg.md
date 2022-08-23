## linalg

```python
module linalg
```
linalg模块提供了[numpy.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html)模块的部分函数，提供了对矩阵的求解和线性代数运算的函数

---
### `norm(a, ord=None, axis=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) 函数，计算矩阵或向量的范数。

参数：
- `a:Var` 将要计算的变量
- `ord:str` 范数的顺序，见下表
- `axis:int` 计算的轴
- `keepdims:bool` 是否保留计数维度

|    ord   |        矩阵范数       |        数组范数       |
|:---------|:---------------------|:---------------------|
| `None` | Frobenius norm | 2-norm |
| `fro` | Frobenius norm | - |
| `nuc` | nuclear norm | - |
| `inf` | max(sum(abs(x), axis=1)) | max(abs(x)) |
| `-inf` | min(sum(abs(x), axis=1)) | min(abs(x)) |
| `0` | - | sum(x != 0) |
| `1` | max(sum(abs(x), axis=0)) | as below |
| `-1` | min(sum(abs(x), axis=0)) | as below |
| `2` | 2-norm | as below |
| `-2` | smallest singular value | as below |
| other | - | sum(abs(x)**ord)**(1./ord) |

返回：计数得到的变量 

返回类型：`Var` 

示例

```python
>>> a = np.arange(9) - 4
>>> np.linalg.norm(a)
array(7.745967, dtype=float32)
```

---
### `svd(a, |full_matrices, compute_uv, hermitian)`
作用等同与 `numpy` 中的 [`np.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) 函数，奇异值分解

*`full_matrices`, `compute_uv`, `hermitian`参数无效只能按照默认值执行*

参数：
- `a:Var` 输入矩阵
- `full_matrices:bool` 是否计算全部的奇异值，numpy兼容参数
- `compute_uv:bool` 是否计算奇异值，numpy兼容参数
- `hermitian:bool` 是否计算对称矩阵的奇异值，numpy兼容参数

返回：以`(u, w, vt)`的顺序返回奇异值分解结果

返回类型：`tuple` of `Var`

示例

```python
>>> x = np.arange(9.).reshape(3, 3)
>>> np.linalg.svd(x)
(array([[ 0.13511899, -0.90281564,  0.40824836],
       [ 0.49633518, -0.2949318 , -0.81649655],
       [ 0.8575514 ,  0.31295216,  0.40824828]], dtype=float32), array([1.4226706e+01, 1.2652264e+00, 7.3003456e-08], dtype=float32), array([[ 0.46632814,  0.57099086,  0.67565346],
       [ 0.7847747 ,  0.08545685, -0.6138614 ],
       [-0.40824845,  0.8164966 , -0.4082482 ]], dtype=float32))
```