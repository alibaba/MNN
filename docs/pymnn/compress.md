## compress

```python
module compress
```
compress模块用来做`Quantization-Aware-Training(QAT)`训练量化，提供了训练量化的接口

---
### `compress.Feature_Scale_Method`
对特征的量化方式，可以针对整个特征进行量化，也可以针对每个channel进行量化
- 类型：`Enum`
- 枚举值：
  - `PER_TENSOR`
  - `PER_CHANNEL`

---
### `compress.Scale_Update_Method`
scale的更新方式
- 类型：`Enum`
- 枚举值：
  - `MAXIMUM`
  - `MOVING_AVERAGE`

---
### `train_quant(module, |quant_bits, feature_scale_method, scale_update_method)`

训练量化

参数：
- `module` 待训练模型
- `quant_bits` 量化位数，默认为 `8`
- `feature_scale_method` 特征的量化方式，默认为 `PER_TENSOR`
- `scale_update_method` scale的更新方式，默认为 `MOVING_AVERAGE`

返回：是否成功

返回类型：`bool`

示例

```python
# args are self-explained
nn.compress.train_quant(module, quant_bits = 8, feature_scale_method = Feature_Scale_Method.PER_TENSOR, scale_update_method = Scale_Update_Method.MOVING_AVERAGE)
```