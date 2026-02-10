# 🚀 MNN 基于后训练的模型压缩指南

> **一句话总结**：无需训练、一键压缩 —— 适配绝大多数部署场景。


## 📌 基于后训练的模型压缩方案

基于后训练的模型压缩，无需训练，包括如下方案：

- ✅ **权值量化（Weight Quantization）** —— 模型体积缩小 75%，结合**动态量化**可以提速+省内存
- ✅ **FP16 压缩** —— 模型体积缩小 50%，精度基本无损
- ✅ **自动压缩策略（auto_quant.py）** —— 基于权值量化，自动为各算子选择不同的量化方案，以保障模型精度
- ✅ **离线量化（少量校准数据）** —— 全图 int8 推理，提速 + 省内存 + 省体积


## 🧰 压缩方案总览

| 压缩类型 | 是否需要数据 | 是否需要训练 | 压缩率 | 推理加速 | 使用复杂度 |
|----------|--------------|--------------|--------|-----------|-------------|
| 权值量化（2-8bit） | ❌ | ❌ | 75%-87% ↓ | ❌（默认）✅（开启动态量化） | ⭐ |
| FP16 压缩 | ❌ | ❌ | 50% ↓ | ❌ | ⭐ |
| 自动量化调优（4-8bit） | ✅（测试数据集） | ❌ | 75%-87% ↓ | ❌（默认）✅（开启动态量化） | ⭐⭐ |
| 离线量化（8bit） | ✅（少量校准图） | ❌ | 75% ↓ | ✅ | ⭐⭐ |

> ✅ 推荐优先使用：**权值量化 + 动态量化加速**，或 **离线量化**


## 🛠️ 一、安装模型转换工具

### 1. C++ 工具（推荐用于生产环境）

```bash
cd MNN
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_QUANTOOLS=ON
make -j8
```

生成工具：
- `MNNConvert`：模型转换 + 权值量化 / FP16
- `quantized.out`：离线量化工具

### 2. Python 工具（推荐用于快速实验）

```bash
pip install MNN
```

安装后命令行工具：
- `mnnconvert` → `MNNConvert` 的 Python 封装
- `mnnquant` → `quantized.out` 的 Python 封装
- `mnn` → 工具总入口



## 📦 二、权值量化（推荐首选）

> **仅压缩模型体积，不改变计算精度，推理速度不变，一键完成**

### ✅ 适用场景
- 模型太大，需减小体积
- 不希望改变推理行为
- 需要快速部署，无校准数据

### 🧪 使用方法

```bash
# ONNX → MNN + 权值量化
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model_quant.mnn --weightQuantBits 8

# 使用 HQQ 量化算法（量化时间增长，一般情况下精度增加）
./MNNConvert ... --weightQuantBits 8 --hqq

# 或分块量化（精度更高，体积略增）【可与HQQ叠加使用】
./MNNConvert ... --weightQuantBits 8 --weightQuantBlock 128
```

> 📌 `--weightQuantBlock` 越小精度越高，建议 32~128

> 📌 `--hqq` 可以和 `--weightQuantBlock` 叠加使用

### ⚡ 开启动态量化加速（真正提速）

权值量化默认推理时还原为 float，**开启动态量化可真正使用 int8 计算**：

#### 1. 编译时开启低内存模式：

```bash
cmake .. -DMNN_LOW_MEMORY=ON
```

#### 2. 推理时设置 Memory Mode：

```cpp
MNN::ScheduleConfig config;
BackendConfig backendConfig;
backendConfig.memory = BackendConfig::Memory_Low; // ✅ 关键！
config.backendConfig = &backendConfig;
```

> ✅ 开启后，卷积等核心算子将使用 int8 计算（权重存储使用 int4 或 int8），内存占用更低，速度更快



## 🧊 三、FP16 压缩（无损半精度）

> **模型体积减半，精度几乎无损，不影响推理性能 **

### ✅ 适用场景
- 模型体积敏感
- 不希望损失精度

### 🧪 使用方法

```bash
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model_fp16.mnn --fp16
```

### ⚡ 推理时开启 FP16 加速：

```cpp
BackendConfig backendConfig;
backendConfig.precision = BackendConfig::Precision_Low; // ✅ 开启低精度加速
config.backendConfig = &backendConfig;
```

> 📌 注意：`--fp16` 是**存储压缩**，`Precision_Low` 是**运行时加速**，两者是独立使用的
> 📌 没有使用`--fp16`压缩的模型，也可以通过设置 `Precision_Low` 以支持**运行时加速**

## 🤖 四、自动压缩工具（auto_quant.py）

> **自动搜索最优量化策略，跳过敏感层，保障精度**

### ✅ 适用场景
- 不确定哪些层该量化
- 量化后精度下降明显
- 希望自动化压缩流程

### 🧪 使用步骤

#### 1. 将模型转成MNN格式，示例：
```
./MNNConvert -f ONNX --modelFile src.onnx --MNNModel float.mnn
```

#### 2. 参考[正确性校验](convert.html#id3)，构建测试文件夹 mnntest

结构：
```
mnntest/
├── input.json   # 输入输出配置
├── input0.txt   # 输入0数据
├── input1.txt   # 输入1数据
├── output.txt  # 输出数据
```

#### 3. 执行自动压缩

```bash
python tools/converter/tools/auto_quant.py \
    --model float.mnn \
    --quant_model auto_quant.mnn \
    --test_dir mnntest \
    --rate 0.05  # 允许最大误差率
```

> ✅ 自动生成 `auto_quant.mnn` 和 `auto_quant.mnn.json`（压缩策略文件）

#### 4. （可选）手动调整策略

编辑 `auto_quant.mnn.json`，将敏感层 `bits` 设为 0（跳过量化）或 16（高精度）：

```json
      "weight": [
       {
        "name": "Convolution11",
        "bits": 8,
        "asymmetric": true,
        "blockSize": -1
       }
      ],

```

重新转换：

```bash
./MNNConvert -f ONNX --modelFile float.mnn --MNNModel auto_quant.mnn --compressionParamsFile auto_quant.mnn.json --hqq
```

## 🔍 五、离线量化（推荐用于加速）

> **使用少量校准数据，将模型转为全 int8 推理，体积缩小 75%，推理加速**

### ✅ 适用场景
- 需要最大推理速度
- 有少量代表性数据（100~1000 张图片）
- 不想训练，但接受轻微精度损失
- 输入输出尺寸大，需要减少这部分的内存

### 🧪 使用方法

#### Step 1: 先转为 float MNN

```bash
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model_float.mnn
```

#### Step 2: 准备数据集，编写量化json文件

参考如下文件编写 quant.json：
```json
{
    "format":"RGB",
    "mean":[
        103.94,
        116.78,
        123.68
    ],
    "normal":[
        0.017,
        0.017,
        0.017
    ],
    "width":224,
    "height":224,
    "path":"../resource/images/",
    "used_image_num":2,
    "feature_quantize_method":"KL",
    "weight_quantize_method":"MAX_ABS",
    "model":"mobilenet.mnn"
}
```

该Json的配置信息如下表所示：

|  key   |  value  |  说明  |
|--------|---------|-------|
| format |  "RGB", "BGR", "RGBA", "GRAY" | 图片统一按RGBA读取，然后转换到`format`指定格式 |
| mean/normal | `[float]` | `dst = (src - mean) * normal` |
| width/height | `int` | 模型输入的宽高 |
| path | `str` | 存放校正特征量化系数的图片目录 |
| used_image_num | `int` | 用于指定使用上述目录下多少张图片进行校正，默认使用`path`下全部图片 |
| feature_quantize_method | "KL", "ADMM", "EMA" | 指定计算特征量化系数的方法，默认："KL" |
| weight_quantize_method | "MAX_ABS", "ADMM" | 指定权值量化方法，默认："MAX_ABS" |
| feature_clamp_value | `int` | 特征的量化范围，默认为127，即[-127, 127]对称量化，有时，量化之后溢出会很多，造成误差较大，可适当减小此范围，如减小至120，但范围减小太多会导致分辨率下降，使用时需测试 |
| weight_clamp_value | `int` | 权值的量化范围，默认127，作用同feature_clamp_value，由于权值精度模型效果影响较大，建议调整feature_clamp_value即可 |
| batch_size | `int` | EMA方法中指定batch size，和模型训练时差不多 |
| quant_bits | `int` | 量化后的bit数，默认为8 |
| skip_quant_op_names | `[str]` | 跳过不量化的op的卷积op名字，因为有些层，如第一层卷积层，对模型精度影响较大，可以选择跳过不量化，可用netron可视化模型，找到相关op名字 |
| input_type | `str` | 输入数据的类型，默认为"image" |
| debug | `bool` | 是否输出debug信息，true或者false，输出的debug信息包含原始模型和量化模型各层输入输出的余弦距离和溢出率 |

| feature_quantize_method | 说明 |
|--------------------|------|
| KL | 使用KL散度进行特征量化系数的校正，一般需要100 ~ 1000张图片(若发现精度损失严重，可以适当增减样本数量，特别是检测/对齐等回归任务模型，样本建议适当减少) |
| ADMM | 使用ADMM（Alternating Direction Method of Multipliers）方法进行特征量化系数的校正，一般需要一个batch的数据 |
| EMA | 使用指数滑动平均来计算特征量化参数，这个方法会对特征进行非对称量化，精度可能比上面两种更好。使用这个方法时batch size应设置为和训练时差不多最好。|

| weight_quantize_method | 说明 |
|--------------------|------|
| MAX_ABS | 使用权值的绝对值的最大值进行对称量化 |
| ADMM | 使用ADMM方法进行权值量化 |

#### 多输入模型的配置

对于多输入模型，quant.json文件需要特别指定参数

| 需要特别指定的参数 | 设置值 |
|--------------------|------|
| input_type | `str`：输入数据的类型，"sequence" |
| path | `str`：存放校正特征量化系数的输入数据目录 |

例如在quant.json文件中 "path": "/home/data/inputs_dir/"，你所构造的矫正数据集有两个，分别存放在input_0和input_1子目录下，即"/home/data/inputs_dir/input_0"和"/home/data/inputs_dir/input_1".由GetMNNInfo工具可以得到模型的输入输出名称，例如该模型的输入有三个：data0, data1, data2，输出有两个：out1, out2. 那么在input_0和input_1子目录下分别有六个文件：data0.txt, data1.txt, data2.txt, out1.txt, out2.txt, input.json. 其中的五个文件名要和模型的输入输出名对应，最后一个input.json文件则描述的是输入名和对应的shape内容：
```json
{
    "inputs": [
        {
            "name": "data0",
            "shape": [
                2,
                4,
		        64,
		        64
            ]
        },
	        {
            "name": "data1",
            "shape": [
                1
            ]
        },
        {
            "name": "data2",
            "shape": [
                2,
                512,
                768
            ]
        }
    ],
    "outputs": [
        "out1", "out2"
    ]
}
```

#### Step 3: 使用 `quantized.out` 或 `mnnquant` 进行离线量化

```bash
./quantized.out model_float.mnn model_quant_int8.mnn quant.json
```

或 Python：

```bash
mnnquant model_float.mnn model_quant_int8.mnn quant.json
```

---


## 🎯 六、推荐压缩策略

| 需求 | 推荐方案 | 命令示例 |
|------|----------|----------|
| **只想缩小模型体积** | 权值量化 8bit | `--weightQuantBits 8` |
| **想缩小体积 + 加速** | 权值量化 + 动态量化 | `--weightQuantBits 8` + 编译 `-DMNN_LOW_MEMORY=ON` + 推理 `Memory_Low` |
| **怕精度掉，想自动调优** | auto_quant.py | `auto_quant.py --rate 0.05` |
| **想最大加速 + 有校准数据** | 离线量化 | `quantized.out` + 校准数据集 |
| **想无损压缩 + 用 GPU 加速** | FP16 存储 + Precision_Low | `--fp16` + 推理 `Precision_Low` |

---

## 📈 压缩效果参考（ImageNet 模型，华为 P20 Pro）

| 模型 | 原始体积 | 权值量化体积 | 离线量化体积 | 离线量化加速（ARMv8） |
|------|----------|--------------|--------------|------------------------|
| ResNet-18 | 45M | 12M (-73%) | 12M | 187ms → 167ms |
| MobileNetV2 | 14M | 3.5M (-75%) | 3.5M | 62ms → 42ms |
| EfficientNet-B0 | 21M | 5.3M (-75%) | 5.3M | 128ms → 100ms |


---

## ❓ 常见问题

### Q1: 权值量化后精度下降怎么办？
- 尝试 `--hqq`
- 尝试 `--weightQuantBlock 128`
- 使用 `auto_quant.py` 自动跳过敏感层

### Q2: 动态量化没加速？
- 确认编译时加了 `-DMNN_LOW_MEMORY=ON`
- 确认推理时设置了 `Memory_Low`
- 某些小模型或算子可能无加速效果

### Q3: 离线量化需要多少数据？
- 100~1000 张有代表性图片即可
- 数据分布应覆盖真实场景

### Q4: FP16 会损失精度吗？
- 在视觉模型上通常 <0.1% 精度损失
- 数值敏感任务（如语音、检测小目标）建议测试

---

## 📚 附录：命令行参数速查

### MNNConvert 常用压缩参数

```bash
--weightQuantBits 8          # 权值量化 8bit ，可选 2-8
--hqq      # 启用 HQQ 量化算法
--weightQuantBlock 128        # 分块量化大小
--fp16                       # FP16 存储压缩
--compressionParamsFile xxx.json  # 自定义压缩策略
```

---

## ✅ 总结：三步完成模型压缩

1. **选方案**：
   - 只缩体积 → 权值量化
   - 要加速 → 离线量化 or 权值量化 + 动态量化
   - 怕掉点 → auto_quant.py

2. **转模型**：
   ```bash
   ./MNNConvert ... --weightQuantBits 8
   ```

3. **（可选）调推理配置**：
   ```cpp
   backendConfig.memory = BackendConfig::Memory_Low; // 开启动态量化
   backendConfig.precision = BackendConfig::Precision_Low; // 开启FP16加速
   ```


如需基于训练实现的更高精度或压缩率的方案（剪枝/低秩/训练量化），请参考 [mnncompress 文档](mnncompress.md)，但**90% 场景转换压缩已足够**。


