# 步骤 1：Schema 定义

> **目标**：在 MNN 的 FlatBuffers Schema 中定义新算子的类型和参数。
>
> **前置条件**：明确算子名称、语义（做什么）、输入输出的含义。
>

---

## 1.1 检查算子是否已存在

```bash
# 查看当前支持的所有算子
grep -n "OpType" schema/default/MNN.fbs | head -200

# 搜索可能的已有算子名
grep -i "算子名" schema/default/MNN.fbs
```

如果已存在，**不需要修改 Schema**，跳到步骤 2。

---

## 1.2 确定算子信息

在开始修改前，明确以下信息：

```
算子名称（PascalCase）：____（例如 MyCustomOp）
算子语义：____（一句话描述做什么）
输入 Tensor 数量和含义：____
输出 Tensor 数量和含义：____
是否需要参数：是 / 否
参数字段列表（如需要）：____
```

---

## 1.3 添加算子类型

编辑 `schema/default/MNN.fbs`，在 `OpType` 枚举中追加新算子名称：

```fbs
enum OpType : int {
    AbsVal,
    QuantizedAdd,
    ...
    // ← 在列表末尾添加
    MyCustomOp
}
```

> **注意**：OpType 名必须是 PascalCase，且**只能追加到末尾，不能插入中间**（FlatBuffers 的枚举序号不可变）。

---

## 1.4 添加算子参数（如需要）

如果算子不包含任何参数（仅靠输入 Tensor 决定行为），**跳过此步**。

### 1.4.1 定义参数 table

在合适的 `.fbs` 文件中添加参数定义：
- 通用算子 → 在 `schema/default/MNN.fbs` 中添加
- Caffe 来源 → `schema/default/CaffeOps.fbs`
- TensorFlow 来源 → `schema/default/TensorflowOp.fbs`

```fbs
table MyCustomOpParam {
    axis:int = 0;           // 操作的轴
    keepDims:bool = false;  // 是否保持维度
    // 根据算子需要添加字段
}
```

### 1.4.2 注册参数到 OpParameter

在 `schema/default/MNN.fbs` 的 `OpParameter` union 中追加：

```fbs
union OpParameter {
    QuantizedAdd,
    ArgMax,
    ...
    // ← 在列表末尾添加
    MyCustomOpParam
}
```

> **注意**：同样只能追加到末尾。

---

## 1.5 生成头文件

修改完 Schema 后，需要重新生成 C++ 头文件：

```bash
cd schema/default
./generate.sh
```

> 如果 `generate.sh` 不存在或无法执行，使用 flatc 手动编译：
> ```bash
> flatc -c -b --gen-object-api --reflect-names MNN.fbs
> ```

---

## 步骤 1 测试标准

### 测试方法

```bash
# 1. Schema 编译无错误
cd schema/default
./generate.sh
# 应该无报错

# 2. 验证 OpType 已添加
grep "MyCustomOp" schema/default/MNN_generated.h
# 应该找到 OpType_MyCustomOp

# 3. 如有参数，验证参数 table 已生成
grep "MyCustomOpParam" schema/default/MNN_generated.h
# 应该找到对应的结构体
```

### 通过标准

- [ ] `generate.sh` 运行无错误
- [ ] `MNN_generated.h` 中包含 `OpType_MyCustomOp`
- [ ] 如有参数，`MNN_generated.h` 中包含参数结构体
- [ ] Schema 修改只在列表末尾追加，未修改已有定义

### 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| flatc 编译失败 | fbs 语法错误 | 检查 table/enum 定义的语法 |
| OpType 值冲突 | 插入了中间位置 | 只在末尾追加 |
| generate.sh 不存在 | 路径不对 | 确认在 `schema/default/` 下执行 |

---

## 下一步

- **输出形状与输入不同** → 进入 `step2-shape.md`（形状计算）
- **输出形状与输入一致** → 跳过步骤 2，进入 `step3-compute.md`（计算实现）
