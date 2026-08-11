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
cd schema        # 注意：脚本在 schema/ 下，不在 schema/default/
./generate.sh
```

生成产物统一落在 **`schema/current/*.h`**（不是 `schema/default/`）。

### ⚠️ 双 Schema：`default` 与 `private`

`generate.sh` 会**优先使用 `schema/private/`**（见脚本内 `DIR` 判断）：

```bash
DIR="default"
if [ -d "private" ]; then
  DIR="private"      # private 存在时，default 完全不参与生成
fi
```

内部仓库同时存在两份 `MNN.fbs`：

| 文件 | 角色 | 说明 |
|---|---|---|
| `schema/private/MNN.fbs` | 内部版，**本仓库实际生效** | 受访问限制；改动前须获得明确授权 |
| `schema/default/MNN.fbs` | 开源版 | 本仓库构建时**不参与生成**，但决定开源发布内容 |

**因此：算子若计划随开源版发布，必须同步加到两份 `.fbs`**。只改 private 会导致开源检出（无 `private/` 目录）生成的头文件缺少该 OpType，相关代码编译失败。

两份文件的枚举风格不同（`default` 大量隐式递增，`private` 显式赋值），**同名算子在两边的数值可能不同**（例如 `ConvInt8`：default=513 / private=517）。所以**不能整段照搬**，要按各自风格插入，并验证关键数值一致：

```bash
# 用 default 单独生成到临时目录，比对 OpType 值与 union 索引
mkdir -p /tmp/schema_check && cp schema/default/*.fbs /tmp/schema_check/
cd /tmp/schema_check && <repo>/3rd_party/flatbuffers/tmp/flatc \
    -c -b --gen-object-api --reflect-names *.fbs
grep -n "OpType_MyCustomOp = \|OpParameter_MyCustomOpParam = " MNN_generated.h
# 与 schema/current/MNN_generated.h 中同两行必须完全一致
```

`OpParameter` union 成员按**位置**编码进模型二进制，所以两边都必须**追加到 union 末尾**且顺序一致，否则同一模型在两个版本间会被解析成不同参数类型。

### 手改生成头文件的坑（不推荐，但需知道）

正常流程应改 `.fbs` 后跑 `generate.sh`。若不得已直接手改 `schema/current/MNN_generated.h`，注意 MiniReflect 是**多个并行数组**，改一处不够：

- `EnumNamesOpType()` 是**按枚举值索引的补空表**，删条目会让其后所有算子错位（曾出现 `Extra` 从 512 变成 511），须补 `""` 占位。
- `OpTypeTypeTable()` 的 `type_codes` / `names` / `values` 三数组长度必须相等，**且 `TypeTable tt` 结构里的计数也要同步改**。漏改计数时 `MNNConvert -f JSON` 能通过，但 `MNNDump2Json` 会 `std::bad_alloc` abort，症状表现为导出图里大量 `Extra` 未被 rebuild、op 总数暴增，报错信息指向完全无关的算子。

改完后跑 `generate.sh` 并 `diff` 手改结果，是验证正确性最可靠的方式。

---

## 步骤 1 测试标准

### 测试方法

```bash
# 1. Schema 编译无错误
cd schema
./generate.sh
# 应该无报错

# 2. 验证 OpType 已添加
grep "MyCustomOp" schema/current/MNN_generated.h
# 应该找到 OpType_MyCustomOp

# 3. 如有参数，验证参数 table 已生成
grep "MyCustomOpParam" schema/current/MNN_generated.h
# 应该找到对应的结构体
```

### 通过标准

- [ ] `generate.sh` 运行无错误
- [ ] `schema/current/MNN_generated.h` 中包含 `OpType_MyCustomOp`
- [ ] 如有参数，`MNN_generated.h` 中包含参数结构体
- [ ] Schema 修改只在列表末尾追加，未修改已有定义
- [ ] **若计划开源发布**：`default` 与 `private` 两份 `.fbs` 均已添加，且 OpType 值与 `OpParameter` union 索引一致

### 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| flatc 编译失败 | fbs 语法错误 | 检查 table/enum 定义的语法 |
| OpType 值冲突 | 插入了中间位置 | 只在末尾追加 |
| generate.sh 不存在 | 路径不对 | 脚本在 `schema/` 下，不在 `schema/default/` |
| 改了 `default` 却不生效 | `private/` 存在时优先级更高 | 内部仓库改 `private/`；开源发布则两份都改 |
| 开源版编译报缺少 OpType | 只加了 `private/` | 同步加到 `schema/default/MNN.fbs` |

---

## 下一步

- **输出形状与输入不同** → 进入 `step2-shape.md`（形状计算）
- **输出形状与输入一致** → 跳过步骤 2，进入 `step3-compute.md`（计算实现）
