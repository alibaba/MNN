# MNN.llm JavaScript 绑定使用文档

本文档介绍如何使用 MNN 的 JavaScript LLM 绑定进行模型推理。

## 环境要求

- Node.js >= 14.0.0
- macOS (arm64) 或其他支持的平台
- CMake >= 3.15

## 构建步骤

### 1. 构建 MNN 核心库

首先需要使用正确的配置构建 MNN 库，确保包含 LLM 和 Express 支持：

```bash
cd /path/to/MNNNPU

# 清理旧的构建（如果存在）
rm -rf build && mkdir build

# 配置 CMake
cmake -B build -S . \
    -DMNN_BUILD_LLM=ON \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_METAL=ON \
    -DMNN_SEP_BUILD=OFF

# 编译
cmake --build build -j$(sysctl -n hw.ncpu)
```

> **重要**: 必须设置 `-DMNN_BUILD_LLM=ON` 和 `-DMNN_SEP_BUILD=OFF` 才能将 LLM 引擎和 Express 模块编译到 libMNN 中。

### 2. 构建 JavaScript 绑定

```bash
cd js

# 安装依赖
npm install

# 构建绑定
npm run build
```

### 3. 运行测试

```bash
# 运行基础 API 测试
npm test

# 或者单独运行 LLM 测试
npx mocha test/llm.test.js
```

## 使用示例

### 基本用法

```javascript
const MNN = require('@alibaba/mnn');
const path = require('path');

// 1. 创建 LLM 实例
const configPath = path.join(__dirname, 'models/qwen3-0.6b/config.json');
const llm = MNN.llm.create(configPath);

// 2. 加载模型
const loaded = llm.load();
if (!loaded) {
    console.error('Failed to load model');
    process.exit(1);
}

// 3. 生成回复
const response = llm.response('你好');
console.log('Response:', response);
```

### 高级用法

#### 不使用历史记录

```javascript
// 每次调用都重置上下文
const response = llm.response('介绍一下你自己', false);
```

#### 使用 Token IDs 生成

```javascript
// 直接使用 token IDs 进行生成
const inputIds = [151644, 872, 198, 108386, 151645];
const outputIds = llm.generate(inputIds);
console.log('Generated tokens:', outputIds);
```

#### 应用聊天模板

```javascript
const templated = llm.applyChatTemplate('你好');
console.log('Templated prompt:', templated);
```

#### 动态配置

```javascript
// 运行时修改配置
llm.setConfig(JSON.stringify({
    temperature: 0.8,
    topP: 0.9,
    topK: 40
}));
```

## 下载和使用模型

### 从 ModelScope 下载模型

以 Qwen3-0.6B 为例：

```bash
# 安装 git-lfs（如果尚未安装）
brew install git-lfs
git lfs install

# 克隆模型仓库
mkdir -p models
git clone https://www.modelscope.cn/MNN/Qwen3-0.6B-MNN.git models/qwen3-0.6b

# 拉取 LFS 文件
cd models/qwen3-0.6b
git lfs pull
```

### 模型目录结构

下载后的模型目录应包含以下文件：

```
models/qwen3-0.6b/
├── config.json           # 模型配置
├── llm.mnn              # 模型结构
├── llm.mnn.weight       # 模型权重
├── llm_config.json      # LLM 配置
└── tokenizer.txt        # 分词器
```

### 运行推理测试

```bash
# 使用提供的测试脚本
node js/test/qwen_inference.js
```

## API 参考

### MNN.llm.create(configPath)

创建 LLM 实例。

**参数:**
- `configPath` (string): 模型配置文件路径（config.json）

**返回:**
- `Llm` 实例

### llm.load()

加载模型到内存。

**返回:**
- `boolean`: 加载成功返回 true

### llm.response(query, history?)

生成文本回复。

**参数:**
- `query` (string): 输入文本
- `history` (boolean, 可选): 是否使用历史上下文，默认 true

**返回:**
- `string`: 生成的回复文本

### llm.generate(inputIds)

使用 token IDs 生成。

**参数:**
- `inputIds` (number[]): 输入 token ID 数组

**返回:**
- `number[]`: 输出 token ID 数组

### llm.applyChatTemplate(content)

应用聊天模板。

**参数:**
- `content` (string): 用户输入内容

**返回:**
- `string`: 应用模板后的提示文本

### llm.setConfig(config)

动态设置配置。

**参数:**
- `config` (string): JSON 格式的配置字符串

## 性能参考

基于 Qwen3-0.6B 模型在 Apple Silicon (M系列) 上的测试结果：

- **模型加载**: ~600ms
- **首次推理**: ~1100ms
- **后续推理**: ~400-500ms
- **生成速度**: ~600 tokens/s

## 故障排查

### 问题: 运行时崩溃

**原因**: MNN 库未包含 LLM 支持

**解决方案**: 确保使用正确的 CMake 配置重新构建 MNN：
```bash
cmake -B build -S . -DMNN_BUILD_LLM=ON -DMNN_SEP_BUILD=OFF
```

### 问题: 找不到模型文件

**原因**: git-lfs 未正确拉取大文件

**解决方案**:
```bash
cd models/qwen3-0.6b
git lfs pull
```

### 问题: 内存不足

**原因**: 模型太大

**解决方案**: 使用更小的模型或增加系统内存

## 更多示例

完整的测试示例请参考：
- [llm.test.js](file:///Users/songjinde/git/MNNX/MNNNPU/js/test/llm.test.js) - 基础 API 测试
- [qwen_inference.js](file:///Users/songjinde/git/MNNX/MNNNPU/js/test/qwen_inference.js) - 完整推理示例

## 相关资源

- [MNN 官方文档](https://www.mnn.zone/)
- [MNN GitHub](https://github.com/alibaba/MNN)
- [ModelScope 模型库](https://modelscope.cn/)
