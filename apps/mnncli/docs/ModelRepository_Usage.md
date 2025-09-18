# ModelRepository 使用说明

## 概述

`ModelRepository` 类是一个 C++ 实现，用于管理模型市场数据，功能类似于 Android 版本的 `ModelRepository.kt`。它能够从 `model_market.json` 文件中加载模型信息，并根据指定的下载提供商（download provider）为模型创建正确的下载 ID。

## 主要功能

### 1. 模型市场数据管理
- 从 assets 目录加载 `model_market.json` 文件
- 解析模型信息，包括常规模型、TTS 模型和 ASR 模型
- 支持版本比较和缓存机制

### 2. 模型 ID 生成
- 根据模型名称和下载提供商生成正确的模型 ID
- 格式：`{Provider}/{RepoPath}`，例如：`HuggingFace/taobao-mnn/gpt-oss-20b-MNN`

### 3. 模型类型识别
- 自动识别模型类型：ASR、TTS 或 LLM
- 支持多种下载提供商：HuggingFace、ModelScope、Modelers

## 使用方法

### 基本用法

```cpp
#include "model_repository.hpp"

// 创建实例
auto& repo = mnncli::ModelRepository::getInstance("./cache");

// 设置下载提供商
repo.setDownloadProvider("HuggingFace");

// 获取模型 ID 用于下载
auto modelId = repo.getModelIdForDownload("gpt-oss-20b-MNN", "HuggingFace");
if (modelId) {
    std::cout << "Model ID: " << *modelId << std::endl;
    // 输出: Model ID: HuggingFace/taobao-mnn/gpt-oss-20b-MNN
}
```

### 获取模型列表

```cpp
// 获取所有模型
auto models = repo.getModels();

// 获取 TTS 模型
auto ttsModels = repo.getTtsModels();

// 获取 ASR 模型
auto asrModels = repo.getAsrModels();

// 处理模型信息
for (const auto& model : models) {
    std::cout << "Model: " << model.modelName << std::endl;
    std::cout << "ID: " << model.modelId << std::endl;
    std::cout << "Source: " << model.currentSource << std::endl;
    std::cout << "Repo Path: " << model.currentRepoPath << std::endl;
}
```

### 模型类型识别

```cpp
// 识别模型类型
std::string modelType = repo.getModelType("HuggingFace/taobao-mnn/gpt-oss-20b-MNN");
std::cout << "Model type: " << modelType << std::endl;
// 输出: Model type: LLM
```

### 与 ModelDownloadManager 集成

```cpp
#include "model_download_manager.hpp"

// 创建下载管理器
auto& downloadManager = mnncli::ModelDownloadManager::getInstance("./cache");

// 使用 ModelRepository 获取模型 ID
auto& repo = mnncli::ModelRepository::getInstance("./cache");
repo.setDownloadProvider("HuggingFace");

auto modelId = repo.getModelIdForDownload("gpt-oss-20b-MNN", "HuggingFace");
if (modelId) {
    // 开始下载
    downloadManager.startDownload(*modelId);
}
```

## 配置选项

### 下载提供商优先级

默认的下载提供商优先级（按顺序）：
1. `HuggingFace` - 默认首选
2. `ModelScope` - 备选方案
3. `Modelers` - 第三选择

### 文件路径配置

`ModelRepository` 会按以下顺序查找 `model_market.json` 文件：
1. `{cache_root_path}/assets/model_market.json`
2. `{cache_root_path}/model_market.json`
3. `./model_market.json`
4. `../assets/model_market.json`
5. `../../assets/model_market.json`

## 错误处理

### 模型未找到
```cpp
auto modelId = repo.getModelIdForDownload("non-existent-model", "HuggingFace");
if (!modelId) {
    std::cout << "Model not found in repository" << std::endl;
    // 可以回退到直接使用模型名称
}
```

### 下载提供商不支持
```cpp
auto modelId = repo.getModelIdForDownload("gpt-oss-20b-MNN", "UnsupportedProvider");
if (!modelId) {
    std::cout << "Download provider not supported for this model" << std::endl;
}
```

## 测试

运行测试程序：
```bash
# 启用测试构建
cmake -DBUILD_MNNCLI_TEST=ON ..

# 构建测试
make

# 运行测试
./test/model_repository_test
```

## 注意事项

1. **JSON 依赖**: 需要 `nlohmann/json.hpp` 库支持
2. **C++17 支持**: 需要 C++17 或更高版本
3. **文件系统**: 需要 C++17 的 `std::filesystem` 支持
4. **异常处理**: 类使用异常处理机制，调用者需要适当的异常处理

## 与 Kotlin 版本的对应关系

| Kotlin 功能 | C++ 实现 |
|-------------|----------|
| `getModelMarketData()` | `getModelMarketData()` |
| `getModels()` | `getModels()` |
| `getTtsModels()` | `getTtsModels()` |
| `getAsrModels()` | `getAsrModels()` |
| `getModelType()` | `getModelType()` |
| 网络请求 | 目前仅支持 assets 加载 |
| 缓存机制 | 基础缓存支持 |
| 版本比较 | `isVersionLower()` 方法 |

## 未来扩展

- 网络请求支持（类似 Kotlin 版本）
- 本地缓存文件支持
- 更多下载提供商支持
- 模型元数据更新检查
