# 预转换模型下载

MNN 团队在 ModelScope 和 Hugging Face 上持续发布预转换的 MNN 格式模型（已量化，开箱即用）。

## 模型仓库

| 平台 | 地址 |
|------|------|
| **ModelScope** | [https://modelscope.cn/organization/MNN](https://modelscope.cn/organization/MNN) |
| **Hugging Face** | [https://huggingface.co/taobao-mnn](https://huggingface.co/taobao-mnn) |

在上述页面中搜索模型名称（如 `Qwen2.5`、`DeepSeek`、`Llama` 等）即可找到对应的 MNN 预转换版本。

## 使用方式

以 Qwen2.5-1.5B-Instruct 为例：

```bash
# 从 ModelScope 下载
git lfs install
git clone https://modelscope.cn/models/MNN/Qwen2.5-1.5B-Instruct-MNN.git

# 直接运行
./llm_demo Qwen2.5-1.5B-Instruct-MNN/config.json
```

## 自行导出模型

如果预转换模型列表中没有你需要的模型，可以使用 `llmexport` 工具自行导出，参考 [LLM 部署指南](llm.md)。
