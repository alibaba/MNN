# 离线量化工具

> **本页内容已合并至 [模型压缩指南](compress.md)**，请前往查看完整的离线量化使用说明（包含权值量化、离线量化、FP16 压缩、自动量化调优等方案）。

## 快速参考

离线量化使用 `quantized.out`（C++）或 `mnnquant`（Python）工具：

```bash
# C++
./quantized.out origin.mnn quantized.mnn quant.json

# Python (pip install MNN 后可用)
mnnquant origin.mnn quantized.mnn quant.json
```

详细的 `quant.json` 配置说明和使用方法，请参考 [模型压缩指南 - 离线量化](compress.md) 章节。
