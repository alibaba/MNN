# iOS arm64 二进制体积裁剪报告(3.6.x)

> 目标:MNN 3.3.0.21 → 3.6.1 升级在最终 iOS App arm64 产物中带来约 330~350KB 增量,
> 通过 CMake 开关与源码裁剪将其降到预算内,且不影响线上功能与推理性能。
> 基线 commit:a3f4e6b1(master)。

## 一、根因(Mach-O 内容归因)

对 `-Oz` Release 静态库逐对象 `size -m` 分析(测量口径为 `__TEXT` 段):

| 根因 | 可省体积 | 说明 |
|---|---|---|
| Metal 新增 shader 以未压缩源码字符串嵌入 `__cstring` | ~164KB | 注释/缩进/空行占 24~34%;旧 AllShader 路径本就 minified,新 shader(Attention/LinearAttention/FlashAttn/ConvSimdGroup/LayerNorm/Rope/TopKV2/SharedGather/Softmax/Loop 等)未做 |
| 投机解码(eagle/mtp/lookahead/dflash)无编译开关 | 104,684B | 经 llm.cpp 策略工厂被整体链入 |
| omni.cpp 被 GLOB 无条件编译 | 109,263B | `MNN_BUILD_LLM_OMNI=OFF` 从未真正排除 omni.cpp;`LLM_SUPPORT_VISION` 选项不存在,被 `MNN_BUILD_OPENCV` 强制打开 |
| tokenizer 内 jinja/rapidjson 聊天模板引擎 | 104,081B | 源码已有 `#else` 回退分支,但 CMake 中"always enabled" |
| 2/3-bit 权重量化分支 | 少量 | 线上不需要,按全后端一致方式裁剪 |

Metal/LLM/Express/Shape/CPU 新算子中的其余增量为文本 LLM + Transformer Fuse 必需功能,不裁剪。

## 二、改动内容

### 1. 新增编译开关

| 开关 | 默认值 | 作用 |
|---|---|---|
| `MNN_LLM_SPECULATE` | OFF | 不编译 eagle/mtp/lookahead/dflash;配置了投机类型时告警并回退自回归解码 |
| `MNN_SUPPORT_QUANT_W2W3` | OFF | 关闭 2/3-bit 权重低内存路径。核心门控在 `ConvolutionCommon::load`(强制 float 反量化回退),CPU/ARM82/Vulkan 自动一致;Metal conv1x1 与 OpenCL buffer/image dispatch 显式 guard 到 float 卷积路径 |
| `MNN_LLM_USE_JINJA` | ON | 关闭后 tokenizer 的 chat template 回退为消息拼接 |
| `LLM_SUPPORT_VISION` | ON | 修复:之前只要开 OpenCV 就强制启用 LLM 视觉;现在可显式关闭 |
| `MNN_BUILD_LLM_OMNI` | OFF | 修复:OFF 时真正排除 omni.cpp(宏 `LLM_SUPPORT_OMNI` 门控 `Llm::createLLM` 分发) |

开关全开(ON)时行为与改动前完全一致,兼容路径已编译验证。

### 2. Metal shader 压缩

新增 `source/backend/metal/minify_shader.py`(状态机剥离 `//` 与 `/* */` 注释、空行、缩进,
压缩连续空格;不重命名 token,不影响 preprocessorMacros 运行时替换)。
已对 20 个内嵌 shader 的 `.hpp/.mm/.cpp` 执行。**今后直接修改这些 shader 后需重跑脚本。**

### 3. 未改动项

tokenizer/unicode 及旧 `tokenizer.txt` 兼容逻辑 0 改动;Metal kernel 逻辑 0 改动(仅文本压缩)。

## 三、体积测量(arm64,Release -Oz,__TEXT 合计)

baseline 与 candidate 使用相同 AppleClang、相同 CMake 选项、独立全新构建目录:

```
cmake -S <src> -B <dir> -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_CXX_FLAGS_RELEASE=-Oz -DCMAKE_C_FLAGS_RELEASE=-Oz \
  -DMNN_AAPL_FMWK=ON -DMNN_SEP_BUILD=OFF -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_REDUCE_SIZE=ON \
  -DMNN_METAL=ON -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=OFF -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_LOW_MEMORY=ON -DMNN_ARM82=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=OFF \
  -DMNN_BUILD_AUDIO=OFF -DLLM_SUPPORT_VISION=OFF -DLLM_SUPPORT_HTTP_RESOURCE=OFF \
  -DMNN_METAL_TENSOR=OFF -DMNN_USE_THREAD_POOL=OFF -DMNN_KLEIDIAI=OFF -DMNN_SME2=OFF
```

| 构建 | MNNMetal | llm | 总 __TEXT | 相对 baseline |
|---|---|---|---|---|
| baseline | 1,036,995 | 745,156 | 4,190,054 | — |
| candidate(默认开关) | 872,651 | 531,001 | 3,811,687 | **−378,367** |
| 线上变体(+`MNN_LLM_USE_JINJA=OFF`) | 872,651 | 426,920 | 3,707,606 | **−482,448** |

说明:测量机无 Xcode/iOS SDK,使用 macOS arm64 等价口径;iOS framework 绝对值需在
Xcode 环境按 `cmake/ios.toolchain.cmake` 复测,预期差值与本表同量级。

## 四、正确性验证

- LLM 端到端(3B 文本模型):CPU 77.4 tok/s、Metal 173.4 tok/s,输出连贯;jinja-OFF 变体同样通过。
- Metal 关键算子:convolution、conv_wquant_metal(W2/3/4/8 数值对拍)、softmax、layernorm、
  attention、rope、matmul、cast、argmax、topk、reduction 全部通过。
- CPU 关键算子:convolution、softmax、layernorm、attention、matmul、binary 等全部通过。
- 全量 op 套件中个别失败/崩溃经 baseline 对照确认均为既有问题(位置与数量完全一致)。
- 兼容性:`MNN_BUILD_LLM_OMNI=ON + MNN_LLM_SPECULATE=ON + MNN_SUPPORT_QUANT_W2W3=ON` 全开编译通过。
- 性能:shader 仅文本压缩、语义不变;LLM demo A/B 无回退。

## 五、线上接入

现有打包配置无需改动,原有参数保持不变即可获得大部分收益;如不使用聊天模板,追加
`-DMNN_LLM_USE_JINJA=OFF`。
