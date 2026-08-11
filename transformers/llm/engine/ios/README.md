# mnn-llm ios demo

🚀 本示例代码全部由`ChatGPT-4`生成。

## 一键性能测试（推荐）

连接 iPhone/iPad（USB，开启开发者模式并信任本机）后，在 MNN 根目录执行：

```bash
sh transformers/llm/engine/ios/ios_llm_bench.sh \
    --model /path/to/EXPORTED_MNN_MODEL \
    --team YOUR_APPLE_TEAM_ID
```

脚本会自动：基于当前分支编译含 LLM 的 `MNN.framework` → 将模型打包进 mnn-llm App → 编译签名 → 安装到设备 → 自动运行 benchmark → 在终端输出 prefill/decode tok/s 报告（日志存于 `bench_logs/`）。

常用选项：`--device UDID` 指定设备、`--skip-framework` 复用已编好的 framework、`--build-only` 只编译不安装、`--cmake-args` 追加 CMake 选项、`--timeout` 等待超时（默认 1800s）。首次安装需在手机 `设置 > 通用 > VPN与设备管理` 中信任开发者证书后重跑。

**定长 benchmark**（指定后端 / prompt 长度 / decode 长度，类似 `llm_bench`）：

```bash
sh transformers/llm/engine/ios/ios_llm_bench.sh --model ... --team ... --skip-framework \
    --backend metal --prompt-len 512 --decode-len 128 [--repeat 3] [--threads 4]
```

后端可选 `cpu` / `metal`；共跑 repeat+1 轮，首轮为 warmup 不计入平均值。App 内聊天框也可手动输入 `bench metal 512 128` 触发同样的测试。

`--prompt-len` / `--decode-len` 支持逗号分隔的多组长度，自动跑全组合矩阵（framework 与 App 只构建安装一次，每组独立日志，报告汇总所有平均值）：

```bash
sh transformers/llm/engine/ios/ios_llm_bench.sh --model ... --team ... --skip-framework \
    --backend metal --prompt-len 512,1024,2048 --decode-len 128,2000   # 3×2=6 组
```

模型目录需为已导出的 MNN 格式（含 `config.json`、`llm.mnn` 等）；若目录中有 `bench.txt` 则用其作为测试 prompts，否则使用默认的 `ios/bench.txt`。

## 速度

[旧版测试prompt](../resource/prompt.txt)
- Qwen-1.8b-chat 4bit
  - iPhone 11    : pefill  52.00 tok/s, decode 16.23 tok/s
  - iPhone 14 Pro: pefill 102.63 tok/s, decode 33.53 tok/s
- Qwen-1.8b-chat 8bit
  - iPhone 11    : pefill  61.90 tok/s, decode 14.75 tok/s
  - iPhone 14 Pro: pefill 105.41 tok/s, decode 25.45 tok/s

---

[新版测试prompt](../resource/bench.txt)
- Qwen1.5-0.5b-chat 4bit
  - iPhone 15 Pro: pefill 282.73 tok/s, decode 51.68 tok/s
- Qwen2-0.5b-instruct 4bit
  - iPhone 15 Pro: pefill 234.51 tok/s, decode 51.36 tok/s
- Qwen2-1.5b-instruct 4bit
  - iPhone 15 Pro: pefill 107.64 tok/s, decode 25.57 tok/s

## 编译
1. 编译 MNN iOS Framework: 在 MNN 根目录下执行
```
sh package_scripts/ios/buildiOS.sh "-DMNN_ARM82=true -DMNN_LOW_MEMORY=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_BUILD_LLM=true"
mv MNN-iOS-CPU-GPU/Static/MNN.framework transformers/llm/engine/ios/MNN.framework
```
2. 下载模型文件: [Qwen1.5-0.5B-Chat-MNN](https://modelscope.cn/models/zhaode/Qwen1.5-0.5B-Chat-MNN/files) ，或者使用 export 下面的脚本导出模型
3. 将模型文件拷贝到`${MNN根目录}/transformers/llm/engine/model/`目录下
4. 在xcode项目属性中`Signing & Capabilities` > `Team`输入自己的账号；`Bundle Identifier`可以重新命名；
5. 连接iPhone并编译执行，需要在手机端打开开发者模式，并在安装完成后在：`设置` > `通用` > `VPN与设备管理`中选择信任该账号；

备注：如测试其他模型，可以将`ios/mnn-llm/model/`替换为其他模型的文件夹；同时修改`LLMInferenceEngineWrapper.m +38`的模型路径；

## 性能
等待模型加载完成后，发送：`benchmark`，即可进行benchmark测试；

## 测试
等待模型加载完成后即可发送信息，如下图所示：

![ios-app](./ios_app.jpg)
