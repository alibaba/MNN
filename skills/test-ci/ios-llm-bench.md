# iOS 真机 LLM Benchmark

> **触发**：需要在 iPhone/iPad 真机上测 LLM 性能（prefill/decode tok/s）；对比两个分支/两份代码在 iOS 上的 Metal/CPU 性能；验证 Metal kernel 改动在真机上的效果。
>
> **前置条件**：Mac 安装完整 Xcode（非仅 CommandLineTools）；iPhone/iPad USB 连接、开启开发者模式并信任本机；有 Apple Development Team ID（首次装机后需在手机 设置 > 通用 > VPN与设备管理 中信任证书）；模型已导出为 MNN 格式（含 `config.json` / `llm.mnn`）。

## 一键测试

```bash
sh transformers/llm/engine/ios/ios_llm_bench.sh \
    --model /path/to/EXPORTED_MNN_MODEL \
    --team YOUR_TEAM_ID \
    --backend metal --prompt-len 512 --decode-len 128
```

脚本流程（全自动，无需手动操作手机）：

1. `package_scripts/ios/buildiOS.sh` 编译含 LLM 的静态 `MNN.framework`（`-DMNN_LOW_MEMORY=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_BUILD_LLM=true`）
2. 模型拷入 `mnn-llm/model/` 随 App 打包
3. `xcodebuild` 编译签名 mnn-llm.app（无模拟器 runtime 时自动去掉 asset catalogs 重试）
4. `xcrun devicectl` 选设备并安装
5. `devicectl device process launch --console` 带参数启动，轮询日志中的 `MNN_BENCH_DONE` / `MNN_BENCH_ERROR` 标记，汇总输出报告；日志存 `bench_logs/bench_*.log`

常用选项：

| 选项 | 说明 |
|---|---|
| `--backend cpu\|metal --prompt-len N --decode-len N` | 定长 bench（类似 `llm_bench`），不带则跑 `bench.txt` prompt 文件模式 |
| `--prompt-len 512,1024,2048 --decode-len 128,2000` | 逗号分隔的多组长度，自动跑全组合矩阵（此例 3×2=6 组），framework/App 只构建安装一次，每组独立日志 `bench_*_p<P>_d<D>.log`，报告汇总所有 avg 行 |
| `--repeat N` / `--threads N` | 定长 bench 轮数（首轮 warmup 不计入）/ CPU 线程数 |
| `--skip-framework` | 复用现有 `ios/MNN.framework`，跳过 C++ 编译（对比测试时关键） |
| `--build-only` / `--device UDID` / `--cmake-args "…"` / `--timeout SEC` | 只编不装 / 指定设备 / 额外 CMake 参数 / 超时（默认 1800s，按单组计） |

## App 内 bench 协议

`LLMInferenceEngineWrapper.mm` 支持命令式 benchmark，结果以 NSLog 标记输出供脚本抓取：

- 启动参数 `--auto-bench`：加载模型后自动跑 `bench.txt` prompts
- 启动参数 `--bench-cmd "bench metal 512 128 3 4"`：定长 bench，格式 `bench <cpu|metal> <prompt_len> <decode_len> [repeat] [threads] [attention_mode]`；聊天框手动输入同样生效
- 输出标记：`[MNN_BENCH] run=… prefill_tok_s=… decode_tok_s=…`、`[MNN_BENCH] avg …`、`[MNN_BENCH_DONE]`、`[MNN_BENCH_ERROR] <原因>`

## 分支性能对比方法

同一设备、同一模型、同样定长参数下依次测各分支：

1. 分支 A：正常跑一次（脚本自动记录 branch + commit 到报告头）
2. 分支 B：`git stash` 携带 bench 基建改动切分支（或直接在另一份 checkout 里编 framework，`rsync -a --delete` 覆盖 `ios/MNN.framework/` 后用 `--skip-framework`）
3. 对比 `bench_logs/` 中各次 avg 行；3 轮定长 bench 波动通常 <1%，可直接比较

## 已知陷阱

- **设备锁屏**：锁屏时 `devicectl` 无法启动 App（FBSOpenApplicationErrorDomain error 7 "Locked"），脚本会立即报 `app failed to launch (device locked?)`。测试前保持屏幕解锁（建议 设置 > 显示与亮度 > 自动锁定 设为"永不"）。
- **iOS 26.5 Metal4 Tensor API 探测（本 skill 相关 bugfix）**：MPP `matmul2d` 要求 M/N 至少一个是 16 的倍数、静态 K 是 16 的倍数。探测 kernel 描述符需用 `(16, 8, dynamic_extent)`；同时 `MetalAttentionShader.hpp` 中 legacy 16x16x8 tensor 路径（静态 K=8）必须保持禁用（宏 `MNN_METAL_TENSOR_OPS_LEGACY_8X8`），否则探测通过但运行时反复编译失败，prefill 反而大幅回退（953 → 717 tok/s）。完整修复后 tensor API 生效，prefill 953 → 1884 tok/s（Qwen3.5-2B，prompt=512）。
- **GPU 开关**：通过 `devicectl` 启动时 App 处于 Inactive 状态，Metal backend 若在此时创建，必须监听 `UIApplicationDidBecomeActiveNotification`（而非 WillEnterForeground）才能恢复 GPU，否则 bench 卡死。
- **xcode-select 指向 CommandLineTools**：cmake iOS toolchain 会报 `get_filename_component` 错误；脚本已自动设置 `DEVELOPER_DIR`，手动编译时需 `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`。
- **无模拟器 runtime**：Xcode 15+ actool 编译 asset catalogs 需要 iOS 模拟器 runtime，脚本检测到后自动排除 xcassets 重编（App 无图标，不影响测试）。

## 参考基线（iPhone 17 Pro 级设备，Qwen3.5-2B Q4，metal, prompt=512 / decode=128）

| 代码 | Prefill tok/s | Decode tok/s |
|---|---|---|
| master（tensor API 探测失败被禁用） | ~953 | ~86 |
| master + tensor API 探测/shader 修复 | ~1884 | ~86 |
| feature/linear-attn-opt-metal | ~2253 | ~89 |
