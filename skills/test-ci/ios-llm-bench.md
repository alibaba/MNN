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
| `--backend cpu\|metal --prompt-len N --decode-len N` | 定长 bench，不带则跑 `bench.txt` prompt 文件模式。⚠️ 语义等价于桌面端 **`llm_bench -pg P,D`**（prefill P 个 token 后**复用该 KV cache** 续写 D 个 token，prefill/decode 分开计时），**不是** `-p P -n D`（那是两个独立的 prefill-only / decode-only 测试，其 decode 从 kv≈0 起算、数值偏高） |
| `--prompt-len 512,1024,2048 --decode-len 128,2000` | 逗号分隔的多组长度，自动跑全组合矩阵（此例 3×2=6 组），framework/App 只构建安装一次，每组独立日志 `bench_*_p<P>_d<D>.log`，报告汇总所有 avg 行 |
| `--repeat N` / `--threads N` | 定长 bench 轮数（首轮 warmup 不计入）/ CPU 线程数 |
| `--skip-framework` | 复用现有 `ios/MNN.framework`，跳过 C++ 编译（对比测试时关键） |
| `--build-only` / `--device UDID` / `--cmake-args "…"` / `--timeout SEC` | 只编不装 / 指定设备 / 额外 CMake 参数 / 超时（默认 1800s，按单组计） |

## App 内 bench 协议

`LLMInferenceEngineWrapper.mm` 支持命令式 benchmark，结果以 NSLog 标记输出供脚本抓取：

- 启动参数 `--auto-bench`：加载模型后自动跑 `bench.txt` prompts
- 启动参数 `--bench-cmd "bench metal 512 128 3 4"`：定长 bench，格式 `bench <cpu|metal> <prompt_len> <decode_len> [repeat] [threads] [attention_mode]`；聊天框手动输入同样生效
- 输出标记：`[MNN_BENCH] run=… prefill_tok_s=… decode_tok_s=…`、`[MNN_BENCH] avg …`、`[MNN_BENCH_DONE]`、`[MNN_BENCH_ERROR] <原因>`

## 长矩阵的断点续跑与设备恢复

多模型、多提交点的 iOS App 往往超过 1 GB，断线后从头安装/重跑既慢又会改变热状态。长矩阵 runner 应满足：

1. 每个正式样本单独落日志；只有同时出现 `MNN_BENCH_DONE`、Metal `avg`，且没有 error/fallback 标记时才算有效。
2. 重启时先检查日志；同一 stage 的所有上下文都有效就直接跳过整个 stage，**不要重复安装 App**。只有部分有效时才安装一次并补缺失上下文。
3. install/launch 显式设置上限（大 App 可用 `--timeout 300` / `--timeout 600`）。runner 自己也要轮询完成标记并清理本机 `devicectl` 子进程，不能无限等待。
4. 未完成日志可以覆盖重跑，绝不能从只有 `model_loaded` 或部分 `run=` 的日志提取结果。

当 `devicectl --console` 卡在 `Acquired usage assertion`，或 App 已打印 `model_loaded` 但长期不进入 bench 时，按以下顺序恢复：

```bash
xcrun devicectl list devices
xcrun devicectl device info lockState --device <id>
xcrun devicectl device info processes --device <id>
```

- 设备是 `connected` 但存在旧 benchmark PID：先用 `device process terminate --pid <pid> --kill` 清理残留，再结束本机无响应的 `devicectl` wrapper，运行一次 prompt=64 的短 smoke。
- `devicectl` 显示 `unavailable`、`xctrace list devices` 同时列为 Offline：这是设备通道问题，不要继续重试 benchmark。保持设备解锁，检查 Finder/USB 是否能看到设备，恢复数据连接后再续跑。
- 恢复后的短 smoke 必须重新看到 Metal avg 与 `MNN_BENCH_DONE`；通过后才继续正式矩阵。

## 分支性能对比方法

同一设备、同一模型、同样定长参数下依次测各分支：

0. 先确认模型是用两个分支共同支持的 schema/converter 导出的；若对比范围包含 Op schema 变更，应从共同基线（或新格式分支）重新导出一次，再让所有分支共用该导出物。
1. 分支 A：正常跑一次（脚本自动记录 branch + commit 到报告头）
2. 分支 B：`git stash` 携带 bench 基建改动切分支（或直接在另一份 checkout 里编 framework，`rsync -a --delete` 覆盖 `ios/MNN.framework/` 后用 `--skip-framework`）
3. 对比 `bench_logs/` 中各次 avg 行；3 轮定长 bench 波动通常 <1%，可直接比较

## 已知陷阱

- **shell 环境的 `SDKROOT` / `CPATH` 指向 MacOSX.sdk 会打爆整个 iOS 编译**（2026-07-30 实锤）：症状是几百个 `<cstddef> tried including <stddef.h> but didn't find libc++'s <stddef.h>`，libc++ 头来自 iPhoneOS.sdk 而 C 头来自 MacOSX.sdk。且 `buildiOS.sh` 失败后仍 `exit 0`，只留下残缺 framework（仅 Headers + Info.plist，无二进制、`Headers/llm` 为空），下游 App 编译报 `MNN/llm/llm.hpp not found` 误导排查方向。**处置**：跑本脚本一律 `env -u SDKROOT -u CPATH DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer sh ios_llm_bench.sh ...`；怀疑 framework 残缺时先 `ls MNN.framework/MNN` 确认二进制存在。
- **Team ID 必须查本机，不能抄文档**：`--team` 用错会报 `No Account for Team`。查法：`security find-identity -v -p codesigning` 看证书，或 `security cms -D -i ~/Library/Developer/Xcode/UserData/Provisioning\ Profiles/*.mobileprovision | grep -A2 TeamIdentifier`。本机（jbyang，2026-07-30）为 `3TLG5LZ643`。
- **Personal Team 可能没有可列出的 codesign identity**：`security find-identity` 返回 0 时，可从 `defaults read com.apple.dt.Xcode` 的 `IDEProvisioningTeamByIdentifier` 找当前 Personal Team，再用 `codesign -dvvv <app>` 核对实际 `TeamIdentifier`。复用已安装的 bundle id 时，新旧 App 的 Team 也必须一致，否则会报 `MismatchedApplicationIdentifierEntitlement`。
- **免费开发者证书每台设备最多 3 个 App**：安装报 `CoreDeviceError 3002` + `maximum number of installed apps using a free developer profile`，错误信息会列出占位的 3 个 bundle id。**处置**：优先用 `--bundle-id` 复用其中同 Team 的旧 bench App 原地覆盖（如 `com.jiuqi.mnn-llm-bench`），不必删设备上的 App。
- **schema 变更后的旧模型不能用于性能对比**：旧 FlatBuffer 可能不会给出清晰的“版本不兼容”错误，而是在加载阶段表现为 `std::bad_alloc` / `SIGABRT` / `SIGSEGV`。若多个分支都未进入 `[MNN_BENCH] run=...` 就崩溃，先用当前 schema 和 `MNNConvert` 重导模型，并检查 `export_args.json` 及 `llm.mnn.json` 的融合 Op，不要将加载崩溃误判为性能回归。
- **签名未信任的启动失败是"秒失败"，但脚本会傻等满 `--timeout`（默认 1800s）**：换新 bundle id / 新 profile 首次安装后必须在 iPad 上手动信任（设置 > 通用 > VPN与设备管理 > 开发者App）。日志特征：`FBSOpenApplicationErrorDomain error 3` + `its profile has not been explicitly trusted by the user`。看到 TIMEOUT 先翻 `bench_logs/*.log` 头部有没有这个错误，别真等 30 分钟。
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

## 参考基线（iPad Pro 11" M5 · iPad17,1，Q4 b64，metal，prompt=512 / decode=128 / repeat=6，`fecee95475`，2026-07-30）

| 模型 | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Qwen3-0.6B | 7795.2 | 241.7 |
| Qwen3.5-0.8B | 3017.5 | 195.9 |
| Qwen3.5-2B | 1942.8 | 95.8 |
| Qwen3-4B | 1231.1 | 47.9 |

对照同分支 M4 Pro Mac（b64 同口径）：iPad M5 prefill 全面更高（+14%~+74%，tensor-API/NAX 生效），decode 全面更低（−30%~−35%，内存带宽约减半）。与 M5 Mac 参考值（0.6B prefill ~7488 / decode ~227）同量级。
