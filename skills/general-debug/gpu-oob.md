# §6 GPU Shader 越界 / Command Buffer 故障

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
>
> **不在本文**：Metal kernel 的写法与优化陷阱清单在 [`metal-optimize/kernel-dev-and-optimize.md`](../metal-optimize/kernel-dev-and-optimize.md)，
> env 开关语义以 [`metal-optimize/env-registry.md`](../metal-optimize/env-registry.md) 现表为准；
> 越界表现为静默数值损坏而非崩溃时，一并对照 [`memory-aliasing.md`](memory-aliasing.md)。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

**触发**（满足以下之一强烈怀疑本类）：
- 运行中出现 `[METAL] command buffer error`（`InnocentVictim` / `SubmissionsIgnored` / GPU restart），此后速度数字**假快数百倍**（forward 实际没跑）；
- 只在**某个 shape 阈值之上**触发（某 kv 长度、某模型 head_dim、某 batch），阈值之下完全正常；
- 关掉某条 kernel 路径的 env 开关后消失；
- ⚠️ 反面：同样的越界在别的模型上可能**不崩而是静默数值损坏**（踩到的是已映射内存）——"没崩"不等于"没越界"。

## 6.1 排查流程（按成本从低到高）

1. **先看第一条错误，别被 victim 骗**：`InnocentVictim`/`SubmissionsIgnored` 都是**受害者**代码，真凶 buffer 常常不在日志里。不要基于 victim 的 op 去猜。
2. **用 env 开关把"路径"与"触发变量"解耦**（判别性探针，代价一次 run）：逐个关可疑路径（如 `MNN_METAL_DECODE_SDPA=0`、`MNN_METAL_DISABLE_REPLAY=1`）看谁消失。⚠️ 注意"关 A 消失"不等于"A 是根因"——replay 常只是放大面；要看**最小共同集**（本案例：0.8B 关 replay 也好，2B 只有关 splitkv 才好 ⇒ 根在 splitkv）。
3. **解耦相关变量**：阈值型触发常有多个共变量（kv 长度 ↔ nwg=ceil(kv/256)）。用 pin 类旋钮做 2×2（当年是 `MNN_METAL_DECODE_SPLITKV_NWG=19/20` × 安全/故障 kv）——本案例一轮就锁定"nwg>16 而非 kv 本身"。⚠️ 本案例的 split-KV 路径及其 `MNN_METAL_DECODE_SPLITKV` / `_NWG` 两个 env **已于 2026-07-30 删除**（收敛到单 pass `MNN_METAL_DECODE_SDPA`）；方法论照用，具体开关名以 `metal-optimize/env-registry.md` 现表为准。
4. **Metal Shader Validation 拿实锤**（最有力，几分钟）：
   ```bash
   MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
   MTL_SHADER_VALIDATION_FAIL_MODE=allow <重现命令，n 可缩到 32>
   ```
   直接报 **kernel 名 + 越界 offset**（`Invalid device store at offset N, executing kernel "xxx"`）。
5. **对 offset 做算术反推**：拿第一个非法 offset 除以已知 stride，反推 kernel 以为的 buffer 尺寸 vs 实际分配尺寸。本案例：非法 offset ≈133120B = `8×32×(128+2)×4B`——正好是"元素数对、字节数减半"，直指 fp16 后端 `createDevice<float>` 按 2B 存储的陷阱（`metal-optimize/kernel-dev-and-optimize.md` 陷阱 F）。
6. **加一次性尺寸日志坐实**（分配处 + dispatch 处各一行，打印 elementSize / MTLBuffer.length / 索引参数），修复后删除。
7. **修复验证矩阵**：validation 0 OOB + 原故障配置 e2e 零错误 + greedy 对拍（⚠️ 用 metal+greedy config，见 `metal-optimize/build-and-test.md` Step 1.5——本案例曾被默认 mixed-sampler config 污染出一个假 bug）+ `run_test.out` 全过。

## 6.2 测试覆盖教训

阈值型 bug 能长期潜伏是因为**测试矩阵恰好停在结构阈值上**：splitkv 的 nwg 在 kv=4096 时恰为 16（越界临界），而历史性能/对拍全部 ≤p2048~p4096。**改 kernel 后的覆盖至少要跨过它的每个结构常数边界**（nwg cap、tile 对齐、tg mem 档位），各取"边界±1"各测一档。

## 6.3 参考案例：split-KV partial buffer 半长分配（2026-07-29，`6975fa71e7`）

`mTempSplitKV` 用 `createDevice<float>` 分配、shader 按 `device float*` 写：fp16 后端下存储 2B/元素 ⇒ buffer 半长，nwg>16（kv>4096）越界。HD=256（Qwen3.5）撞未映射页 → GPU 故障链；HD=128（Qwen3）同条件仅静默损坏。修复 = 按字节分配（`createDevice<uint8_t>`，公式显式 `* sizeof(float)`）。完整陷阱条目见 `metal-optimize/kernel-dev-and-optimize.md` 陷阱 F。
