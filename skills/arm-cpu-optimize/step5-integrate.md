# 步骤 5：集成、回归与交付记录

> **目标**：确认新实现和 runtime dispatch、packer、测试矩阵一致，并留下足够的性能/正确性记录。

---

## 5.1 集成检查

新增或替换 kernel 时检查：

- asm symbol 和 C++ 声明完全一致。
- 新 `.S` 文件已加入对应 build list。
- 函数指针只在对应 ISA 支持时注册。
- runtime disable flag 能走到预期 fallback。
- `MNNGetGemmUnit`、`UNIT/SRC_UNIT/DST_XUNIT` 和 packer 匹配。
- low bit packer、online reorder、mixed reorder 没有喂错 kernel。
- block32、block64、per-channel、tail 保留安全路径。

SME2 或其他可变 pack mode 路径尤其要确认：不要用新 packer 配旧 kernel。

---

## 5.2 回归测试

按改动范围选择测试，不要盲目承诺全量但也不要只跑一个 case。

低 bit LLM kernel 推荐：

```bash
cd build
cmake --build . -j 8

./run_test.out op/lowMemory/blockConv 0 1 4
./run_test.out op/lowMemory/HybridConv 0 1 4
./run_test.out op/lowMemory/blockConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑
./run_test.out op/lowMemory/HybridConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑

./run_test.out speed/GemvBW 0 2
```

模型 sanity：

```bash
./llm_demo /path/to/w3/config.json prompt.txt 64 1
./llm_demo /path/to/w3/config.json prompt.txt 64 1  # 在 SDOT 目标设备/构建配置上复跑
./llm_demo /path/to/w2/config.json prompt.txt 64 1
./llm_demo /path/to/w2/config.json prompt.txt 64 1  # 在 SDOT 目标设备/构建配置上复跑
```

普通算子则跑对应 `op/Xxx`、相关邻近算子和已有 speed test。

---

## 5.3 质量检查

提交或交付前确认：

- 没有残留 debug print、临时 dump、未使用变量。
- `.hpp` 注释和 `.cpp` 实现的 layout/size 描述一致。
- onResize buffer acquire/release 配对。
- C++ oracle/debug 代码若不应提交，已移除或放入测试。
- 没有回退或覆盖用户无关改动。
- 没有读取、依赖或修改 `schema/private/`、`source/internal/`。

---

## 5.4 记录结果

不强制每次创建长报告；但非平凡性能优化必须在最终回复或报告文件里记录关键数据。

推荐格式：

```markdown
## ARM CPU optimization result

Scope:
- bit/precision:
- ISA:
- shape/block:
- threads:

Correctness:
- build:
- op tests:
- model sanity:

Performance:
| test | bit | ISA | before | after | eff GB/s | note |
|------|-----|-----|--------|-------|----------|------|

Notes:
- paths not covered:
- residual risk:
```

如果用户要求报告文件，或数据较多，再写入 `<topic>_optimization.md`。

---

## 通过标准

- build 通过。
- 目标路径和 fallback 路径的 focused correctness tests 通过。
- 模型级 sanity 覆盖会影响 LLM 输出的低 bit kernel。
- 性能数据能解释本次改动是否有效。
- 已明确未覆盖路径和剩余风险。
