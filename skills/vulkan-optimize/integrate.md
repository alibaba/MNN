# 步骤 2：验证和集成

> **目标**：把优化集成进 MNN 主体，跑全量回归，确保不影响其他算子/后端。
>
> **前置**：步骤 1 已通过。
>
> **复杂度**：中。
>
> **参考**：编译/推送/正确性见 `SKILL.md`。

---

## 2.1 全量回归测试

编译 + 推送见 `SKILL.md`。**换 shader 后先 `rm tmp/mnn_cachefile.bin`**。

```bash
# 优化算子单测（Vulkan = forward type 7）
adb -s <serial> shell "cd /data/local/tmp/MNN && LD_LIBRARY_PATH=. ./run_test.out op/XxxTest 7 <precision> <numThread>"
# 全量 op
adb -s <serial> shell "cd /data/local/tmp/MNN && LD_LIBRARY_PATH=. ./run_test.out op/ 7 <precision> <numThread>"
```

**通过标准**：`all tests passed`。

端到端（关 sampler 随机性 `temperature:0.0`）：

```bash
adb -s <serial> shell "cd /data/local/tmp/MNN && rm -f tmp/mnn_cachefile.bin && \
  LD_LIBRARY_PATH=. ./llm_demo <model>/config_vk.json prompt.txt <ndecode> 2>&1 | tail -20"
```

**跨模型规模验证**：优化可能对小/大模型收益相反（如融合 epilogue，handbook 技巧 5）。至少在**一个小模型 + 一个大模型**（如 0.6B + 4B）上各测一遍，确认没有一个规模退步。

---

## 2.2 代码质量审查

### 通用
```
□ .hpp 注释与 .cpp 实现一致
□ 无声明未用变量 / 残留调试代码（MNN_PRINT、临时 env dump、注释掉的旧码）
□ .comp 注释与实际代码一致
```

### Vulkan 特定
```
□ 只外科式重生成了改动 shader 的数组，AllShader.cpp 无关数组未动（git diff --stat 确认）
□ 新增 shader 在 AllShader.h + VulkanShaderMap.cpp + macro.json 三处都注册
□ coop/subgroup shader 用了 --target-env vulkan1.1；SPIR-V 过 spirv-val
□ subgroup 与 nosubgroup 双变体都改（陷阱 K）
□ local_size 按 getSubgroupSize() 动态设，未 hardcode
□ shared 用量评估过 occupancy 影响（陷阱 I）；大规模有分离/fallback 路径
□ dispatcher 对未支持路径显式 fallback（coop 只 int4/8，陷阱 H）；buffer size 按 bit 重算（陷阱 M）
□ 阈值/开关做成 env 可调（如 MNN_VK_CONV_FUSE_MAXN）
```

### 设计合理性
```
□ 多个小 dispatch 可否合并？（Vulkan 命令录制重，收益大）
□ 有无不必要的 temp buffer 往返 / CPU<->GPU 拷贝？
□ Fallback 完善？（不支持 coop/subgroup 的设备、超大 shape）
□ buffer / image 两后端是否都要改（还是显式声明仅 buffer）？
```

---

## 2.3 性能报告

**报告文件**：`<算子名>_vulkan_optimization.md`，实测数据，不写"预期"。

```markdown
# Xxx Vulkan 性能优化报告
## 1. 概述：对象、平台、优化前/后、加速比、主要技术
## 2. 瓶颈定位：CPU 调度 vs GPU kernel 的判断依据（GPU 累计 vs wall）
## 3. 性能数据（实测，交替 A/B）：基线 vs 优化后（按场景/模型规模分组，含 shader 耗时）；各手段贡献；优化历程
## 4. 正确性：op 测试、全量回归、端到端逐 token 对比结果
## 5. 代码质量：外科式 shader 重生成确认、三处注册、双变体、occupancy、fallback
## 6. 修改文件清单
## 7. 技术细节：关键优化的实现/收益/限制；不同模型规模差异原因
## 8. 未优化 / 后续方向
## 9. 经验总结：成功经验、踩的坑、建议
```

---

## 2.4 提交前检查

```bash
git status   # 确认包含：
# - .comp（kernel）+ 新增 matmul_coop_rm.comp / COOP_to_C4.comp 等
# - AllShader.cpp / AllShader.h / VulkanShaderMap.cpp（只改动数组 + 注册）
# - XxxExecution.cpp/.hpp（host 选路/dispatch）
# - <算子>_vulkan_optimization.md（报告）
```

> **clang-format pre-commit hook**：MNN 有 clang-format-diff 钩子，只查改动行。手写 .cpp/.hpp 被拦时用 `git clang-format HEAD` 只格式化改动行，但**不要让它重排 AllShader.cpp 的字节数组**（`git checkout` 掉对生成文件的 reflow，保留 xxd 格式）。

提交信息：
```
[Vulkan:Perf] Optimize Xxx (coalesce epilogue / gate fusion by N / ...)

- 技术1
- 技术2

Performance: 0.6B +x% / 2B +x% / 4B +x% prefill (Adreno, interleaved A/B)
All op/ tests passed
```

---

## 2.5 沉淀经验到手册

产生了可复用方法论（新技巧 / 非显而易见的坑 / 新分析方法 / 验证或排除某方向）时回写 `optimization-handbook.md`（唯一来源）。触发判断表、回写位置见 `SKILL.md`「收尾」。单次任务的具体数字/文件清单留在 2.3 报告里，不进手册；纯套用无新经验则跳过。

---

## 通过标准

- [ ] 全量 op/ 通过
- [ ] 小 + 大模型各测，无规模退步
- [ ] 性能报告完整（9 章 + 实测交替 A/B 数据）
- [ ] 代码质量审查通过（含 shader 三处注册 / 双变体 / occupancy / fallback）
- [ ] AllShader.cpp 只有目标数组变（无污染）
- [ ] 可复用经验已回写手册（或确认无）
