# 步骤 1：优化 Kernel 性能

> **目标**：基于基线，尝试多种优化手段提升 kernel（或调度）性能。
>
> **前置**：步骤 0 已通过（有基线 + 已判 CPU/GPU 瓶颈）。
>
> **复杂度**：高。
>
> **参考**：shader 修改/正确性验证见 `SKILL.md`；技巧与陷阱见 `optimization-handbook.md`。

---

## 1.0 分析瓶颈

先按 `optimization-handbook.md` §1 定位：
1. **CPU 调度 vs GPU kernel**（§1.4）——Vulkan 必做，别跳。CPU-bound 直接去技巧 6/7 + §6，不要在 kernel 上耗。
2. GPU-bound 时再算计算强度 / BW 利用率 / occupancy（§1.1–1.3）。

```markdown
## Kernel 性能分析
**shader**: glsl_xxx_comp   **位置**: .../glsl/xxx.comp   **占 GPU**: xx% / xx ms
**瓶颈**: CPU 调度 / memory-bound / compute-bound / occupancy 墙
**代码级**:
- 访存：global 读写次数？重复读？连续（合并）？可否 shared 缓存？
- 计算：私有数组大小？循环嵌套/依赖？并行粒度？
- workgroup：local_size / dispatch grid？subgroup size 是否 hardcode？
- occupancy：用了多少 shared？寄存器 tile 多宽？
```

---

## 1.1 优化策略选择

**先按瓶颈选杠杆**（判断表以 handbook §1 为准，不重复）。下面按**症状**导航：

```
瓶颈在 CPU 调度？
  ├─ submit/命令录制多 → indirect batch（技巧 7）
  ├─ per-op uniform 开销 → push_constant（技巧 6）
  └─ 同 shape 反复 encode → fixResizeCache（§6 候选 A）/ 算子融合减 op
GPU: global 访存频繁？
  ├─ 写非合并（scatter）→ epilogue 合并写（技巧 3）
  ├─ 多 dispatch + temp 往返 → 融合 epilogue（技巧 4，注意 occupancy）
  └─ 重复读 → shared 缓存（注意 occupancy，陷阱 I）
GPU: matmul 且 Adreno + coop？→ cooperative matrix 重写（技巧 1）
GPU: 归约有 barrier？→ subgroup 归约（技巧 2）
GPU: occupancy 墙（融合用了大 shared）？→ 按规模门控融合/分离（技巧 5）
```

技巧的完整定义（做法/注意/收益/案例）一律以 `optimization-handbook.md` §2 + §5 速查表为准，改前读对应条目。

---

## 1.2 实施优化

```
1. 记录当前性能
2. 实施单个优化（只改一个点）
3. 更新 host 调用（Execution.cpp：spec constant/push constant/dispatch grid/pipeline 选路）
4. 外科式重生成改动 shader 的数组进 AllShader.cpp（SKILL.md「Shader 修改流程」，勿跑全量 makeshader）
5. 编译（make llm_demo）→ 推真机 → rm mnn_cachefile.bin → 正确性验证 → 交替 A/B 测速
6. 提升→保留 / 下降→回退 / 正确性失败→修复
```

> kernel 内用 `FLOAT/FLOAT4` 宏（fp16/fp32 兼容），不用裸 `float4`。coop/subgroup shader 记得 `--target-env vulkan1.1`（陷阱 D）。

---

## 1.3 性能验证（交替 A/B）

正确性（三层 oracle，见 SKILL.md）+ 性能。**性能必须交替 A/B**（base↔opt 背靠背配对，每轮清 `mnn_cachefile.bin`），看每轮胜负而非两组绝对值（热漂移 ~±10%，陷阱 G）。

```markdown
## 优化尝试X: [名称]
**方案**: ...   **修改**: xxx.comp, XxxExecution.cpp
| 轮 | base(tok/s) | opt(tok/s) | opt 胜? |
|---|---|---|---|
| 1 | xx | xx | ✓ |
...
**决策**: 保留 / 回退（记录为什么不 work）
```

---

## 1.4 迭代优化（至少 3 种）

不能只试一种。从 handbook §5 速查表按难度从低到高、按症状挑，至少 3 种：
- **先手（低）**：push_constant（技巧 6）、indirect batch（技巧 7）、epilogue 合并写（技巧 3）
- **再上（中）**：subgroup 归约（技巧 2）、融合 epilogue（技巧 4）+ 按规模门控（技巧 5）
- **最后（高）**：cooperative matrix 重写（技巧 1）、算子整体重写 / 组合多种

**停止条件（全满足）**：已试 ≥3 种 / 达标或连续 3 次 <5% 提升 / 每次都有 A/B 数据记录。

---

## 通过标准

- [ ] 已试 ≥3 种不同技术
- [ ] 每次都有正确性验证（与 baseline 逐 token 一致或误差内）
- [ ] 每次都有交替 A/B 性能数据
- [ ] 最终相对基线有显著提升（或已确认瓶颈在 CPU 调度、kernel 优化无效并转向调度）
- [ ] 优化记录已文档化
