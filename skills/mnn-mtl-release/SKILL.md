---
name: mnn-mtl-release
description: MNN MTL CLOUD 自动打包发布。支持 Android/iOS 平台，自动创建迭代、添加模块、触发构建、跟踪状态。支持 SNAPSHOT 和 Release 包，按模块依赖顺序编排。
---

# MNN MTL 打包发布 SKILL

> **触发条件**：当用户请求在 MTL 上打包、发布、集成 MNN 模块时触发。常见表述包括："帮我打个 Android 包"、"MTL 打包"、"打 iOS SNAPSHOT"、"发布 MNN 正式版"等。

## 概述

本 SKILL 指导 AI Agent 在 MTL CLOUD 上完成 MNN 模块的打包发布流程。通过 `aone-kit call-tool mc-cloud-server::xxx` 调用 MTL API，自动创建迭代、添加模块、触发构建，并在关键决策点与用户交互。

### 核心原则

1. **按依赖顺序打包**：基础库先打，依赖库后打
2. **关键操作前确认**：创建迭代、触发构建、提交集成等写操作前必须向用户确认
3. **构建状态跟踪**：触发构建后主动查询状态，失败时拉取错误日志分析
4. **灵活编排**：不强制执行全部步骤，用户可按需选择打哪些模块、走到哪一步

## 平台与模块信息

### Android（客户端 appId=1，手机淘宝Android客户端）

| 模块 | moduleId | 代码仓库 | 依赖关系 |
|------|----------|---------|---------|
| AliNN4Android | 8120 | AliNNPrivate.git | 基础库，无依赖 |
| MNNPyBridge4Android | 12692 | AliNNPrivate.git | 依赖 AliNN，通过构建参数 `-PALINN_VERSION=xxx` 指定版本 |
| AliNNKitCore4Android | 8672 | AliNNKitCore.git | 依赖 AliNN，需修改 `android/build.gradle` 第78行 `alinnVersion` |

**Android 打包顺序**：AliNN4Android → MNNPyBridge4Android / AliNNKitCore4Android

**注意**：
- MNNPyBridge4Android 不需要改代码，通过 MTL 构建参数传递 AliNN 版本号
- AliNNKitCore4Android 在独立仓库（/Users/wangzhaode/workspace/AliNNKitCore），需要修改 `android/build.gradle` 中的 `alinnVersion` 变量
- MNNPyBridge4Android 还有 `BUILD_LLM` 选项（true/false），控制是否构建 LLM 功能

### iOS（客户端 appId=6，手机淘宝iOS客户端）

| 模块 | moduleId | 代码仓库 | 依赖关系 |
|------|----------|---------|---------|
| MNN | 7270 | AliNNPrivate.git | 基础库，无依赖 |
| MNNBridge | 12532 | AliNNPrivate.git | Pod 依赖 MNN，需修改 Podfile |

**iOS 打包流程**（以当前分支 A 为例）：

1. 在分支 A 上打 MNN SNAPSHOT 包
2. 如需打 MNNBridge：
   - 从 A checkout 新分支 `A_bridge`（用于修改 Podfile 依赖）
   - 修改 `pymnn/iOS/MNNPyBridge/Podfile` 中 MNN Pod 版本为步骤1的 SNAPSHOT 版本
   - 可直接用 `A_bridge` 打 SNAPSHOT 包
   - 如需打正式包：创建 `A_release` 分支，在 aone 上提 CR 将 `A_bridge` 合并到 `A_release`（正式包分支必须通过 CR）
3. 在 `A_release` 分支上可打 SNAPSHOT 或 Release 包

**Podfile 位置**：`pymnn/iOS/MNNPyBridge/Podfile`，修改 `pod 'MNN', 'xxx'` 行的版本号。

## 执行流程

### Step 1：收集打包意图

启动后询问/推断以下信息（已知的不重复问）：

- **平台**：Android / iOS
- **分支**：默认使用当前 git 分支
- **模块范围**：打哪些模块
- **包类型**：SNAPSHOT / Release（影响迭代类型和分支策略）
- **后续流程**：是否需要提测、发布正式版、集成

### Step 2：创建迭代

1. 确认迭代名称（建议格式：`{分支名}_{平台}_{日期}`）
2. 调用 `mtl4_create_normal_iteration`（自助验证）或 `mtl4_create_ci_iteration`（持续集成，需版本计划）
3. 返回迭代链接：`https://mc.alibaba-inc.com/#/iterations/alterSheet/detail?entityId={迭代id}`

### Step 3：按顺序打包模块

对每个模块执行：

1. **添加模块到迭代** — `mtl4_add_module_in_iteration`，指定分支和变更类型（SOURCE）
2. **多模块时重置流水线** — 迭代下 ≥2 个模块时，先调用 `mtl4_reinit_iteration_pipelines`
3. **触发构建** — `mtl4_execute_iteration_mixed_pipeline`
4. **跟踪构建状态** — 用 `mtl4_find_module_latest_pipeline_instance` 或 `mtl4_find_pipeline_instance_record` 查询
5. **构建失败处理** — 用 `mtl4_find_pipeline_instance_error_log` 拉取错误日志，分析原因并汇报
6. **构建成功** — 汇报版本号，询问是否继续打下一个模块

### Step 4：后续流程（按需）

根据用户需求执行：

- **提测** — `mtl4_ci_submit_test`（需要测试验证人和提测说明）
- **发布正式版** — `mtl4_execute_iteration_module_deploy_pipeline`
- **提交集成** — `mtl4_ci_create_continuous_integrate_sheet` + `mtl4_ci_submit_continuous_integrate_sheet`

每步完成后询问用户是否需要下一步操作。

## 常用 MTL API 速查

| 操作 | API |
|------|-----|
| 创建自助验证迭代 | `mtl4_create_normal_iteration` |
| 创建持续集成迭代 | `mtl4_create_ci_iteration` |
| 添加模块到迭代 | `mtl4_add_module_in_iteration` |
| 重置流水线（多模块） | `mtl4_reinit_iteration_pipelines` |
| 触发混合流水线 | `mtl4_execute_iteration_mixed_pipeline` |
| 触发模块正式发布 | `mtl4_execute_iteration_module_deploy_pipeline` |
| 查询构建状态 | `mtl4_find_module_latest_pipeline_instance` |
| 查询错误日志 | `mtl4_find_pipeline_instance_error_log` |
| 查询版本计划 | `mtl4_find_version_plan` |
| 创建提测单 | `mtl4_ci_submit_test` |
| 创建集成单 | `mtl4_ci_create_continuous_integrate_sheet` |
| 提交集成单 | `mtl4_ci_submit_continuous_integrate_sheet` |
