# AI Agent 协作指南

本文面向使用 Claude Code、Codex、Qoder 等 AI Agent 工具参与 MNN 开发的贡献者。
目标是让 Agent 以渐进式披露的方式读取仓库信息，先理解任务边界，再读取必要文档和代码，避免全量扫描、误读内部目录或做无关改动。

## 入口顺序

Agent 进入仓库后，建议按以下顺序读取上下文：

1. `AGENTS.md`：仓库级规则、架构概览、代码风格、测试要求。
2. `docs/index.rst`：文档站目录，判断当前任务属于构建、推理、转换、贡献、性能还是测试。
3. 与任务直接相关的 `docs/` 页面：只读取当前任务需要的文档。
4. 与任务匹配的 `skills/*/SKILL.md`：如果任务命中 skill，再按 skill 中的步骤文档继续读取。
5. 相关源码和测试：在明确目标后再读取代码，优先用 `rg` 定位。

不要一开始全量读取 `docs/`、`skills/` 或整个源码树。MNN 代码量较大，全量扫描会增加噪声，也更容易把旧文档、示例代码或无关 backend 当成当前任务依据。

## 任务路由

常见任务可以按下表选择入口：

| 任务 | 优先读取 |
|------|----------|
| 新增或修改算子 | `skills/add-new-op/SKILL.md`、`docs/contribute/op.md`、`docs/testing.md` |
| LLM 模型适配或导出 | `skills/support-new-llm/SKILL.md`、`docs/transformers/llm.md` |
| ARM CPU 性能优化 | `skills/arm-cpu-optimize/SKILL.md`、`docs/perf/README.md` |
| OpenCL / Metal / Vulkan 优化 | 对应 backend skill、`docs/perf/README.md`、相关 backend 文档 |
| 测试、CI、测试阶段调整 | `skills/test-ci/SKILL.md`、`docs/testing.md` |
| C++ 推理用法 | `docs/start/quickstart_cpp.md`、`docs/inference/module.md`、`docs/inference/session.md` |
| Python 推理用法 | `docs/start/quickstart_python.md`、`docs/start/python.md`、`docs/pymnn/MNN.md` |
| 构建问题 | `docs/compile/engine.md`、`docs/compile/cmake.md`、`docs/faq.md` |
| 文档修改 | `docs/index.rst`、目标文档所在目录、本文档 |

如果文档和代码表现不一致，应以当前源码和测试结果为准，并在修改中同步更新相关文档或说明剩余差异。

## 读取原则

Agent 读取上下文时应遵循以下原则：

- 先读入口，再读细节：先确认任务类型、模块边界和验证方式，再进入实现文件。
- 按路径收敛：从 `docs/index.rst` 或 skill 入口跳到具体文档，不跨目录随意扩展。
- 按证据推进：每次读取应服务于一个明确问题，例如 API 用法、测试入口、backend 注册方式或性能指标。
- 保持最小改动：只修改完成任务所需的文件，不顺手重构无关代码。
- 区分指南和参考：`docs/start/`、`docs/inference/` 更偏使用指南；`docs/cpp/`、`docs/pymnn/` 更偏 API 参考；`skills/` 是 Agent 执行流程。

## 安全边界

Agent 必须遵守仓库根目录 `AGENTS.md` 中的限制：

- 不回滚用户已有改动，不用破坏性 git 命令清理工作区。
- 不把构建产物、临时日志、下载模型或本地环境文件加入提交。
- 不手动修改由脚本生成的文件，除非任务明确要求，并且同时说明生成方式。

涉及性能或体积的改动时，需要特别谨慎。MNN 是推理引擎，代码变更应优先考虑运行时性能、二进制大小、跨平台行为和低端设备资源限制。

## 执行与验证

Agent 在完成修改后，应根据变更类型选择验证方式：

| 修改类型 | 推荐验证 |
|----------|----------|
| 文档修改 | `cd docs && make clean html` |
| 测试脚本或测试矩阵 | `bash -n test.sh`、`python3 -m json.tool test_stages.json`、必要时运行 `./test.sh help` 或目标测试 |
| C++ 代码 | 对应模块编译、相关单元测试、必要时运行 `clang-format -i -style=file` |
| Python 代码 | 相关脚本的最小可运行测试，必要时补充单元测试 |
| backend 性能优化 | 正确性测试、目标 benchmark、改动前后性能记录 |

如果本地缺少依赖、设备或模型，应明确说明未运行的验证项和原因，不要把未验证结果写成已通过。

## 更新文档和 skill

当一次变更改变了开发流程、测试入口、目录结构或常见陷阱时，需要同步考虑：

- 更新 `docs/` 中面向人的说明。
- 更新相关 `skills/` 中面向 Agent 的步骤或注意事项。
- 对非平凡 skill 任务，按 `skills/retrospective/SKILL.md` 做复盘，将新经验沉淀回对应 skill。

文档负责解释“为什么”和“如何使用”，skill 负责约束 Agent “按什么步骤做”。两者保持一致，Agent 才能在后续任务中正确、高效地渐进读取上下文。
