# MNN Project Instructions

MNN is a lightweight deep learning **inference engine** (not a training framework), targeting mobile and server platforms. Supports CNN / Transformer / LLM / Diffusion models. Code must prioritize **performance and binary size**.

## Restricted Access

> The following directories contain internal proprietary code. **Do NOT read, modify, or reference** any files within:
> - `schema/private/`
> - `source/internal/`

## Architecture Overview

MNN uses a **graph optimization + heterogeneous backend scheduling** architecture.

Two inference APIs are available:
- **Session API** (low-level): `Interpreter → createSession → runSession`, operates on Tensor directly
- **Module API** (high-level, recommended): `Module::load → onForward(VARP)`, Express-based dynamic graph. Used by LLM / Diffusion and most modern workloads

**Key abstractions** (see corresponding headers under `source/core/`):
- **Interpreter** / **Session**: model loading and inference session management
- **Backend** / **Execution**: hardware backend abstraction and per-op implementation (CPU/Metal/CUDA/OpenCL/Vulkan/...)
- **Tensor**: data container; internally uses NC4HW4 format (channels packed by 4 for SIMD)
- **Op / Schema**: FlatBuffers-defined operator descriptors (`schema/default/*.fbs`)

**Op registration pattern**: Schema definition → shape inference (`source/shape/`) → Geometry decomposition (optional) → Backend Execution implementation

## LLM Subsystem

MNN supports end-to-end LLM export and inference:

- **Python export** (`transformers/llm/export/`): HuggingFace model → MNN format. Core modules: `llmexport.py` entry point, `utils/model_mapper.py` (model field mapping), `utils/model.py` (unified LlmModel class), `utils/transformers.py` (Attention/Decoder/RoPE export)
- **C++ inference** (`transformers/llm/engine/`): `llm.cpp` (text inference), `omni.cpp` (multimodal: vision/audio), includes KVCache management and sampling strategies

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `include/MNN/` | Public C++ headers |
| `source/core/` | Inference core (Interpreter, Session, Pipeline, Backend) |
| `source/backend/` | Hardware backend implementations (cpu, arm82, metal, cuda, opencl, vulkan, ...) |
| `source/shape/` | Shape inference |
| `source/geometry/` | Geometry computation (op decomposition) |
| `express/` | Express API (high-level dynamic graph, VARP) |
| `schema/default/` | FlatBuffers schema (op definitions) |
| `tools/converter/` | Model converter (ONNX/TF/Caffe → MNN) |
| `transformers/llm/` | LLM export (Python) + inference engine (C++) |
| `transformers/diffusion/` | Diffusion model support |
| `pymnn/` | Python bindings |
| `test/` | Test cases |
| `skills/` | AI Agent Skills |

## Coding Style

- **C++**: Google Style variant, see `.clang-format`. 4-space indent, 120-char line width, attached braces. Class names `PascalCase`, functions `camelCase`, member variables `mCamelCase`. RTTI and exceptions disabled (`-fno-rtti -fno-exceptions`). Default standard: C++11.
- **Python**: Standard Python conventions
- **Formatting**: `clang-format -i -style=file <file>`
- **Comments.** Use comments sparingly and aim for two sentences or fewer. Omit comments when the code is self-explanatory. Explain rationale that the code cannot convey and leave change history to commit messages. Prefer a single overview comment at the top of a struct or class and avoid commenting on each member. Group related functions together.

> ⚠️ **C++ namespace-scope objects must not require dynamic initialization.** This applies to headers as well. Dynamic initialization adds startup overhead and can fail packaging checks. Use `constexpr` or statically initialized POD constants, lazy function-local statics, or construct objects on demand.

## Build & Test

> **Build and run test binaries within the current working directory.** Prefer its `build/` subdirectory for build outputs. Run each binary from its build output directory. Never run test binaries from `/tmp`, `/private/tmp`, or other temporary directories, including their subdirectories.

```bash
# Build C++ (with LLM)
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)

# Common CMake options: MNN_BUILD_TEST, MNN_BUILD_CONVERTER, MNN_METAL, MNN_OPENCL,
# MNN_VULKAN, MNN_CUDA, MNN_ARM82, MNN_BUILD_QUANTOOLS, MNN_SUPPORT_TRANSFORMER_FUSE
# Full list: see option() declarations at the top of CMakeLists.txt

# Unit tests
cd build && ./run_test.out

# LLM export
cd transformers/llm/export
python llmexport.py --path /path/to/model --export mnn --hqq --dst_path ./MODEL

# LLM test
cd build
./llm_demo /path/to/MODEL/config.json prompt.txt

# LLM benchmark
./llm_bench -m /path/to/MODEL/config.json
```

Test suite includes: unit tests (`run_test.out`), model tests, conversion tests (ONNX/TF/TFLite/Torch), quantization tests, LLM tests, PyMNN tests. See `test.sh`, `test_stages.json`, and `test/` directory for details.

## Development Workflow

- **Branch**: Use `feature/<short-description>`, for example `feature/llm-streaming`.
- **Commit**: Use a concise, one-line English message: `[Module:Type] Summary`, for example `[LLM:Feature] Add streaming support`.
  - Modules: `LLM`, `CPU`, `Metal`, `CUDA`, `OpenCL`, `Vulkan`, `Core`, `Infra`, `Doc`, etc.
  - Types: `Feature`, `Bugfix`, `Perf`, `Refact`, `Style`, `Doc`, `Test`, `Chore`, `Release`.

## Skills

For the following tasks, **read the Skill entry file first** and execute step by step. Each step must pass its tests before proceeding.

**After non-trivial skill-driven tasks, run Retrospective only when there are reusable lessons.**

上表可能未覆盖全部 skill。遇到打包/发布/摩天轮/MTL/crash/集成等关键词时，**先 `Glob skills/**/SKILL.md` 确认是否有对应 skill**，不要仅凭表格判断。

Public skills are listed below. Environment-dependent skills may exist under `skills/*/SKILL.md`.

| Skill | Entry File | Trigger |
|-------|-----------|---------|
| Support new LLM | `skills/support-new-llm/SKILL.md` | Add / adapt a new LLM model |
| Add new op | `skills/add-new-op/SKILL.md` | Add a new operator |
| CPU (ARM & x86_64 & RISC-V) | `skills/cpu/SKILL.md` | Any CPU-side work. Router with two branches: `optimize/` — diagnose **why** a CPU op / LLM path is slow and **which layer to change**; `kernel/` — **write / port** a kernel once profiling has located it (SIMD intrinsics or asm, pack & ABI contracts, dispatch registration, correctness gate). Architecture is a sub-layer inside each branch: `optimize/arch/{arm,x86_64,riscv}.md` is the **diagnosis** face (which path am I actually on — on ARM two orthogonal axes: architecture level aarch32 / aarch64 NEON baseline / v8.2 sdot / v8.6 smmla / v9.2 SME2, times precision fp32 / fp16 arm82 / bf16; on x86_64 a single ISA axis plain-C++/SSE/AVX2+FMA/AVX512 No-VNNI/AVX512 VNNI; on RISC-V a single ISA axis plain-C++/standard RVV/SpacemiT IME2 vendor matrix extension — and how to prove it), `kernel/arch/{arm,x86_64,riscv}.md` is the **implementation** face. `shared/` holds the cross-branch tools: build & benchmark discipline, env switch registry, RISC-V dev-board remote validation |
| OpenCL optimization | `skills/opencl-optimize/SKILL.md` | Optimize op performance on OpenCL |
| Vulkan optimization | `skills/vulkan-optimize/SKILL.md` | Optimize op performance on Vulkan |
| Metal optimization | `skills/metal-optimize/SKILL.md` | Optimize op performance on Metal |
| Bugfix / debugging | `skills/general-debug/SKILL.md` | Diagnose correctness bugs / regressions in MNN — router that dispatches by symptom to per-category sub-docs (memory aliasing, export/quant, host-side concurrency & refcount races, fp16 range, GPU OOB, kernel assumptions, stale cache, nondeterminism). |
| Run tests / CI | `skills/test-ci/SKILL.md` | Run the regression / CI suite (host or on-device), benchmark LLM on a real iPhone/iPad, or add / select / retune a test stage |
| Retrospective | `skills/retrospective/SKILL.md` | After non-trivial tasks with reusable lessons |
