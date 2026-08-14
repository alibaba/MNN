# MNN Regression / CI Suite (host + Android)

> **Trigger**: when the user wants to run the test suite or CI ("run the tests",
> "run CI", "smoke test", "does this still pass", "verify on the phone/device",
> "benchmark on device"), or to add / select / retune a test stage.

The operational scripts live at the **repository root** and are invoked from
there — this skill is the discovery + usage entry point for AI agents and
humans:

| File | Role |
|------|------|
| [`test.sh`](../../test.sh) | Bash driver. `static`, `local` (host CPU), and `android <serial>` modes. |
| [`test_stages.json`](../../test_stages.json) | Declarative stage matrix. **Edit this** to add / drop / retune stages — no shell edits needed for the common cases. It is self-documenting via its `_documentation` block. |
| [`docs/testing.md`](../../docs/testing.md) | 中文测试文档：阶段说明、字段表、新增算子测试流程。 |

## Quick start

```bash
# Static checks only:
./test.sh static

# Host regression (CPU only): build + unit suite + smoke + LLM smoke.
./test.sh local

# Full on-device matrix on the attached arm64 device:
./test.sh android <serial>          # e.g. ./test.sh android R5CY71BJJ9D
```

`<serial>` comes from `adb devices` (the script prefers `adbk` and falls back
to `adb`). If the device shows as `unauthorized`, the user must tap **Allow USB
debugging** on the phone first.

## Running a subset (filters)

Android mode takes an optional filter as the third argument:

```bash
./test.sh android <serial> cpu        # CPU unit + lowmem + llm
./test.sh android <serial> opencl     # OpenCL unit (image+buffer) + opencl smoke
./test.sh android <serial> vulkan     # Vulkan unit + vulkan smoke
./test.sh android <serial> gpu        # opencl + vulkan
./test.sh android <serial> unit       # all unit/op stages only
./test.sh android <serial> lowmem     # only the low-memory matrix
./test.sh android <serial> android-ci # bench + smoke + llm only (no unit/lowmem)
```

Valid filters: `all` (default) · `cpu` · `opencl` · `opencl-image` ·
`opencl-buffer` · `vulkan` · `gpu` · `unit` · `lowmem` · `android-ci`.

## Reading the result (agent-friendly)

* Each stage prints a delimited `═══ stage: <name> ═══` block, then a
  `PASS` / `FAIL` / `SKIP` line.
* A final **summary** prints `total / passed / failed / skipped` and one line
  per stage. `SKIP` is not a failure — it means the prerequisite was absent
  (e.g. a GPU library, a model, or a missing build artefact).
* **Exit code is non-zero iff any stage failed.** Gate automation on the exit
  code, not on log scraping.
* Combined stdout/stderr for every stage is saved under
  `logs/test-<UTC-timestamp>/<stage>.log` — read the named log of a failing
  stage for the trailing output. `rc=137` ≈ OOM-kill, `rc=139` ≈ SIGSEGV.
* When a remote transport has a shorter timeout than the device workload, a
  blank or truncated client response does not prove that the process exited.
  Write results on the device, then poll the process and result-file size
  before deciding whether the test completed.

## Environment variables

| Var | Mode | Meaning |
|-----|------|---------|
| `ANDROID_NDK` | android | NDK root. Falls back to `$HOME/android-ndk-r21`. |
| `ANDROID_EXTRA_CMAKE` | android | Extra cmake flags appended to the build (e.g. `-DMNN_SME2=OFF`) — handy for bisecting a backend regression. |
| `LLM_MODEL_DIR` | both | Path to an existing on-disk MNN-format LLM model. When set, that directory is used **as-is and nothing is downloaded**. Defaults to `models/<repo-basename>/`. |
| `LLM_MODEL_REPO` | both | Model repo id for the LLM smoke test. Default `taobao-mnn/Qwen2.5-0.5B-Instruct-MNN`. |
| `LLM_MODEL_SOURCE` | both | Download source when `LLM_MODEL_DIR` is unset: `huggingface` (default) or `modelscope`. |
| `LLM_MODEL_URL_BASE` | both | Override the resolve URL prefix outright (wins over `LLM_MODEL_SOURCE`). |
| `MNN_TEST_SKIP` | both | Comma list of exact test names to skip (also set per-stage via the JSON `skip` field). |

### Offline / no-network and mainland-China notes

LLM model provisioning is **lazy**: the download (or `LLM_MODEL_DIR` check) is
deferred until the `llm` stage actually runs, and a provisioning failure skips
**only** that stage. So the unit / smoke / bench stages run fine with no
network.

```bash
# Already have the model on disk → no download attempt at all:
LLM_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct-MNN ./test.sh local

# huggingface.co unreachable (e.g. mainland China) → fetch from ModelScope:
LLM_MODEL_SOURCE=modelscope ./test.sh android <serial>
```

For the built-in default model the ModelScope org is remapped automatically
(`taobao-mnn/*` → `MNN/*`); an explicitly-set `LLM_MODEL_REPO` is used verbatim.

### LLM backend/layout smoke

For backend or tensor-layout optimizations, do not stop at operator tests. Run
an end-to-end `llm_demo` correctness smoke with a short prompt and another
prompt long enough to cross backend prefill branch thresholds. This catches
real exported-graph layout bugs where an op test covers only the output format,
but the graph also changes an input tensor format.

### Split LoRA multi-instance smoke

When testing multiple split LoRA models against one quantized base, use
adapter-specific exact markers and verify both concurrency and switching:

1. Keep the base `Llm` alive until every object returned by `create_lora()` is
   destroyed; adapter modules reference the base module.
2. Load all adapters before inference. Run one request per adapter concurrently,
   then alternate adapters for multiple rounds with `reset()` before each
   independent request.
3. Require each output to contain its own marker and not another adapter's
   marker. Loading without errors is not a sufficient pass condition.
4. Match LoRA-training fake-quant settings to export settings and prefer LoRA
   filenames relative to the base `config.json`.
5. A Python thread-pool test is concurrent only if the native inference binding
   releases the GIL. Align worker entry with a barrier so serialized scheduling
   cannot accidentally satisfy the test.
The runnable reference is
`transformers/llm/finetune/examples/multi_lora/README.md`.

## Configuring stages

Editing [`test_stages.json`](../../test_stages.json) is the supported way to
add, drop, or retune unit / lowmem / smoke / bench stages. Every parameter
(forward type, precision, gpuMode, thread count, tag, memory mode,
dynamic-quant option, KleidiAI flag, per-stage skip list, smoke model list,
benchmark args) lives there, and the `_documentation` block at the top of the
file explains every field and every `skip` entry's rationale.

* **Add a stage that runs an existing test in a new config** → add an object to
  `android.stages` (or `local.stages`). See `docs/testing.md` § "增加专门阶段".
* **Skip a known-broken test on one stage** → add its exact name to that
  stage's `skip` array **and** document why under `_documentation.skip_rationale`.
* **Add a smoke model / bench entry** → see `docs/testing.md` § "新增 smoke 模型或 bench 阶段".

## Auditing stale CI/test scripts

When asked to clean up old CI or test scripts, build a usage map before
recommending deletion:

* Prefer `git ls-files` plus targeted `rg`/`git grep` over broad filesystem
  scans, so generated build directories and local experiments do not look like
  maintained CI surface.
* Classify scripts by role: active CI entrypoints, declarative test driver,
  release/package scripts, manual benchmark helpers, third-party vendored
  tests, and local device/debug helpers.
* Treat lack of in-repo references as a "review/deprecate" signal, not proof
  of dead code; internal CI systems can invoke tracked files by convention.
  Prefer a staged deprecation plan unless a script is both unreferenced and
  clearly superseded by `test.sh` / `test_stages.json`.
* When renaming or consolidating test entrypoints, grep for both executable
  names and generated-artifact prefixes. Update CI config, `.gitignore`,
  `test_stages.json` self-documentation, developer docs, skill docs, and code
  comments in the same change so the old entrypoint disappears completely.

## Adding a new operator test

1. Write the C++ test under `test/op/` (one file, registered with
   `MNNTestSuiteRegister`). The full template + conventions are in
   [`docs/testing.md`](../../docs/testing.md) § "新增算子测试".
2. If its name prefix matches an existing stage (e.g. `op/*`), it is picked up
   automatically — no JSON change needed. Otherwise add a dedicated stage.

### Attention causal-mask assumption (⚠️ non-causal models on Metal)

Metal backend prefill has a **silent-error mode**: both the three-kernel path
(with CAUSAL_TRI / CAUSAL_BOUND, `f28510967` / `78ae7bc55`) and the flash-attn
path (`MetalFlashAttnShader.hpp`) hard-code the assumption **"attention mask
is causal lower-triangular"**. Non-causal architectures (Sliding Window
Attention: Mistral 7B v0.1 / Gemma-2 / Ministral; Prefix LM: Baichuan-Base;
encoder / bidirectional: BERT-family, T5, UL2) will **produce garbled tokens
with no crash and no warning** when routed through Metal.

The default LLM smoke stage uses Qwen2.5-0.5B (causal), so it will not catch
this regression. When adding a **non-causal model** to `test_stages.json`
smoke list, or when introducing a new Attention / softmax / prefill_qk /
prefill_qkv shader change, you must:

1. Include a diff-based A/B in the smoke: run once with `MNN_METAL_QK_CAUSAL_TRI=0`
   and once with default; the first 20 greedy tokens must be identical
   (indicates the model actually is causal-safe under CAUSAL_TRI/BOUND).
   If they diverge, the model is not causal and the smoke must pin
   `MNN_METAL_QK_CAUSAL_TRI=0`.
2. For any model configured with `attention_mode >= 8` (FA enabled), also
   verify with `MNN_ENABLE_FLASH_ATTN_PREFILL=0` — FA's causal hard-code
   has no opt-out short of disabling FA entirely.
3. Add the safe env pin to the model's stage entry (via a wrapper script or a
   TODO in `test_stages.json`), and note in `_documentation.skip_rationale`
   or a new note field why it is required.

Full risk breakdown, gate conditions, and remediation options: see
[`skills/general-debug/SKILL.md`](../general-debug/SKILL.md) §7 (后端 kernel 隐式假设违反)
and [`skills/metal-optimize/build-and-test.md`](../metal-optimize/build-and-test.md)
§ "Attention causal 假设".

For deeper work on operators themselves, see the
[`add-new-op`](../add-new-op/SKILL.md) skill.

## Read next

`docs/testing.md` is the authoritative deep reference — read it for the per-stage
breakdown, the stage-object field table, and worked examples.
