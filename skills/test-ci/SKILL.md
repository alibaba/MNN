---
name: test-ci
description: Run the MNN regression / CI suite for this fork — static checks, host-side (local) tests, and the on-device Android arm64 matrix — via the ./test.sh driver and test_stages.json. Use when the user asks to run the tests, run CI, smoke-test a build, verify a change on a device, benchmark on-device, or add / select / retune a test stage.
---

# MNN Test / CI SKILL

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

## Dynamic-shape device smoke tests

A zero exit code only proves that a backend context ran; it does not prove that
the requested input shape selected the intended dynamic context. For a
shape-sensitive device test, record and validate the runtime-observed input
shape (for example, from a backend dump manifest or the runner's input tensor)
before treating the test as dynamic-shape coverage.

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

For deeper work on operators themselves, see the
[`add-new-op`](../add-new-op/SKILL.md) skill.

## Read next

`docs/testing.md` is the authoritative deep reference — read it for the per-stage
breakdown, the stage-object field table, and worked examples.
