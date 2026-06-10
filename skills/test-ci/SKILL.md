---
name: test-ci
description: Run the MNN regression / CI suite for this fork — host-side (local) tests and the on-device Android arm64 matrix — via the declarative ./test_ci.sh driver and test_stages.json. Use when the user asks to run the tests, run CI, smoke-test a build, verify a change on a device, benchmark on-device, or add / select / retune a test stage.
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
| [`test_ci.sh`](../../test_ci.sh) | Bash driver. `local` (host CPU) and `android <serial>` modes. |
| [`test_stages.json`](../../test_stages.json) | Declarative stage matrix. **Edit this** to add / drop / retune stages — no shell edits needed for the common cases. It is self-documenting via its `_documentation` block. |
| [`TESTING.md`](../../TESTING.md) | Deep reference: per-stage explanation + how to add a new operator test end-to-end. |

> Do **not** read or reference `schema/private/` or `source/internal/` (internal
> proprietary code, per the project CLAUDE.md).

## Quick start

```bash
# Host regression (CPU only): build + unit suite + smoke + LLM smoke.
./test_ci.sh local

# Full on-device matrix on the attached arm64 device:
./test_ci.sh android <serial>          # e.g. ./test_ci.sh android R5CY71BJJ9D
```

`<serial>` comes from `adb devices` (the script prefers `adbk` and falls back
to `adb`). If the device shows as `unauthorized`, the user must tap **Allow USB
debugging** on the phone first.

## Running a subset (filters)

Android mode takes an optional filter as the third argument:

```bash
./test_ci.sh android <serial> cpu        # CPU unit + lowmem + llm
./test_ci.sh android <serial> opencl     # OpenCL unit (image+buffer) + opencl smoke
./test_ci.sh android <serial> vulkan     # Vulkan unit + vulkan smoke
./test_ci.sh android <serial> gpu        # opencl + vulkan
./test_ci.sh android <serial> unit       # all unit/op stages only
./test_ci.sh android <serial> lowmem     # only the low-memory matrix
./test_ci.sh android <serial> android-ci # bench + smoke + llm only (no unit/lowmem)
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
  `logs/test_ci-<UTC-timestamp>/<stage>.log` — read the named log of a failing
  stage for the trailing output. `rc=137` ≈ OOM-kill, `rc=139` ≈ SIGSEGV.

## Important pitfall for rebuild-driven smoke tests

When a verification depends on a freshly rebuilt binary (for example `llm_demo`
or `embedding_demo` after resolving a merge conflict), do not trust smoke-test
results gathered while the target is still compiling or before the final link
step has completed. A stale executable can report an old runtime failure and
send debugging in the wrong direction.

Preferred flow:

1. Wait for the target build to finish and confirm the final executable link
   step succeeded.
2. Only then rerun the smoke test and treat that result as authoritative.

## Important pitfall for `llm_demo` prompt-file smoke tests

`transformers/llm/engine/demo/llm_demo.cpp` reads prompt files **one line per
prompt** in the default build. That means a multiline chat template (for
example an ASR prompt laid out across several lines with `<|im_start|>` /
`<|im_end|>`) is silently split into multiple independent prompts and usually
fails or produces empty output.

For multimodal / ASR smoke tests:

* Prefer a **single-line** prompt file for `llm_demo`.
* If a model requires a full chat template, collapse it to one line before
  running the test.
* Do not treat an empty decode or `decode tokens num = 1` from a multiline
  prompt file as a model failure until you have retried with a one-line prompt.

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
LLM_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct-MNN ./test_ci.sh local

# huggingface.co unreachable (e.g. mainland China) → fetch from ModelScope:
LLM_MODEL_SOURCE=modelscope ./test_ci.sh android <serial>
```

For the built-in default model the ModelScope org is remapped automatically
(`taobao-mnn/*` → `MNN/*`); an explicitly-set `LLM_MODEL_REPO` is used verbatim.

## Configuring stages

Editing [`test_stages.json`](../../test_stages.json) is the supported way to
add, drop, or retune unit / lowmem / smoke / bench stages. Every parameter
(forward type, precision, gpuMode, thread count, tag, memory mode,
dynamic-quant option, KleidiAI flag, per-stage skip list, smoke model list,
benchmark args) lives there, and the `_documentation` block at the top of the
file explains every field and every `skip` entry's rationale.

* **Add a stage that runs an existing test in a new config** → add an object to
  `android.stages` (or `local.stages`). See `TESTING.md` § "Add a dedicated stage".
* **Skip a known-broken test on one stage** → add its exact name to that
  stage's `skip` array **and** document why under `_documentation.skip_rationale`.
* **Add a smoke model / bench entry** → see `TESTING.md` § 2d / 2e.

## Adding a new operator test

1. Write the C++ test under `test/op/` (one file, registered with
   `MNNTestSuiteRegister`). The full template + conventions are in
   [`TESTING.md`](../../TESTING.md) § "How to add a new operator test".
2. If its name prefix matches an existing stage (e.g. `op/*`), it is picked up
   automatically — no JSON change needed. Otherwise add a dedicated stage.

For deeper work on operators themselves, see the
[`add-new-op`](../add-new-op/SKILL.md) skill.

## Read next

`TESTING.md` is the authoritative deep reference — read it for the per-stage
breakdown, the stage-object field table, and worked examples.
