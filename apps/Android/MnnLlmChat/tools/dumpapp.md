# dumpapp Command Guide

`tools/dumpapp` is a thin wrapper around Stetho `dumpapp`. It connects from your computer to the running Android app process and forwards the remaining arguments to the app-side dumper plugins.

This document covers the custom `dumpapp` commands registered by MnnLlmChat in debug builds. Stetho's built-in default commands are not listed here.

## Basic Usage

```bash
./tools/dumpapp <command> [args]
```

Examples:

```bash
./tools/dumpapp models list
./tools/dumpapp llm status
./tools/dumpapp openai start --model ModelScope/MNN/Qwen3___5-0___8B-MNN
```

## Connection Notes

- `-p` or `--process`: override the target Android process name. Default is `com.alibaba.mnnllm.android`.
- `ANDROID_SERIAL`: target a specific device when multiple devices are connected.
- `STETHO_PROCESS`: set the default process name through environment variables.
- `ANDROID_ADB_SERVER_PORT` or `ADB_SERVER_SOCKET`: override adb server connection settings if needed.

## Available Commands

### `models`

Inspect and maintain the app's model list and filesystem entries.

- `dumpapp models list [-v|--verbose]`: list models as they appear to the UI.
- `dumpapp models dump`: dump the raw current model-list state for debugging.
- `dumpapp models refresh`: force a model-list refresh.
- `dumpapp models tags [modelId|--all]`: print model tags and extra tags.
- `dumpapp models find <keyword>`: search models by id or display name.
- `dumpapp models files`: list `.mnnmodels/` entries and symlink targets.
- `dumpapp models unlink <modelId|name>`: remove only the outer symlink entry while preserving underlying data.

### `logs`

Control runtime logging verbosity and file logging.

- `dumpapp logs get`: show the current log level.
- `dumpapp logs set <LEVEL>`: change log level. Supported values are `VERBOSE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `ASSERT`.
- `dumpapp logs file enable`: enable file logging.
- `dumpapp logs file disable`: disable file logging.
- `dumpapp logs file status`: show whether file logging is enabled.

### `market`

Inspect and control the model market network source used by the app.

- `dumpapp market status`: show current market loading state.
- `dumpapp market allow on|off`: allow or block network-based market fetches.
- `dumpapp market env dev|prod`: switch the market endpoint environment.
- `dumpapp market refresh`: force-refresh market data from the network.

### `download`

Operate model downloads without touching the UI.

- `dumpapp download source`: show the current preferred download source.
- `dumpapp download source set <SOURCE>`: switch source. Aliases are `hf`, `ms`, `ml`.
- `dumpapp download test <modelId>`: start a test download for the given model.
- `dumpapp download check-update <modelId>`: check whether the remote model has updates.
- `dumpapp download status <modelId>`: print download state for one model.
- `dumpapp download delete <modelId>`: delete a downloaded model.
- `dumpapp download pause <modelId>`: pause an active download.

### `sana`

Run Sana image-generation models from the command line.

- `dumpapp sana list`: list installed Sana models.
- `dumpapp sana run <model_path> <prompt> [options]`: generate an image with a Sana model.

`sana run` options:

- `--output <path>`: output image path. Default is `/sdcard/sana_output.jpg`.
- `--input <path>`: input image path for img2img mode.
- `--steps <n>`: inference step count.
- `--seed <n>`: random seed.
- `--cfg <true|false>`: enable or disable CFG mode.
- `--cfg-scale <f>`: CFG scale.
- `--width <n>`: output width.
- `--height <n>`: output height.

### `diffusion`

Run non-Sana diffusion image models from the command line.

- `dumpapp diffusion list`: list available diffusion-capable models.
- `dumpapp diffusion run <modelId|modelPath> <prompt> [options]`: generate an image with a diffusion model.

`diffusion run` options:

- `--output <path>`: output image path. Default is `/sdcard/diffusion_output.jpg`.
- `--steps <n>`: inference step count.
- `--seed <n>`: random seed.

### `benchmark`

Run on-device LLM benchmark cases from adb.

- `dumpapp benchmark list`: list benchmarkable models.
- `dumpapp benchmark run <modelId> [options]`: run a benchmark for one model.
- `dumpapp benchmark status`: show the current benchmark state.
- `dumpapp benchmark stop`: stop a running benchmark.

`benchmark run` options:

- `--backend <cpu|opencl>`: select backend.
- `--prompt <n>`: prompt token count.
- `--gen <n>`: generated token count.
- `--repeat <n>`: repeat count.
- `--threads <n>`: thread count.

### `storage`

Inspect and clean app internal storage, especially mmap cache leftovers.

- `dumpapp storage list`: list internal storage directories.
- `dumpapp storage analysis`: print a summarized storage analysis.
- `dumpapp storage analysis detail`: print per-entry drill-down details.
- `dumpapp storage analysis <entryModelId>`: inspect a single storage entry.
- `dumpapp storage mmap`: list mmap cache directories.
- `dumpapp storage orphans`: find orphaned mmap caches.
- `dumpapp storage clean`: clean all orphan mmap caches.
- `dumpapp storage clean <path>`: clean one relative path under `filesDir`.
- `dumpapp storage verify`: output a machine-readable integrity check.

### `config`

Validate model config loading (guards issue #4259).

- `dumpapp config validate <modelId>`: verify config path is a file (not directory) and loadMergedConfig succeeds. Fails if settings would receive wrong path (e.g. model directory instead of config.json).
- `dumpapp config dump <modelId>`: dump merged config (config_path, llm_model, llm_weight, system_prompt). RESULT=FAIL if llm_model/llm_weight empty (corrupted).

### `llm`

Manage the OpenAI-service-side LLM runtime session.

- `dumpapp llm status`: show whether a runtime session exists and which model is active.
- `dumpapp llm ensure <modelId> [--force-reload]`: create or refresh the runtime session for a model.
- `dumpapp llm run <modelId> <prompt> [--force-reload] [--use-app-config]`: ensure the runtime session for a model, then execute one prompt. By default uses `LlmSession.submitFullHistory(...)` with base config only (no custom_config merge). With `--use-app-config`: aligns with ChatActivity—uses `ModelUtils.getConfigPathForModel`, merges custom_config.json, `useNewConfig=false`, and calls `LlmSession.generate(...)` (submitNative); useful for reproducing issues that occur in the app chat but not in dumpapp.
- `dumpapp llm thinking get`: read the current thinking-mode flag from the active runtime session.
- `dumpapp llm thinking set <on|off>`: enable or disable runtime thinking mode.
- `dumpapp llm mock-latex <on|off> [path]`: enable mocked streaming markdown output, optionally loading content from a file. The command name is historical; current smoke flows reuse it for LaTeX and markdown-table rendering.
- `dumpapp llm release`: release the active runtime session.

Note: `llm ensure` remains the lifecycle primitive for loading or switching the active model session. `llm run` is the direct execution entrypoint layered on top of it.

### `openai`

Control the app's OpenAI-compatible local API service.

- `dumpapp openai status`: show service state and current config.
- `dumpapp openai diag`: print service config plus runtime diagnostics.
- `dumpapp openai start [--model <modelId>]`: start `OpenAIService`, optionally with a specific model preselected.
- `dumpapp openai stop`: stop the service.
- `dumpapp openai reset-config`: reset API settings and regenerate a default API key.
- `dumpapp openai test-chat [msg]`: send a test message through the service to validate chat flow.

### `history`

Inspect stored chat-session history.

- `dumpapp history list`: list all sessions and chat-item counts.
- `dumpapp history show <sessionId>`: print all saved chat data for one session.
- `dumpapp history check <sessionId>`: run integrity checks for one session.
- `dumpapp history diag`: run diagnostics across all history data.

### `voice`

Test and validate Voice Chat capabilities (TTS/ASR).

- `dumpapp voice status`: check voice models status (TTS and ASR paths and readiness).
- `dumpapp voice tts init`: initialize the TTS service with the default model.
- `dumpapp voice tts test [text]`: run TTS on the given text and report audio generation metrics.
- `dumpapp voice tts destroy`: release the TTS service.
- `dumpapp voice tts status`: show TTS service internal state.
- `dumpapp voice asr status`: show ASR model path and configuration.

Note: Full ASR testing requires audio input (microphone). For end-to-end ASR validation, use the Voice Chat UI smoke test (`16_regress_voice_ui.sh`).

Examples:

```bash
dumpapp voice status
dumpapp voice tts init
dumpapp voice tts test "Hello, this is a TTS smoke test."
dumpapp voice tts destroy
```

## Source of Truth

The commands above are registered in `app/src/debug/java/com/alibaba/mnnllm/android/StethoInitializer.kt`, and their behavior is implemented in the dumper plugin classes under `app/src/debug/java/com/alibaba/mnnllm/android/debug/` (including `VoiceDumperPlugin.kt`) and `app/src/debug/java/com/alibaba/mnnllm/api/openai/debug/`.
