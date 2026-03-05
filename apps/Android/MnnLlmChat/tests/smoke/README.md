# MnnLlmChat Smoke Tests

This folder contains LLM-friendly smoke testing assets for local Android testing.

## Smoke Test Architecture

The smoke framework uses a hybrid architecture with two independent validation channels:

- UI channel: `adb + uiautomator dump + input tap/swipe + screencap`
- Non-UI channel: `tools/dumpapp` via Stetho DumperPlugin

Main goals:

- Validate user-visible interaction paths (UI channel)
- Validate low-level benchmark/download capabilities (dumpapp channel)
- Preserve reproducible evidence (screenshots, logs, summaries, report)

## Structure

- `instructions/`: prompts/instructions for AI agents
- `workflow/`: human-readable smoke workflow and pass/fail criteria
- `scripts/`: executable scripts for install, regression, benchmark, and report generation
- `artifacts/`: generated screenshots, logs, summaries, and HTML report

Key scripts:

- `scripts/run_priority_regression.sh`: baseline regression pipeline
- `scripts/run_extended_priority_regression.sh`: extended run with benchmark/download/chat/report stages
- `scripts/04_regress_qwen35_download_ops.sh`: Qwen3.5 download/pause/resume/delete regression
- `scripts/05_regress_qwen35_benchmark.sh`: Qwen3.5 benchmark regression (UI + dumpapp, CPU/OpenCL split)
- `scripts/06_regress_chat_text_image.sh`: chat text + image-entry regression
- `scripts/08_regress_api_dumpapp.sh`: API compatibility regression via dumpapp + curl (no-code)
- `scripts/09_regress_api_uiautomator.sh`: API settings UiAutomator instrumentation test (code-based)
- `scripts/07_generate_report.sh`: generates single-page `artifacts/report.html`

## Quick Start

```bash
cd apps/Android/MnnLlmChat/tests/smoke
./scripts/run_priority_regression.sh
```

Single build mode:

```bash
BUILD_KIND=standard_debug ./scripts/run_smoke.sh
BUILD_KIND=aab_release ./scripts/run_smoke.sh
```

Extended run (recommended for full evidence):

```bash
./scripts/run_extended_priority_regression.sh
```

Optional env vars:

- `DEVICE_ID`: adb device serial (default: first connected device)
- `PACKAGE_NAME`: default `com.alibaba.mnnllm.android.release`
- `AAB_PATH`: default `apps/Android/MnnLlmChat/release_outputs/googleplay/app-googleplay-release.aab`
- `DEBUG_APK_PATH`: default `apps/Android/MnnLlmChat/app/build/outputs/apk/standard/debug/app-standard-debug.apk`
- `BUNDLETOOL_JAR`: default `/tmp/bundletool-all-1.17.1.jar`
- `UNINSTALL_CONFLICTING`: default `true`, uninstall package before install
- `BUILD_KIND`: `standard_debug` or `aab_release` (default: `aab_release`)
- `RUN_API_UIAUTOMATOR_TEST`: `true` to run step `09_regress_api_uiautomator.sh` in extended pipeline (default: `false`)

## Runtime Process

Typical end-to-end execution order:

1. Environment/device check and app install
2. `standardDebug` smoke
3. `aab release` smoke
4. Qwen3.5 download regression
5. Qwen3.5 chat text/image-entry regression
6. Qwen3.5 benchmark regression:
   - UI project (CPU/OpenCL cases)
   - dumpapp project (CPU/OpenCL cases)
7. API compatibility regression:
   - dumpapp no-code path (`/v1/models`, `/v1/messages` auth gates)
   - optional UiAutomator code path (API settings switch interaction)
8. Report generation to `artifacts/report.html`

## Key Implementation Logic

- UI actions are coordinate-driven from live UI XML, not fixed hardcoded points.
- Benchmark result success in UI path is detected by result-page signal (for example `基准测试结果`) and screenshot evidence.
- CPU and OpenCL are tracked as independent benchmark projects/cases.
- Report is single-page and renders screenshots/logs inline for direct review.
- If OpenCL benchmark project fails, final OpenCL result screenshot is intentionally hidden in report to avoid misleading evidence.

## Evidence Outputs

Primary artifacts:

- Stage summaries: `artifacts/**/summary.txt` and `artifacts/**/smoke_summary.txt`
- UI evidence: `artifacts/**/shots/*.png`
- dumpapp evidence: `artifacts/qwen35_benchmark/case_*.log`
- Aggregated report: `artifacts/report.html`

## Notes

- `bundletool-all` is required for AAB install.
- These scripts are designed for local testing (non-CI) with a connected device.
