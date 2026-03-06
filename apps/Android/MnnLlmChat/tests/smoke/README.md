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
- `scripts/run_extended_priority_regression.sh`: extended run with no-UI-first ordering, then UI stages and report
- `scripts/04_regress_qwen35_download_ops.sh`: Qwen3.5 download/pause/resume/delete regression
- `scripts/noui/05_regress_qwen35_benchmark_noui.sh`: Qwen3.5 benchmark regression (dumpapp only)
- `scripts/05_regress_qwen35_benchmark_ui.sh`: Qwen3.5 benchmark regression (UI only)
- `scripts/05_regress_qwen35_benchmark.sh`: compatibility entry, delegates to UI benchmark script
- `scripts/06_regress_chat_text_image.sh`: chat text + image-entry regression
- `scripts/noui/08_regress_api_dumpapp.sh`: API compatibility + thinking-mode regression via dumpapp + curl (no-code)
- `scripts/09_regress_api_uiautomator.sh`: API settings UiAutomator instrumentation test (code-based)
- `scripts/noui/10_regress_sana_diffusion_dumpapp.sh`: Sana + Diffusion generation regression (dumpapp only)
- `scripts/11_regress_sana_diffusion_uiautomator.sh`: Sana + Diffusion model-entry regression (UI + uiautomator)
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
- `API_UIAUTOMATOR_TEST_CLASS`: optional instrumentation class for step `09` (default: `com.alibaba.mnnllm.android.api.ApiSettingsUiAutomatorTest`)
- `RUN_SANA_DIFFUSION_REGRESSION`: `true` to run step `10/11` Sana + Diffusion regressions in extended pipeline (default: `false`)
- `SANA_MODEL_PATH`: override model path for `10_regress_sana_diffusion_dumpapp.sh`
- `DIFFUSION_MODEL_ID`: override model id for `10_regress_sana_diffusion_dumpapp.sh`
- `THINKING_MAX_TOKENS`: max completion tokens for step `08_regress_api_dumpapp.sh` thinking probe (default: `16`)

## Runtime Process

Typical end-to-end execution order:

1. Environment/device check and app install
2. `standardDebug` smoke
3. `aab release` smoke
4. Qwen3.5 benchmark no-UI regression (dumpapp project, CPU/OpenCL cases)
5. API compatibility regression (dumpapp no-code path)
6. Qwen3.5 benchmark UI regression (CPU/OpenCL cases)
7. Qwen3.5 chat text/image-entry regression
8. Qwen3.5 download regression
9. Optional API settings UiAutomator regression
10. Optional Sana + Diffusion dumpapp regression
11. Optional Sana + Diffusion UiAutomator regression
12. Report generation to `artifacts/report.html`

API compatibility stage details:
   - dumpapp no-code path (`/v1/models`, Anthropic `/v1/messages`, OpenAI `/v1/chat/completions` auth gates, plus HTTPS runtime probe)
   - Anthropic Claude payload compatibility path:
     - `messages[].content` as string
     - `system` as content-block array
     - both validated on local-forward and LAN direct base URLs
   - dumpapp thinking-mode switch path (`dumpapp llm thinking set/get` + OpenAI response-tag differential check)
   - dumpapp path is service-only: no ChatActivity bootstrap fallback is allowed
   - optional UiAutomator code path (API settings switch interaction)

## Key Implementation Logic

- UI actions are coordinate-driven from live UI XML, not fixed hardcoded points.
- Benchmark result success in UI path is detected by result-page signal (for example `基准测试结果`) and screenshot evidence.
- CPU and OpenCL are tracked as independent benchmark projects/cases.
- UI OpenCL case has dedicated timeout control via `OPENCL_UI_TIMEOUT_SEC` (default `60`); timeout triggers app restart to reduce state interference.
- dumpapp API stage ensures LLM runtime session via `dumpapp llm ensure <modelId>` before OpenAI service probes.
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
- UiAutomator thinking-mode regression currently has a disabled phase-2 skeleton test: `ThinkingModeUiAutomatorSkeletonTest`.
