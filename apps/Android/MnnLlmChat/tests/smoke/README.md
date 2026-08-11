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
- `scripts/12_regress_streaming_latex.sh`: LaTeX streaming rendering smoke via mock content + screenshot evidence
- `scripts/14_regress_streaming_table.sh`: markdown table rendering smoke via mock streaming content + screenshot evidence
- `scripts/13_regress_storage_ui.sh`: storage management UI smoke via settings navigation + screenshot evidence
- `scripts/noui/08_regress_api_dumpapp.sh`: API compatibility + thinking-mode regression via dumpapp + curl (no-code)
- `scripts/09_regress_api_uiautomator.sh`: API settings UiAutomator instrumentation test (code-based)
- `scripts/15_regress_model_settings_config_ui.sh`: Model settings (home + ChatActivity) + config dump UiAutomator (guards #4259; verifies merged config after save)
- `scripts/noui/10_regress_sana_diffusion_dumpapp.sh`: Sana + Diffusion generation regression (dumpapp only)
- `scripts/11_regress_sana_diffusion_uiautomator.sh`: Sana + Diffusion model-entry regression (UI + uiautomator)
- `scripts/noui/13_regress_storage_dumpapp_smoke.sh`: **dumpapp storage** subcommand smoke (list/analysis/mmap/orphans/verify, integrity checks)
- `scripts/noui/15_regress_voice_dumpapp.sh`: **dumpapp voice** TTS smoke (init/test/destroy via dumpapp)
- `scripts/16_regress_voice_ui.sh`: Voice Chat UI smoke (enter/exit voice chat, status verification)
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
- `UNINSTALL_AT_START`: default `false`; set `true` to uninstall both debug and release packages before STEP1 (wipes model data; use if you hit `INSTALL_FAILED_UPDATE_INCOMPATIBLE`)
- `UNINSTALL_CONFLICTING`: default `true`, uninstall package before install
- `BUILD_KIND`: `standard_debug` or `aab_release` (default: `aab_release`)
- `RUN_API_UIAUTOMATOR_TEST`: `true` to run step `09_regress_api_uiautomator.sh` in extended pipeline (default: `false`)
- `API_UIAUTOMATOR_TEST_CLASS`: optional instrumentation class for step `09` (default: `com.alibaba.mnnllm.android.api.ApiSettingsUiAutomatorTest`)
- `RUN_SANA_DIFFUSION_REGRESSION`: `true` to run step `10/11` Sana + Diffusion regressions in extended pipeline (default: `false`)
- `SANA_MODEL_PATH`: override model path for `10_regress_sana_diffusion_dumpapp.sh`
- `DIFFUSION_MODEL_ID`: override model id for `10_regress_sana_diffusion_dumpapp.sh`
- `THINKING_MAX_TOKENS`: max completion tokens for step `08_regress_api_dumpapp.sh` thinking probe (default: `96`)
- `RUN_STORAGE_DUMPAPP_SMOKE`: set to `true` to run step 13 (dumpapp storage smoke) in extended pipeline (default: `false`)
- `RUN_STORAGE_UI_SMOKE`: set to `true` to run storage management UI smoke in extended pipeline (default: `false`)
- `RUN_LATEX_RENDER_SMOKE`: set to `true` to run LaTeX rendering smoke in extended pipeline (default: `false`)
- `RUN_TABLE_RENDER_SMOKE`: set to `true` to run markdown table rendering smoke in extended pipeline (default: `false`)
- `RUN_VOICE_DUMPAPP_SMOKE`: set to `true` to run Voice TTS dumpapp smoke in extended pipeline (default: `false`)
- `RUN_VOICE_UI_SMOKE`: set to `true` to run Voice Chat UI smoke in extended pipeline (default: `false`)
- `RUN_MODEL_SETTINGS_CONFIG_UI_SMOKE`: run Model settings (home+chat) + config dump regression (guards #4259, system prompt persist); default `true`; set `false` to skip

## Runtime Process

Typical end-to-end execution order:

1. Environment/device check and app install
2. `standardDebug` smoke
3. `aab release` smoke
4. Qwen3.5 benchmark no-UI regression (dumpapp project, CPU/OpenCL cases)
5. API compatibility regression (dumpapp no-code path)
6. Qwen3.5 benchmark UI regression (CPU/OpenCL cases)
7. Qwen3.5 chat text/image-entry regression
8. Optional LaTeX rendering smoke
9. Optional markdown table rendering smoke
10. Qwen3.5 download regression
11. Optional API settings UiAutomator regression
12. Optional Sana + Diffusion dumpapp regression
13. Optional Sana + Diffusion UiAutomator regression
14. Optional storage dumpapp smoke
15. Optional storage management UI smoke
16. Optional Voice TTS dumpapp smoke
17. Optional Voice Chat UI smoke
18. Report generation to `artifacts/report.html`

API compatibility stage details:
   - dumpapp no-code path (`/v1/models`, Anthropic `/v1/messages`, OpenAI `/v1/chat/completions` auth gates, plus HTTPS runtime probe)
   - Anthropic Claude payload compatibility path:
     - `messages[].content` as string
     - `system` as content-block array
     - both validated on local-forward and LAN direct base URLs
   - dumpapp thinking-mode switch path (`dumpapp llm thinking set/get` + OpenAI reasoning-response differential check)
   - dumpapp path is service-only: no ChatActivity bootstrap fallback is allowed
   - optional UiAutomator code path (API settings switch interaction)
   - gesture caveat: step `09_regress_api_uiautomator.sh` does not validate history-drawer left-swipe; for gesture issues use `mobile-mcp` (`mobile_swipe_on_screen`, then `mobile_take_screenshot` + `mobile_list_elements_on_screen`) and keep those artifacts

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

Table rendering smoke specifics:

- Script: `scripts/14_regress_streaming_table.sh`
- Artifact root: `artifacts/table_io/`
- Evidence: staged screenshots plus `generating_*.xml` / `finished.xml`
- Acceptance style: screenshot-first review; use human or vision-capable LLM review on screenshots to judge whether the markdown table rendered correctly

LaTeX rendering smoke specifics:

- Script: `scripts/12_regress_streaming_latex.sh`
- Artifact root: `artifacts/latex_io/`
- Evidence: staged screenshots plus `generating_*.xml` / `finished.xml`
- Acceptance style: screenshot-first review; verify formulas render and streaming does not visibly break layout

Storage management UI smoke specifics:

- Script: `scripts/13_regress_storage_ui.sh`
- Artifact root: `artifacts/storage_ui/`
- Evidence: settings page screenshot, storage summary screenshot, optional expanded row screenshot
- Safety: non-destructive by design; it does not tap delete or clean actions

## Debugging My Models source tags (ModelScope / HuggingFace / Modelers)

If models downloaded from ModelScope (or HuggingFace / Modelers) do not show the expected source tag in the **My Models** tab, use dumpapp to inspect `modelId` and derived **Source**:

1. **List models with source** (same logic as UI):
   ```bash
   dumpapp models list
   ```
   Each model shows `ID:`, `Source:` (ModelScope / HuggingFace / Modelers / Builtin / (local) / (none)), and `Tags:`.

2. **Inspect tags and source for all models**:
   ```bash
   dumpapp models tags --all
   ```

3. **Check storage path → modelId mapping** (ensures `.mnnmodels/modelscope/` → `ModelScope/MNN/...`):
   ```bash
   dumpapp models files
   ```
   Entries under container `modelscope` should have `Model ID: ModelScope/MNN/<name>`. If a ModelScope-downloaded model appears under another container or with a different `Model ID` prefix, the source tag in the UI will be wrong.

4. **Single-model tags**:
   ```bash
   dumpapp models tags "ModelScope/MNN/Qwen2.5-0.5B-MNN"
   ```

Interpretation: if `Source:` is `(none)` for a ModelScope-downloaded model, the `ID` likely does not start with `ModelScope/` (e.g. wrong path when scanning, or cached with a different id). Use `dumpapp models files` to confirm the path and container.

## API dumpapp and ANR

`08_regress_api_dumpapp.sh` must run `dumpapp llm ensure` **before** starting OpenAIService. If the service starts first, `onStartCommand` calls `coordinator.startServer()` which invokes `ensureSession()` → `llmSession.load()` on the main thread, blocking it and causing ANR. With session pre-created, `startServer` uses `getActiveSession()` and avoids heavy work on the main thread.

`dumpapp openai start` now auto-enables `enable_api_service` via the plugin when the pref is false, so the API service can be bootstrapped without UI interaction.

## Notes

- `bundletool-all` is required for AAB install.
- These scripts are designed for local testing (non-CI) with a connected device.
- UiAutomator thinking-mode regression currently has a disabled phase-2 skeleton test: `ThinkingModeUiAutomatorSkeletonTest`.
