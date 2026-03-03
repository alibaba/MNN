# LLM Smoke Test Instructions

Use this instruction when an AI agent needs to run a quick local smoke test for the Google Play build.

## Objective

Prioritize regression on:

1. `standardDebug` build (main test target)
2. Google Play `AAB release` build (secondary verification)

Core scenarios to validate are only high-impact cases for Qwen3.5:

- model download
- chat run/inference
- benchmark run

## Required Actions

1. Install and launch `standardDebug` build, then run the 3 core Qwen3.5 cases.

2. Install and launch `googleplay release AAB`, then quickly sanity-check the same 3 cases.

3. Run smoke scripts:

```bash
cd apps/Android/MnnLlmChat/tests/smoke
./scripts/run_priority_regression.sh
```

4. Verify outputs in `tests/smoke/artifacts/`:
   - `smoke_summary.txt`
   - `window_dump.txt`
   - `ui_dump.xml`
   - `main_screenshot.png`

5. Report pass/fail with concrete evidence:
   - standardDebug result for Qwen3.5 download/run/benchmark
   - googleplay AAB result for Qwen3.5 download/run/benchmark
   - install success/failure
   - package/version from `dumpsys package`
   - launch result from `monkey`
   - foreground activity from `dumpsys window`

## Fail Conditions

- No connected device
- Missing AAB or bundletool
- Install fails
- App cannot launch
- Qwen3.5 download fails
- Qwen3.5 chat run/inference fails
- Qwen3.5 benchmark fails or crashes
- Foreground activity cannot be confirmed

## Suggested Follow-up

- If install fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, uninstall package and retry.
- If UI dump fails, still keep partial logs and mark as partial failure.
