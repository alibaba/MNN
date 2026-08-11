#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR"
STEP_WATCH_TIMEOUT_SEC="${STEP_WATCH_TIMEOUT_SEC:-1800}"
RUN_API_UIAUTOMATOR_TEST="${RUN_API_UIAUTOMATOR_TEST:-false}"
RUN_SANA_DIFFUSION_REGRESSION="${RUN_SANA_DIFFUSION_REGRESSION:-false}"
RUN_STORAGE_DUMPAPP_SMOKE="${RUN_STORAGE_DUMPAPP_SMOKE:-false}"
RUN_STORAGE_UI_SMOKE="${RUN_STORAGE_UI_SMOKE:-false}"
RUN_LATEX_RENDER_SMOKE="${RUN_LATEX_RENDER_SMOKE:-false}"
RUN_TABLE_RENDER_SMOKE="${RUN_TABLE_RENDER_SMOKE:-false}"
RUN_VOICE_DUMPAPP_SMOKE="${RUN_VOICE_DUMPAPP_SMOKE:-false}"
RUN_VOICE_UI_SMOKE="${RUN_VOICE_UI_SMOKE:-false}"
RUN_MODEL_SETTINGS_CONFIG_UI_SMOKE="${RUN_MODEL_SETTINGS_CONFIG_UI_SMOKE:-true}"

step_status() {
  local k="$1"
  local v="$2"
  echo "${k}=${v}" >>"$ARTIFACT_DIR/extended_priority_steps.txt"
}

run_step_with_watch() {
  local step_key="$1"
  local step_label="$2"
  local cmd="$3"
  local watch_file="${4:-}"
  local timeout_sec="${5:-$STEP_WATCH_TIMEOUT_SEC}"

  echo "$step_label"
  local start_ts now pid
  start_ts="$(date +%s)"

  bash -lc "$cmd" &
  pid=$!

  while kill -0 "$pid" >/dev/null 2>&1; do
    now="$(date +%s)"
    if [ $((now - start_ts)) -ge "$timeout_sec" ]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" || true
      step_status "$step_key" "FAIL_TIMEOUT"
      overall="FAIL"
      return 1
    fi
    if [ -n "$watch_file" ] && [ -f "$watch_file" ]; then
      # heartbeat by touching a stamp to indicate progress file exists
      : >"$ARTIFACT_DIR/.${step_key}_watch_seen"
    fi
    sleep 2
  done

  if wait "$pid"; then
    step_status "$step_key" PASS
    return 0
  fi
  step_status "$step_key" FAIL
  overall="FAIL"
  return 1
}

: >"$ARTIFACT_DIR/extended_priority_steps.txt"
overall="PASS"

run_step_with_watch "STEP1" "[STEP 1] Baseline install/launch smoke (standardDebug + aabRelease)" \
  "$SCRIPT_DIR/run_priority_regression.sh" \
  "$ARTIFACT_DIR/priority_regression_summary.txt"

run_step_with_watch "STEP2" "[STEP 2] Qwen3.5 benchmark no-UI regression (dumpapp first)" \
  "$SCRIPT_DIR/noui/05_regress_qwen35_benchmark_noui.sh" \
  "$ARTIFACT_DIR/qwen35_benchmark/noui_summary.txt"

run_step_with_watch "STEP3" "[STEP 3] API compatibility regression via dumpapp (no-code)" \
  "$SCRIPT_DIR/noui/08_regress_api_dumpapp.sh" \
  "$ARTIFACT_DIR/api_dumpapp/summary.txt"

run_step_with_watch "STEP4" "[STEP 4] Qwen3.5 benchmark UI regression" \
  "$SCRIPT_DIR/05_regress_qwen35_benchmark_ui.sh" \
  "$ARTIFACT_DIR/qwen35_benchmark/ui_summary.txt"

run_step_with_watch "STEP5" "[STEP 5] Chat text/image input entry regression" \
  "$SCRIPT_DIR/06_regress_chat_text_image.sh" \
  "$ARTIFACT_DIR/chat_io/summary.txt"

if [ "$RUN_LATEX_RENDER_SMOKE" = "true" ]; then
  run_step_with_watch "STEP_LATEX" "[STEP] LaTeX rendering smoke (UI screenshots)" \
    "$SCRIPT_DIR/12_regress_streaming_latex.sh" \
    "$ARTIFACT_DIR/latex_io/summary.txt"
fi

if [ "$RUN_TABLE_RENDER_SMOKE" = "true" ]; then
  run_step_with_watch "STEP_TABLE" "[STEP] Markdown table rendering smoke (UI screenshots)" \
    "$SCRIPT_DIR/14_regress_streaming_table.sh" \
    "$ARTIFACT_DIR/table_io/summary.txt"
fi

run_step_with_watch "STEP6" "[STEP 6] Qwen3.5 download pause/resume/delete regression" \
  "$SCRIPT_DIR/04_regress_qwen35_download_ops.sh" \
  "$ARTIFACT_DIR/qwen35_download/summary.txt"

if [ "$RUN_API_UIAUTOMATOR_TEST" = "true" ]; then
  run_step_with_watch "STEP7" "[STEP 7] API settings UiAutomator regression (code-based)" \
    "$SCRIPT_DIR/09_regress_api_uiautomator.sh" \
    "$ARTIFACT_DIR/api_uiautomator/summary.txt"
fi

if [ "$RUN_MODEL_SETTINGS_CONFIG_UI_SMOKE" = "true" ]; then
  run_step_with_watch "STEP_MODEL_SETTINGS_CONFIG" "[STEP] Model settings (home+chat) + config dump UiAutomator (guards #4259)" \
    "$SCRIPT_DIR/15_regress_model_settings_config_ui.sh" \
    "$ARTIFACT_DIR/model_settings_config_ui/summary.txt"
fi

if [ "$RUN_SANA_DIFFUSION_REGRESSION" = "true" ]; then
  run_step_with_watch "STEP8" "[STEP 8] Sana+Diffusion no-UI dumpapp regression" \
    "$SCRIPT_DIR/noui/10_regress_sana_diffusion_dumpapp.sh" \
    "$ARTIFACT_DIR/sana_diffusion_dumpapp/summary.txt"

  run_step_with_watch "STEP9" "[STEP 9] Sana+Diffusion UiAutomator regression" \
    "$SCRIPT_DIR/11_regress_sana_diffusion_uiautomator.sh" \
    "$ARTIFACT_DIR/sana_diffusion_ui/summary.txt"
fi

if [ "$RUN_STORAGE_DUMPAPP_SMOKE" = "true" ]; then
  run_step_with_watch "STEP_STORAGE" "[STEP] Storage dumpapp smoke (list/analysis/mmap/orphans/verify)" \
    "$SCRIPT_DIR/noui/13_regress_storage_dumpapp_smoke.sh" \
    "$ARTIFACT_DIR/storage_dumpapp_smoke/summary.txt"
fi

if [ "$RUN_STORAGE_UI_SMOKE" = "true" ]; then
  run_step_with_watch "STEP_STORAGE_UI" "[STEP] Storage management UI smoke (settings navigation + summary capture)" \
    "$SCRIPT_DIR/13_regress_storage_ui.sh" \
    "$ARTIFACT_DIR/storage_ui/summary.txt"
fi

if [ "$RUN_VOICE_DUMPAPP_SMOKE" = "true" ]; then
  run_step_with_watch "STEP_VOICE_DUMPAPP" "[STEP] Voice dumpapp smoke (TTS init/test)" \
    "$SCRIPT_DIR/noui/15_regress_voice_dumpapp.sh" \
    "$ARTIFACT_DIR/voice_dumpapp/summary.txt"
fi

if [ "$RUN_VOICE_UI_SMOKE" = "true" ]; then
  run_step_with_watch "STEP_VOICE_UI" "[STEP] Voice Chat UI smoke (enter/exit voice chat)" \
    "$SCRIPT_DIR/16_regress_voice_ui.sh" \
    "$ARTIFACT_DIR/voice_ui/summary.txt"
fi

run_step_with_watch "STEP10" "[STEP 10] Generate single-page HTML report" \
  "$SCRIPT_DIR/07_generate_report.sh" \
  "$ARTIFACT_DIR/report.html" \
  300

{
  echo "EXTENDED_PRIORITY_REGRESSION=${overall}"
  echo "COVERAGE=benchmark_noui,api_dumpapp,benchmark_ui,text_input,image_input_entry,latex_render_optional,table_render_optional,download_pause_resume_delete,model_settings_config_ui,api_uiautomator_optional,sana_diffusion_dumpapp_optional,sana_diffusion_uiautomator_optional,storage_dumpapp_smoke_optional,storage_ui_optional,voice_dumpapp_optional,voice_ui_optional"
  echo "ARTIFACT_ROOT=$ARTIFACT_DIR"
  echo "REPORT_HTML=$ARTIFACT_DIR/report.html"
  echo "RUN_API_UIAUTOMATOR_TEST=$RUN_API_UIAUTOMATOR_TEST"
  echo "RUN_SANA_DIFFUSION_REGRESSION=$RUN_SANA_DIFFUSION_REGRESSION"
  echo "RUN_STORAGE_DUMPAPP_SMOKE=$RUN_STORAGE_DUMPAPP_SMOKE"
  echo "RUN_STORAGE_UI_SMOKE=$RUN_STORAGE_UI_SMOKE"
  echo "RUN_LATEX_RENDER_SMOKE=$RUN_LATEX_RENDER_SMOKE"
  echo "RUN_TABLE_RENDER_SMOKE=$RUN_TABLE_RENDER_SMOKE"
  echo "RUN_VOICE_DUMPAPP_SMOKE=$RUN_VOICE_DUMPAPP_SMOKE"
  echo "RUN_VOICE_UI_SMOKE=$RUN_VOICE_UI_SMOKE"
  echo "RUN_MODEL_SETTINGS_CONFIG_UI_SMOKE=$RUN_MODEL_SETTINGS_CONFIG_UI_SMOKE"
} >"$ARTIFACT_DIR/extended_priority_summary.txt"

cat "$ARTIFACT_DIR/extended_priority_summary.txt"
cat "$ARTIFACT_DIR/extended_priority_steps.txt"

if [ "$overall" != "PASS" ]; then
  exit 1
fi
