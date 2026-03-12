#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
REPORT_PATH="$ARTIFACT_DIR/report.html"

std_summary="$ARTIFACT_DIR/standard_debug/smoke_summary.txt"
aab_summary="$ARTIFACT_DIR/aab_release/smoke_summary.txt"
dl_summary="$ARTIFACT_DIR/qwen35_download/summary.txt"
chat_summary="$ARTIFACT_DIR/chat_io/summary.txt"
table_summary="$ARTIFACT_DIR/table_io/summary.txt"
bench_summary="$ARTIFACT_DIR/qwen35_benchmark/summary.txt"
bench_noui_summary="$ARTIFACT_DIR/qwen35_benchmark/noui_summary.txt"
bench_ui_summary="$ARTIFACT_DIR/qwen35_benchmark/ui_summary.txt"
api_dump_summary="$ARTIFACT_DIR/api_dumpapp/summary.txt"
api_ui_summary="$ARTIFACT_DIR/api_uiautomator/summary.txt"
sana_diff_dump_summary="$ARTIFACT_DIR/sana_diffusion_dumpapp/summary.txt"
sana_diff_ui_summary="$ARTIFACT_DIR/sana_diffusion_ui/summary.txt"
bench_cpu64_log="$ARTIFACT_DIR/qwen35_benchmark/case_cpu_64_64.log"
bench_cpu128_log="$ARTIFACT_DIR/qwen35_benchmark/case_cpu_128_128.log"
bench_opencl_log="$ARTIFACT_DIR/qwen35_benchmark/case_opencl_64_64.log"

status_from_file() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "MISSING"
    return
  fi
  if rg -q "=PASS|SMOKE_TEST=PASS" "$f"; then
    echo "PASS"
  else
    echo "FAIL"
  fi
}

summary_value() {
  local key="$1"
  local fallback="${2:-MISSING}"
  if [ ! -f "$bench_summary" ]; then
    echo "$fallback"
    return
  fi
  local v
  v="$(rg -m1 "^${key}=" "$bench_summary" | cut -d= -f2- || true)"
  if [ -z "$v" ]; then
    echo "$fallback"
  else
    echo "$v"
  fi
}

summary_value_from() {
  local file="$1"
  local key="$2"
  local fallback="${3:-MISSING}"
  if [ ! -f "$file" ]; then
    echo "$fallback"
    return
  fi
  local v
  v="$(rg -m1 "^${key}=" "$file" | cut -d= -f2- || true)"
  if [ -z "$v" ]; then
    echo "$fallback"
  else
    echo "$v"
  fi
}

extract_metric() {
  local f="$1"
  local k="$2"
  if [ ! -f "$f" ]; then
    echo "-"
    return
  fi
  local v
  v="$(rg -m1 "^${k}:" "$f" 2>/dev/null | sed -E 's/^.*: *//' || true)"
  if [ -z "$v" ]; then
    echo "-"
  else
    echo "$v"
  fi
}

escape_html() {
  sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

inline_log_tail() {
  local f="$1"
  local n="${2:-80}"
  if [ -f "$f" ]; then
    tail -n "$n" "$f" | escape_html
  else
    echo "[missing] $f"
  fi
}

synthesize_benchmark_summary_if_needed() {
  if [ -f "$bench_summary" ]; then
    return
  fi
  if [ ! -f "$bench_noui_summary" ] || [ ! -f "$bench_ui_summary" ]; then
    return
  fi

  local ui_project dump_project ui_cpu ui_opencl dump_cpu64 dump_cpu128 dump_opencl active_model
  ui_project="$(rg -m1 '^UI_PROJECT=' "$bench_ui_summary" | cut -d= -f2- || echo FAIL)"
  dump_project="$(rg -m1 '^DUMPAPP_PROJECT=' "$bench_noui_summary" | cut -d= -f2- || echo FAIL)"
  ui_cpu="$(rg -m1 '^UI_CPU_CASE=' "$bench_ui_summary" | cut -d= -f2- || echo FAIL)"
  ui_opencl="$(rg -m1 '^UI_OPENCL_CASE=' "$bench_ui_summary" | cut -d= -f2- || echo FAIL)"
  dump_cpu64="$(rg -m1 '^DUMP_CPU64_CASE=' "$bench_noui_summary" | cut -d= -f2- || echo FAIL)"
  dump_cpu128="$(rg -m1 '^DUMP_CPU128_CASE=' "$bench_noui_summary" | cut -d= -f2- || echo FAIL)"
  dump_opencl="$(rg -m1 '^DUMP_OPENCL_CASE=' "$bench_noui_summary" | cut -d= -f2- || echo FAIL)"
  active_model="$(rg -m1 '^ACTIVE_MODEL_ID=' "$bench_noui_summary" | cut -d= -f2- || echo unknown)"

  local cpu_project opencl_project overall
  cpu_project="FAIL"
  opencl_project="FAIL"
  overall="FAIL"

  if [ "$ui_cpu" = "PASS" ] && [ "$dump_cpu64" = "PASS" ] && [ "$dump_cpu128" = "PASS" ]; then
    cpu_project="PASS"
  fi
  if [ "$ui_opencl" = "PASS" ] && [ "$dump_opencl" = "PASS" ]; then
    opencl_project="PASS"
  fi
  if [ "$ui_project" = "PASS" ] && [ "$dump_project" = "PASS" ]; then
    overall="PASS"
  fi

  {
    echo "QWEN35_BENCHMARK=$overall"
    echo "UI_PROJECT=$ui_project"
    echo "DUMPAPP_PROJECT=$dump_project"
    echo "CPU_PROJECT=$cpu_project"
    echo "OPENCL_PROJECT=$opencl_project"
    echo "UI_CPU_CASE=$ui_cpu"
    echo "UI_OPENCL_CASE=$ui_opencl"
    echo "DUMP_CPU64_CASE=$dump_cpu64"
    echo "DUMP_CPU128_CASE=$dump_cpu128"
    echo "DUMP_OPENCL_CASE=$dump_opencl"
    echo "SCREENSHOTS=$ARTIFACT_DIR/qwen35_benchmark/shots"
    echo "UI_ACTION_LOG=$ARTIFACT_DIR/qwen35_benchmark/ui_actions.log"
    echo "ACTIVE_MODEL_ID=$active_model"
  } >"$bench_summary"
}

synthesize_benchmark_summary_if_needed

std_status="$(status_from_file "$std_summary")"
aab_status="$(status_from_file "$aab_summary")"
dl_status="$(status_from_file "$dl_summary")"
chat_status="$(status_from_file "$chat_summary")"
table_status="$(status_from_file "$table_summary")"

bench_status="$(summary_value QWEN35_BENCHMARK FAIL)"
api_dump_status="$(status_from_file "$api_dump_summary")"
api_ui_status="$(status_from_file "$api_ui_summary")"
sana_diff_dump_status="$(status_from_file "$sana_diff_dump_summary")"
sana_diff_ui_status="$(status_from_file "$sana_diff_ui_summary")"
ui_project="$(summary_value UI_PROJECT MISSING)"
dumpapp_project="$(summary_value DUMPAPP_PROJECT MISSING)"
cpu_project="$(summary_value CPU_PROJECT MISSING)"
opencl_project="$(summary_value OPENCL_PROJECT MISSING)"
thinking_config_switch="$(summary_value_from "$api_dump_summary" THINKING_CONFIG_SWITCH MISSING)"
thinking_response_switch="$(summary_value_from "$api_dump_summary" THINKING_RESPONSE_SWITCH MISSING)"
thinking_mode_regression="$(summary_value_from "$api_dump_summary" THINKING_MODE_REGRESSION MISSING)"
duplicate_start_regression="$(summary_value_from "$api_dump_summary" API_DUPLICATE_START_REGRESSION MISSING)"

cpu64_prefill="$(extract_metric "$bench_cpu64_log" "Prefill")"
cpu64_decode="$(extract_metric "$bench_cpu64_log" "Decode")"
cpu128_prefill="$(extract_metric "$bench_cpu128_log" "Prefill")"
cpu128_decode="$(extract_metric "$bench_cpu128_log" "Decode")"
ocl64_prefill="$(extract_metric "$bench_opencl_log" "Prefill")"
ocl64_decode="$(extract_metric "$bench_opencl_log" "Decode")"

opencl_shot_html='<div class="shot"><div>OpenCL result screenshot is hidden because OpenCL project failed.</div></div>'
if [ "$opencl_project" = "PASS" ] && [ -f "$ARTIFACT_DIR/qwen35_benchmark/shots/06_after_opencl_64_64_result.png" ]; then
  opencl_shot_html='<div class="shot"><div>qwen35_benchmark/shots/06_after_opencl_64_64_result.png</div><img src="qwen35_benchmark/shots/06_after_opencl_64_64_result.png" alt="bench_opencl_result"></div>'
fi

table_shot_html=''
if [ -f "$ARTIFACT_DIR/table_io/shots/06_finished.png" ]; then
  table_shot_html='<div class="shot"><div>table_io/shots/06_finished.png</div><img src="table_io/shots/06_finished.png" alt="table_render_result"></div>'
fi

now="$(date '+%Y-%m-%d %H:%M:%S %z')"

cat >"$REPORT_PATH" <<EOF_HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MnnLlmChat Smoke Report</title>
  <style>
    :root { --ok:#0a7a2f; --bad:#b42318; --bg:#f6f8fb; --card:#fff; --txt:#111827; --muted:#6b7280; }
    body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif; margin:0; background:var(--bg); color:var(--txt); }
    .wrap { max-width: 1080px; margin: 20px auto; padding: 0 16px; }
    .card { background:var(--card); border:1px solid #e5e7eb; border-radius:12px; padding:16px; margin-bottom:14px; }
    h1,h2 { margin:0 0 10px; }
    table { width:100%; border-collapse:collapse; }
    th,td { border-bottom:1px solid #eef2f7; text-align:left; padding:8px 6px; font-size:14px; }
    .ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; } .muted { color:var(--muted); }
    .shots { display:grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap:10px; }
    .shot { border:1px solid #e5e7eb; border-radius:10px; padding:8px; background:#fff; }
    .shot img { width:100%; height:auto; border-radius:8px; border:1px solid #eef2f7; }
    pre { white-space: pre-wrap; word-break: break-word; font-size:12px; background:#0b1020; color:#d1e4ff; padding:10px; border-radius:8px; max-height:280px; overflow:auto; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>MnnLlmChat Smoke Report</h1>
      <div class="muted">Generated at: ${now}</div>
      <div class="muted">Artifact root: ${ARTIFACT_DIR}</div>
    </div>

    <div class="card">
      <h2>Stage Status</h2>
      <table>
        <tr><th>Stage</th><th>Status</th><th>Summary</th></tr>
        <tr><td>standardDebug smoke</td><td class="$( [ "$std_status" = PASS ] && echo ok || echo bad )">${std_status}</td><td>standard_debug/smoke_summary.txt</td></tr>
        <tr><td>aab release smoke</td><td class="$( [ "$aab_status" = PASS ] && echo ok || echo bad )">${aab_status}</td><td>aab_release/smoke_summary.txt</td></tr>
        <tr><td>Qwen3.5 download ops</td><td class="$( [ "$dl_status" = PASS ] && echo ok || echo bad )">${dl_status}</td><td>qwen35_download/summary.txt</td></tr>
        <tr><td>Qwen3.5 chat text/image entry</td><td class="$( [ "$chat_status" = PASS ] && echo ok || echo bad )">${chat_status}</td><td>chat_io/summary.txt</td></tr>
        <tr><td>Table rendering smoke</td><td class="$( [ "$table_status" = PASS ] && echo ok || echo bad )">${table_status}</td><td>table_io/summary.txt</td></tr>
        <tr><td>Qwen3.5 benchmark overall</td><td class="$( [ "$bench_status" = PASS ] && echo ok || echo bad )">${bench_status}</td><td>qwen35_benchmark/summary.txt</td></tr>
        <tr><td>API dumpapp compatibility</td><td class="$( [ "$api_dump_status" = PASS ] && echo ok || echo bad )">${api_dump_status}</td><td>api_dumpapp/summary.txt</td></tr>
        <tr><td>API dumpapp thinking regression</td><td class="$( [ "$thinking_mode_regression" = PASS ] && echo ok || echo bad )">${thinking_mode_regression}</td><td>api_dumpapp/summary.txt</td></tr>
        <tr><td>API duplicate-start regression</td><td class="$( [ "$duplicate_start_regression" = PASS ] && echo ok || echo bad )">${duplicate_start_regression}</td><td>api_dumpapp/summary.txt</td></tr>
        <tr><td>API UiAutomator compatibility</td><td class="$( [ "$api_ui_status" = PASS ] && echo ok || echo bad )">${api_ui_status}</td><td>api_uiautomator/summary.txt</td></tr>
        <tr><td>Sana+Diffusion dumpapp regression</td><td class="$( [ "$sana_diff_dump_status" = PASS ] && echo ok || echo bad )">${sana_diff_dump_status}</td><td>sana_diffusion_dumpapp/summary.txt</td></tr>
        <tr><td>Sana+Diffusion UiAutomator regression</td><td class="$( [ "$sana_diff_ui_status" = PASS ] && echo ok || echo bad )">${sana_diff_ui_status}</td><td>sana_diffusion_ui/summary.txt</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Thinking Mode</h2>
      <table>
        <tr><th>Check</th><th>Status</th></tr>
        <tr><td>Config switch (on/off)</td><td class="$( [ "$thinking_config_switch" = PASS ] && echo ok || echo bad )">${thinking_config_switch}</td></tr>
        <tr><td>Response tag differential</td><td class="$( [ "$thinking_response_switch" = PASS ] && echo ok || echo bad )">${thinking_response_switch}</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Benchmark Projects</h2>
      <table>
        <tr><th>Project</th><th>Status</th></tr>
        <tr><td>UI Project</td><td class="$( [ "$ui_project" = PASS ] && echo ok || echo bad )">${ui_project}</td></tr>
        <tr><td>Dumpapp Project</td><td class="$( [ "$dumpapp_project" = PASS ] && echo ok || echo bad )">${dumpapp_project}</td></tr>
        <tr><td>CPU Project</td><td class="$( [ "$cpu_project" = PASS ] && echo ok || echo bad )">${cpu_project}</td></tr>
        <tr><td>OpenCL Project</td><td class="$( [ "$opencl_project" = PASS ] && echo ok || echo bad )">${opencl_project}</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Benchmark Metrics</h2>
      <table>
        <tr><th>Case</th><th>Prefill</th><th>Decode</th></tr>
        <tr><td>CPU pp64+tg64</td><td>${cpu64_prefill}</td><td>${cpu64_decode}</td></tr>
        <tr><td>CPU pp128+tg128</td><td>${cpu128_prefill}</td><td>${cpu128_decode}</td></tr>
        <tr><td>OpenCL pp64+tg64</td><td>${ocl64_prefill}</td><td>${ocl64_decode}</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Screenshots (Inline)</h2>
      <div class="shots">
        <div class="shot"><div>qwen35_download/shots/03_after_pause.png</div><img src="qwen35_download/shots/03_after_pause.png" alt="download_pause"></div>
        <div class="shot"><div>chat_io/shots/07_image_test_entry.png</div><img src="chat_io/shots/07_image_test_entry.png" alt="chat_image_entry"></div>
        ${table_shot_html}
        <div class="shot"><div>qwen35_benchmark/shots/05_after_cpu_128_128_result.png</div><img src="qwen35_benchmark/shots/05_after_cpu_128_128_result.png" alt="bench_cpu128_result"></div>
        ${opencl_shot_html}
      </div>
    </div>

    <div class="card">
      <h2>UI Action Logs</h2>
      <pre>$(inline_log_tail "$ARTIFACT_DIR/qwen35_benchmark/ui_actions.log" 160)</pre>
    </div>

    <div class="card">
      <h2>Dumpapp Logs</h2>
      <h3>case_cpu_64_64.log</h3>
      <pre>$(inline_log_tail "$bench_cpu64_log" 100)</pre>
      <h3>case_cpu_128_128.log</h3>
      <pre>$(inline_log_tail "$bench_cpu128_log" 100)</pre>
      <h3>case_opencl_64_64.log</h3>
      <pre>$(inline_log_tail "$bench_opencl_log" 100)</pre>
    </div>
  </div>
</body>
</html>
EOF_HTML

echo "REPORT_PATH=$REPORT_PATH"
