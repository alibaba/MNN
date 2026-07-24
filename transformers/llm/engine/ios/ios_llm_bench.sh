#!/bin/bash
# One-command iOS LLM benchmark:
#   build MNN.framework (current branch) -> build mnn-llm app with model bundled
#   -> install to a connected iPhone/iPad -> auto-run benchmark -> report tok/s.
#
# Usage:
#   sh transformers/llm/engine/ios/ios_llm_bench.sh --model /path/to/MNN_MODEL --team TEAM_ID [options]
#
# Options:
#   --model DIR       (required) exported MNN model dir (config.json / llm.mnn / ...)
#   --team ID         Apple Development Team ID (required unless --build-only without signing)
#   --device UDID     target device id (default: first connected iPhone/iPad)
#   --bundle-id ID    app bundle id (default: com.<user>.mnn-llm-bench)
#   --skip-framework  reuse existing MNN.framework, skip C++ build
#   --build-only      build framework + app only, do not install/run
#   --cmake-args "…"  extra CMake args appended to the framework build
#   --timeout SEC     benchmark wait timeout (default 1800)
#   --backend B       fixed-length bench backend: cpu | metal
#   --prompt-len N    fixed prompt length(s) in tokens, comma-separated for a matrix
#                     e.g. --prompt-len 512,1024,2048 (used with --backend)
#   --decode-len N    fixed decode length(s) in tokens, comma-separated for a matrix
#                     e.g. --decode-len 128,2000 (used with --backend)
#                     all prompt x decode combinations are benchmarked sequentially
#   --repeat N        fixed bench repeat count, first warmup run excluded (default 3)
#   --threads N       cpu thread num for fixed bench (default 4)
#   --prompt-dir DIR  prompt-file bench: run every *.txt in DIR as a prompt, log
#                     per-file prefill/decode tok/s and the full greedy answer
#                     (accuracy check). Uses --backend (default metal).
#   --max-new N       prompt-file bench: max new tokens per prompt (default 1024)
#
# Without --backend the default prompt-file benchmark (bench.txt) is used.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
APP_DIR="$SCRIPT_DIR/mnn-llm"
MODEL_DST="$APP_DIR/model"
LOG_DIR="$SCRIPT_DIR/bench_logs"

MODEL_DIR=""
TEAM_ID=""
DEVICE_ID=""
BUNDLE_ID=""
SKIP_FRAMEWORK=0
BUILD_ONLY=0
EXTRA_CMAKE_ARGS=""
TIMEOUT=1800
BACKEND=""
PROMPT_LEN=""
DECODE_LEN=""
REPEAT=3
THREADS=4
PROMPT_DIR=""
MAX_NEW=1024

while [ $# -gt 0 ]; do
    case "$1" in
        --model)          MODEL_DIR="$2"; shift 2 ;;
        --team)           TEAM_ID="$2"; shift 2 ;;
        --device)         DEVICE_ID="$2"; shift 2 ;;
        --bundle-id)      BUNDLE_ID="$2"; shift 2 ;;
        --skip-framework) SKIP_FRAMEWORK=1; shift ;;
        --build-only)     BUILD_ONLY=1; shift ;;
        --cmake-args)     EXTRA_CMAKE_ARGS="$2"; shift 2 ;;
        --timeout)        TIMEOUT="$2"; shift 2 ;;
        --backend)        BACKEND="$2"; shift 2 ;;
        --prompt-len)     PROMPT_LEN="$2"; shift 2 ;;
        --decode-len)     DECODE_LEN="$2"; shift 2 ;;
        --repeat)         REPEAT="$2"; shift 2 ;;
        --threads)        THREADS="$2"; shift 2 ;;
        --prompt-dir)     PROMPT_DIR="$2"; shift 2 ;;
        --max-new)        MAX_NEW="$2"; shift 2 ;;
        -h|--help)        sed -n '2,34p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Fixed-length mode: expand comma-separated prompt/decode lists into a combo matrix.
BENCH_COMBOS=()
if [ -n "$PROMPT_DIR" ]; then
    [ -d "$PROMPT_DIR" ] || { echo "[ERROR] --prompt-dir not found: $PROMPT_DIR" >&2; exit 1; }
    ls "$PROMPT_DIR"/*.txt >/dev/null 2>&1 || { echo "[ERROR] no *.txt files in $PROMPT_DIR" >&2; exit 1; }
    [ -n "$BACKEND" ] || BACKEND=metal
    case "$BACKEND" in cpu|metal) ;; *) echo "[ERROR] --backend must be cpu or metal" >&2; exit 1 ;; esac
elif [ -n "$BACKEND" ]; then
    case "$BACKEND" in cpu|metal) ;; *) echo "[ERROR] --backend must be cpu or metal" >&2; exit 1 ;; esac
    [ -n "$PROMPT_LEN" ] && [ -n "$DECODE_LEN" ] || { echo "[ERROR] --backend requires --prompt-len and --decode-len" >&2; exit 1; }
    for P in $(echo "$PROMPT_LEN" | tr ',' ' '); do
        for D in $(echo "$DECODE_LEN" | tr ',' ' '); do
            BENCH_COMBOS+=("$P $D")
        done
    done
fi

fail() { echo "[ERROR] $1" >&2; exit 1; }

# xcodebuild needs a full Xcode, not CommandLineTools
if ! xcode-select -p 2>/dev/null | grep -q "Xcode.app"; then
    if [ -d "/Applications/Xcode.app/Contents/Developer" ]; then
        export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    else
        fail "full Xcode not found; install Xcode or run: sudo xcode-select -s /Applications/Xcode.app"
    fi
fi

[ -n "$MODEL_DIR" ] || fail "--model is required (exported MNN model dir)"
[ -d "$MODEL_DIR" ] || fail "model dir not found: $MODEL_DIR"
[ -f "$MODEL_DIR/config.json" ] || fail "config.json not found in $MODEL_DIR"
[ -f "$MODEL_DIR/llm.mnn" ] || fail "llm.mnn not found in $MODEL_DIR"
if [ "$BUILD_ONLY" -eq 0 ] || [ -n "$TEAM_ID" ]; then
    [ -n "$TEAM_ID" ] || fail "--team is required to sign for a real device"
fi
if [ -z "$BUNDLE_ID" ]; then
    SAFE_USER=$(echo "$USER" | tr -cd '[:alnum:]')
    BUNDLE_ID="com.${SAFE_USER:-mnn}.mnn-llm-bench"
fi

echo "==> Repo root      : $ROOT_DIR"
echo "==> Model          : $MODEL_DIR"
echo "==> Bundle id      : $BUNDLE_ID"
echo "==> Git branch     : $(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD) ($(git -C "$ROOT_DIR" rev-parse --short HEAD))"

# ---------- 1. Build MNN.framework (with LLM) ----------
if [ "$SKIP_FRAMEWORK" -eq 1 ] && [ -d "$SCRIPT_DIR/MNN.framework" ]; then
    echo "==> [1/5] Skip framework build, reuse $SCRIPT_DIR/MNN.framework"
else
    echo "==> [1/5] Building MNN.framework (this can take a few minutes)..."
    cd "$ROOT_DIR"
    sh package_scripts/ios/buildiOS.sh "-DMNN_LOW_MEMORY=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_BUILD_LLM=true $EXTRA_CMAKE_ARGS"
    [ -d "$ROOT_DIR/MNN-iOS-CPU-GPU/Static/MNN.framework" ] || fail "framework build failed"
    rm -rf "$SCRIPT_DIR/MNN.framework"
    cp -R "$ROOT_DIR/MNN-iOS-CPU-GPU/Static/MNN.framework" "$SCRIPT_DIR/MNN.framework"
    echo "==> MNN.framework ready: $SCRIPT_DIR/MNN.framework"
fi

# ---------- 2. Prepare model ----------
echo "==> [2/5] Bundling model files..."
if [ "$(cd "$MODEL_DIR" && pwd)" = "$MODEL_DST" ]; then
    echo "    model already in $MODEL_DST, skip copy"
else
    rm -rf "$MODEL_DST"
    mkdir -p "$MODEL_DST"
    cp -R "$MODEL_DIR/." "$MODEL_DST/"
fi
if [ ! -f "$MODEL_DST/bench.txt" ]; then
    cp "$SCRIPT_DIR/bench.txt" "$MODEL_DST/bench.txt"
    echo "    bench.txt not in model dir, using default prompts"
fi
rm -f "$MODEL_DST"/benchprompt_*.txt
if [ -n "$PROMPT_DIR" ]; then
    N_PROMPTS=0
    for F in "$PROMPT_DIR"/*.txt; do
        BASE=$(basename "$F" .txt)
        BASE=${BASE//[^a-zA-Z0-9._-]/_}
        cp "$F" "$MODEL_DST/benchprompt_${BASE}.txt"
        N_PROMPTS=$((N_PROMPTS + 1))
    done
    echo "    bundled $N_PROMPTS prompt files from $PROMPT_DIR"
fi

# ---------- 3. Build app ----------
echo "==> [3/5] Building mnn-llm.app..."
cd "$APP_DIR"
APP_PATH="$APP_DIR/build/Release-iphoneos/mnn-llm.app"
rm -rf "$APP_PATH"
XCODE_ARGS=(-project mnn-llm.xcodeproj -target mnn-llm -configuration Release -sdk iphoneos
            SYMROOT="$APP_DIR/build" PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID")
if [ -n "$TEAM_ID" ]; then
    XCODE_ARGS+=(DEVELOPMENT_TEAM="$TEAM_ID" CODE_SIGN_STYLE=Automatic -allowProvisioningUpdates)
else
    XCODE_ARGS+=(CODE_SIGNING_ALLOWED=NO)
fi
XCODE_LOG=$(mktemp /tmp/mnn_xcodebuild.XXXXXX)
xcodebuild "${XCODE_ARGS[@]}" build > "$XCODE_LOG" 2>&1 || true
if grep -q "No available simulator runtimes" "$XCODE_LOG"; then
    # Xcode 15+ actool needs an iOS simulator runtime to compile asset catalogs.
    # Fall back to building without asset catalogs (app has no icon, benchmark unaffected).
    echo "    no simulator runtime installed, rebuilding without asset catalogs..."
    rm -rf "$APP_PATH"
    xcodebuild "${XCODE_ARGS[@]}" EXCLUDED_SOURCE_FILE_NAMES="*.xcassets" \
        ASSETCATALOG_COMPILER_APPICON_NAME="" build > "$XCODE_LOG" 2>&1 || true
fi
grep -E "error|BUILD" "$XCODE_LOG" || true
if [ ! -f "$APP_PATH/mnn-llm" ]; then
    echo "---- xcodebuild tail ----"; tail -30 "$XCODE_LOG"
    fail "app build failed, full log: $XCODE_LOG"
fi
rm -f "$XCODE_LOG"
echo "==> App built: $APP_PATH"

if [ "$BUILD_ONLY" -eq 1 ]; then
    echo "==> --build-only set, done."
    exit 0
fi

# ---------- 4. Select device & install ----------
echo "==> [4/5] Looking for a connected iPhone/iPad..."
if [ -z "$DEVICE_ID" ]; then
    DEVICE_JSON=$(mktemp /tmp/mnn_devices.XXXXXX)
    xcrun devicectl list devices --json-output "$DEVICE_JSON" >/dev/null
    DEVICE_ID=$(python3 - "$DEVICE_JSON" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
for d in data.get("result", {}).get("devices", []):
    props = d.get("deviceProperties", {})
    hw = d.get("hardwareProperties", {})
    state = d.get("connectionProperties", {}).get("tunnelState", "")
    if hw.get("deviceType") in ("iPhone", "iPad") and state != "unavailable":
        print(d.get("identifier", ""))
        break
EOF
)
    rm -f "$DEVICE_JSON"
    [ -n "$DEVICE_ID" ] || fail "no connected iPhone/iPad found; plug in via USB and trust this Mac, or pass --device UDID"
fi
echo "==> Target device: $DEVICE_ID"
echo "==> Installing app (first install: trust the developer cert in Settings > General > VPN & Device Management)..."
xcrun devicectl device install app --device "$DEVICE_ID" "$APP_PATH"

# ---------- 5. Launch & collect benchmark ----------
mkdir -p "$LOG_DIR"
RUN_TAG=$(date +%Y%m%d_%H%M%S)

# run_one_bench LOG_FILE LAUNCH_ARG... ; sets STATUS to done|error|timeout
run_one_bench() {
    local LOG_FILE="$1"; shift
    xcrun devicectl device process launch --console --terminate-existing \
        --device "$DEVICE_ID" "$BUNDLE_ID" "$@" > "$LOG_FILE" 2>&1 &
    local LAUNCH_PID=$!
    local ELAPSED=0
    STATUS="timeout"
    while kill -0 "$LAUNCH_PID" 2>/dev/null; do
        if grep -q "MNN_BENCH_DONE" "$LOG_FILE" 2>/dev/null; then STATUS="done"; break; fi
        if grep -q "MNN_BENCH_ERROR" "$LOG_FILE" 2>/dev/null; then STATUS="error"; break; fi
        if grep -q "App terminated due to signal" "$LOG_FILE" 2>/dev/null; then
            STATUS="error"
            SIG=$(grep -o "App terminated due to signal [0-9]*" "$LOG_FILE" | head -1)
            echo "[MNN_BENCH_ERROR] app crashed: $SIG" >> "$LOG_FILE"
            break
        fi
        if grep -q "The application failed to launch" "$LOG_FILE" 2>/dev/null; then
            STATUS="error"
            echo "[MNN_BENCH_ERROR] app failed to launch (device locked? unlock the screen and retry)" >> "$LOG_FILE"
            break
        fi
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then break; fi
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done
    kill "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
    # devicectl may exit between polls (e.g. app crash); re-check markers.
    if [ "$STATUS" = "timeout" ]; then
        if grep -q "MNN_BENCH_DONE" "$LOG_FILE" 2>/dev/null; then
            STATUS="done"
        elif grep -q "App terminated due to signal" "$LOG_FILE" 2>/dev/null; then
            STATUS="error"
            SIG=$(grep -o "App terminated due to signal [0-9]*" "$LOG_FILE" | head -1)
            echo "[MNN_BENCH_ERROR] app crashed: $SIG" >> "$LOG_FILE"
        fi
    fi
}

SUMMARY=""
FAILED=0
LOG_FILES=""
if [ -n "$PROMPT_DIR" ]; then
    PROMPT_FILES=($(cd "$MODEL_DST" && ls benchprompt_*.txt | sort))
    TOTAL=${#PROMPT_FILES[@]}
    for ((FI = 0; FI < TOTAL; FI++)); do
        FNAME=${PROMPT_FILES[$FI]}
        LOG_FILE="$LOG_DIR/bench_${RUN_TAG}_file${FI}_${FNAME%.txt}.log"
        LOG_FILES="$LOG_FILES $LOG_FILE"
        echo "==> [5/5] ($((FI + 1))/$TOTAL) $FNAME ($BACKEND, max_new=$MAX_NEW)..."
        run_one_bench "$LOG_FILE" --bench-cmd "benchfiles $BACKEND $THREADS $MAX_NEW $FI"
        case "$STATUS" in
            done)
                LINE=$(grep -o "\[MNN_FILE_PERF\] .*" "$LOG_FILE" | sed 's/\[MNN_FILE_PERF\] //') ;;
            error)
                LINE="FAILED file=$FNAME: $(grep -o "\[MNN_BENCH_ERROR\] .*" "$LOG_FILE" | head -1)"; FAILED=1 ;;
            timeout)
                LINE="TIMEOUT file=$FNAME after ${TIMEOUT}s"; FAILED=1 ;;
        esac
        echo "    $LINE"
        SUMMARY="$SUMMARY$LINE
"
    done
elif [ ${#BENCH_COMBOS[@]} -gt 0 ]; then
    TOTAL=${#BENCH_COMBOS[@]}
    IDX=0
    for COMBO in "${BENCH_COMBOS[@]}"; do
        IDX=$((IDX + 1))
        P=${COMBO% *}
        D=${COMBO#* }
        LOG_FILE="$LOG_DIR/bench_${RUN_TAG}_p${P}_d${D}.log"
        LOG_FILES="$LOG_FILES $LOG_FILE"
        echo "==> [5/5] ($IDX/$TOTAL) bench $BACKEND prompt=$P decode=$D (model load may take a while)..."
        run_one_bench "$LOG_FILE" --bench-cmd "bench $BACKEND $P $D $REPEAT $THREADS"
        case "$STATUS" in
            done)
                LINE=$(grep -o "\[MNN_BENCH\] avg .*" "$LOG_FILE" | sed 's/\[MNN_BENCH\] //') ;;
            error)
                LINE="FAILED prompt=$P decode=$D: $(grep -o "\[MNN_BENCH_ERROR\] .*" "$LOG_FILE" | head -1)"; FAILED=1 ;;
            timeout)
                LINE="TIMEOUT prompt=$P decode=$D after ${TIMEOUT}s"; FAILED=1 ;;
        esac
        echo "    $LINE"
        SUMMARY="$SUMMARY$LINE
"
    done
else
    LOG_FILE="$LOG_DIR/bench_${RUN_TAG}.log"
    LOG_FILES="$LOG_FILE"
    echo "==> [5/5] Launching benchmark (model load may take a while)..."
    run_one_bench "$LOG_FILE" --auto-bench
    case "$STATUS" in
        done)
            SUMMARY=$(grep -o "\[MNN_BENCH\] .*" "$LOG_FILE" | sed 's/\[MNN_BENCH\] //') ;;
        error)
            SUMMARY="FAILED: $(grep -o "\[MNN_BENCH_ERROR\] .*" "$LOG_FILE" | head -1)"; FAILED=1 ;;
        timeout)
            SUMMARY="TIMEOUT after ${TIMEOUT}s (app may have crashed or model too slow to load)"; FAILED=1 ;;
    esac
fi

echo ""
echo "================ iOS LLM Benchmark Report ================"
echo "branch : $(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD) ($(git -C "$ROOT_DIR" rev-parse --short HEAD))"
echo "model  : $(basename "$MODEL_DIR")"
echo "device : $DEVICE_ID"
printf '%s\n' "$SUMMARY"
echo "log    :$LOG_FILES"
echo "==========================================================="
[ "$FAILED" -eq 0 ] || exit 1
