#!/bin/bash
# Batch image-inference test for visual LLM models via llm_demo.
# Runs each image through the model and collects timing stats.
#
# Usage: llm_vision_test.sh <config.json> <image|dir> [image|dir ...]
# Env:   LLM_DEMO    path to llm_demo binary (default: <repo>/build/llm_demo)
#        PROMPT      question appended after the image tag
#        OUT_DIR     directory for per-image logs (default: /tmp/llm_vision_test)

set -u

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config.json> <image|dir> [image|dir ...]" >&2
    exit 1
fi

CONFIG=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
shift
if [ ! -f "$CONFIG" ]; then
    echo "Error: config not found: $CONFIG" >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
LLM_DEMO=${LLM_DEMO:-"$REPO_ROOT/build/llm_demo"}
PROMPT=${PROMPT:-"描述一下这张图片里的内容。"}
OUT_DIR=${OUT_DIR:-/tmp/llm_vision_test}

if [ ! -x "$LLM_DEMO" ]; then
    echo "Error: llm_demo not found: $LLM_DEMO" >&2
    echo "Build with: cmake .. -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON && make llm_demo" >&2
    exit 1
fi

# collect images
IMAGES=()
for arg in "$@"; do
    if [ -d "$arg" ]; then
        while IFS= read -r f; do
            IMAGES+=("$f")
        done < <(find "$arg" -maxdepth 1 -type f \
                 \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
                    -o -iname '*.bmp' -o -iname '*.webp' \) | sort)
    elif [ -f "$arg" ]; then
        IMAGES+=("$arg")
    else
        echo "Warning: skip non-existent path: $arg" >&2
    fi
done

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "Error: no images found" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"
echo "image,prompt_tokens,decode_tokens,vision_s,vision_mps,prefill_tps,decode_tps,total_s,status" > "$SUMMARY"

PASS=0
FAIL=0
IDX=0
TOTAL=${#IMAGES[@]}

for img in "${IMAGES[@]}"; do
    IDX=$((IDX + 1))
    name=$(basename "$img")
    echo ""
    echo "===== [$IDX/$TOTAL] $name ====="

    prompt_file=$(mktemp)
    printf '<img>%s</img>%s\n' "$img" "$PROMPT" > "$prompt_file"

    run_dir="$OUT_DIR/run_$IDX"
    rm -rf "$run_dir" && mkdir -p "$run_dir"

    start_s=$(date +%s)
    log="$OUT_DIR/$IDX.$name.log"
    if "$LLM_DEMO" "$CONFIG" "$prompt_file" > "$log" 2>&1; then
        rc=0
    else
        rc=1
    fi
    total_s=$(( $(date +%s) - start_s ))
    rm -f "$prompt_file"

    # extract stats from the trailing benchmark block
    prompt_tokens=$(grep "prompt tokens num" "$log" | awk '{print $5}')
    decode_tokens=$(grep "decode tokens num" "$log" | awk '{print $5}')
    vision_s=$(grep "vision time" "$log" | awk '{print $4}')
    vision_mps=$(grep "vision speed" "$log" | awk '{print $4}')
    prefill_tps=$(grep "prefill speed" "$log" | awk '{print $4}')
    decode_tps=$(grep "decode speed" "$log" | awk '{print $4}')

    # show the model response (between prompt file line and the stats block)
    sed -n '/prompt file is/,/^####/p' "$log" | sed '1d;$d'

    status=OK
    if [ $rc -ne 0 ] || ! grep -q "decode speed" "$log"; then
        status=FAIL
        FAIL=$((FAIL + 1))
    else
        PASS=$((PASS + 1))
    fi

    echo "----------------------------------------"
    printf "vision=%.2fs (%.3f MP/s)  prefill=%s tok/s  decode=%s tok/s  total=%ss  [%s]\n" \
           "${vision_s:-0}" "${vision_mps:-0}" "${prefill_tps:-?}" "${decode_tps:-?}" "$total_s" "$status"
    echo "log: $log"

    echo "$name,${prompt_tokens:-},${decode_tokens:-},${vision_s:-},${vision_mps:-},${prefill_tps:-},${decode_tps:-},${total_s},${status}" >> "$SUMMARY"
done

echo ""
echo "========================================"
echo "Done: $PASS passed, $FAIL failed, total $TOTAL"
echo "Summary CSV: $SUMMARY"
