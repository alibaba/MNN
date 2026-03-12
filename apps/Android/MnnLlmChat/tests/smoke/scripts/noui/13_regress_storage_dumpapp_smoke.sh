#!/usr/bin/env bash
# 13_regress_storage_dumpapp_smoke.sh
# Smoke test for dumpapp storage commands (merged from MNN tools/smoke/10_storage_dumpapp_smoke.sh).
#
# Verifies:
#   1. All dumpapp storage subcommands run without error
#   2. Sum of dataDir children sizes == total dataDir size
#   3. Sum of filesDir children sizes == total filesDir size
#   4. Mmap dir-scan total == ModelDeletionHelper analysis total
#   5. Orphan size <= total mmap size
#   6. filesDir total <= dataDir total
#
# Prerequisites:
#   - Device connected via adb
#   - App installed and running (script will attempt to start it)
#
# Usage: ./13_regress_storage_dumpapp_smoke.sh [SUMMARY_FILE]
# Exit codes: 0 = ALL PASS, 1 = HAS FAILURES

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
DUMPAPP="${DUMPAPP:-$SMOKE_DIR/../../tools/dumpapp}"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
OUT_DIR="$ARTIFACT_DIR/storage_dumpapp_smoke"
SUMMARY_FILE="${1:-$OUT_DIR/summary.txt}"
mkdir -p "$OUT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

run_dumpapp() { python3 "$DUMPAPP" -p "$PACKAGE_NAME" "$@" 2>&1; }

# ---------- Preconditions ----------

ensure_device() {
    local devs
    devs=$(adb devices 2>&1 | grep -w "device" | grep -v "List" || true)
    if [ -z "$devs" ]; then
        fail "No device connected (adb devices shows none)"
        exit 1
    fi
    pass "Device connected"
}

ensure_app() {
    if adb shell pidof "$PACKAGE_NAME" > /dev/null 2>&1; then
        pass "App already running"
    else
        info "App not running, launching..."
        adb shell am start -n "$PACKAGE_NAME/.main.MainActivity" > /dev/null 2>&1
        sleep 3
        if adb shell pidof "$PACKAGE_NAME" > /dev/null 2>&1; then
            pass "App launched"
        else
            fail "Could not start app"
            exit 1
        fi
    fi
}

# ---------- Subcommand output tests ----------

test_subcommand() {
    local cmd="$1"
    local expected_pattern="$2"
    local label="$3"

    local output
    output=$(run_dumpapp storage $cmd)

    if echo "$output" | grep -qE "$expected_pattern"; then
        pass "$label"
    else
        fail "$label"
        info "Output: $(echo "$output" | head -5)"
    fi

    # Check for errors / exceptions
    if echo "$output" | grep -qi "exception\|traceback"; then
        fail "$label: unexpected exception in output"
    fi
}

test_all_subcommands() {
    info "--- Subcommand output tests ---"
    test_subcommand ""         "Usage: dumpapp storage"     "usage: shows help"
    test_subcommand "list"     "Internal Storage Analysis"  "list: header present"
    test_subcommand "list"     "Total data dir size:"       "list: total data dir size"
    test_subcommand "list"     "Total files dir size:"      "list: total files dir size"
    test_subcommand "analysis" "Storage Analysis"           "analysis: header present"
    test_subcommand "analysis" "Model Storage:"             "analysis: model storage section"
    test_subcommand "mmap"     "Mmap Cache Directories"     "mmap: header present"
    test_subcommand "orphans"  "Orphan Mmap Caches"         "orphans: header present"
    test_subcommand "verify"   "Storage Verification"       "verify: header present"
}

# ---------- Storage integrity (verify) ----------

test_verify_checks() {
    info "--- Storage integrity checks ---"

    local output
    output=$(run_dumpapp storage verify)
    echo "$output" > "$OUT_DIR/verify.log"

    # Parse each CHECK_ line
    local checks=("CHECK_DATA_DIR_SUM" "CHECK_FILES_DIR_SUM" "CHECK_MMAP_TOTAL"
                   "CHECK_ORPHAN_LE_TOTAL" "CHECK_FILES_LE_DATA")

    for chk in "${checks[@]}"; do
        local line
        line=$(echo "$output" | grep "^${chk}=" || true)
        if [ -z "$line" ]; then
            fail "verify: $chk line missing"
            continue
        fi
        local value="${line#*=}"
        if [ "$value" = "PASS" ]; then
            pass "verify: $chk"
        else
            fail "verify: $chk ($value)"
        fi
    done

    # Overall verdict
    local verdict
    verdict=$(echo "$output" | grep "^VERIFY_RESULT=" | head -1 || true)
    if echo "$verdict" | grep -q "ALL_PASS"; then
        pass "verify: overall ALL_PASS"
    else
        fail "verify: overall ${verdict:-missing}"
    fi

    # Sanity: DATA_DIR_TOTAL_BYTES must be > 0
    local total
    total=$(echo "$output" | grep "^DATA_DIR_TOTAL_BYTES=" | head -1 | cut -d= -f2 || echo "0")
    if [ "$total" -gt 0 ] 2>/dev/null; then
        pass "verify: DATA_DIR_TOTAL_BYTES > 0 ($total bytes)"
    else
        fail "verify: DATA_DIR_TOTAL_BYTES should be > 0 (got: $total)"
    fi

    # Sanity: FILES_DIR_TOTAL_BYTES must be > 0
    total=$(echo "$output" | grep "^FILES_DIR_TOTAL_BYTES=" | head -1 | cut -d= -f2 || echo "0")
    if [ "$total" -gt 0 ] 2>/dev/null; then
        pass "verify: FILES_DIR_TOTAL_BYTES > 0 ($total bytes)"
    else
        fail "verify: FILES_DIR_TOTAL_BYTES should be > 0 (got: $total)"
    fi

    # Cross-check: children listed
    local child_count
    child_count=$(echo "$output" | grep -c "^CHILD:" || echo "0")
    if [ "$child_count" -gt 0 ]; then
        pass "verify: child entries present ($child_count items)"
    else
        fail "verify: no CHILD: entries in output"
    fi

    # Cross-check: sum of CHILD: values == DATA_DIR_CHILDREN_SUM_BYTES
    local children_sum_reported
    children_sum_reported=$(echo "$output" | grep "^DATA_DIR_CHILDREN_SUM_BYTES=" | head -1 | cut -d= -f2 || echo "0")
    local children_sum_calculated=0
    while IFS= read -r line; do
        local val="${line#*=}"
        children_sum_calculated=$((children_sum_calculated + val))
    done < <(echo "$output" | grep "^CHILD:" || true)
    if [ "$children_sum_calculated" -eq "$children_sum_reported" ] 2>/dev/null; then
        pass "verify: CHILD sum ($children_sum_calculated) == reported ($children_sum_reported)"
    else
        fail "verify: CHILD sum ($children_sum_calculated) != reported ($children_sum_reported)"
    fi

    # ---------- Mmap entries consistency (catches missing builtin_temps or extra dirs) ----------
    local mmap_tmps mmap_local mmap_builtin mmap_total analysis_total
    mmap_tmps=$(echo "$output" | grep "^MMAP_TMPS_BYTES=" | head -1 | cut -d= -f2 || echo "0")
    mmap_local=$(echo "$output" | grep "^MMAP_LOCAL_TEMPS_BYTES=" | head -1 | cut -d= -f2 || echo "0")
    mmap_builtin=$(echo "$output" | grep "^MMAP_BUILTIN_TEMPS_BYTES=" | head -1 | cut -d= -f2 || echo "0")
    mmap_total=$(echo "$output" | grep "^MMAP_TOTAL_BYTES=" | head -1 | cut -d= -f2 || echo "0")
    analysis_total=$(echo "$output" | grep "^ANALYSIS_MMAP_TOTAL_BYTES=" | head -1 | cut -d= -f2 || echo "0")

    local mmap_sum=$((mmap_tmps + mmap_local + mmap_builtin))
    if [ "$mmap_sum" -eq "$mmap_total" ] 2>/dev/null; then
        pass "verify: mmap entries sum (tmps+local_temps+builtin_temps)=$mmap_sum == MMAP_TOTAL_BYTES=$mmap_total"
    else
        fail "verify: mmap entries sum ($mmap_sum) != MMAP_TOTAL_BYTES ($mmap_total); tmps=$mmap_tmps local_temps=$mmap_local builtin_temps=$mmap_builtin"
    fi

    if [ "$mmap_total" -eq "$analysis_total" ] 2>/dev/null; then
        pass "verify: MMAP_TOTAL_BYTES ($mmap_total) == ANALYSIS_MMAP_TOTAL_BYTES ($analysis_total)"
    else
        fail "verify: MMAP_TOTAL_BYTES ($mmap_total) != ANALYSIS_MMAP_TOTAL_BYTES ($analysis_total) (ModelDeletionHelper must include same mmap dirs: tmps, local_temps, builtin_temps)"
    fi
}

# ---------- Main ----------

main() {
    echo "=========================================="
    echo "  Storage Dumpapp Smoke Test"
    echo "=========================================="
    echo ""

    ensure_device
    ensure_app

    echo ""
    test_all_subcommands

    echo ""
    test_verify_checks

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT PASS, $FAIL_COUNT FAIL"
    echo "=========================================="

    # Write summary for report pipeline
    {
        echo "STORAGE_DUMPAPP_SMOKE=$([ "$FAIL_COUNT" -eq 0 ] && echo PASS || echo FAIL)"
        echo "PASS_COUNT=$PASS_COUNT"
        echo "FAIL_COUNT=$FAIL_COUNT"
        echo "VERIFY_LOG=$OUT_DIR/verify.log"
    } > "$SUMMARY_FILE"

    if [ "$FAIL_COUNT" -gt 0 ]; then
        exit 1
    fi
    exit 0
}

main "$@"
