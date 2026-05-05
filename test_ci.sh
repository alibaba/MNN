#!/usr/bin/env bash
# test_ci.sh - Focused Android-arm64 + local-host CI driver for MNN.
#
# Subcommands:
#   ./test_ci.sh local                  Run host-side regression: build + the
#                                       built-in unit-test suite (CPU / OpenCL
#                                       / Vulkan) and the LLM smoke test.
#   ./test_ci.sh android <serial>       Build for arm64-v8a, push artefacts and
#                                       the LLM model to /data/local/tmp, then
#                                       run the on-device matrix.
#
# Environment:
#   ANDROID_NDK         (required for android mode) NDK root. Falls back to
#                       $HOME/android-ndk-r21 when unset.
#   LLM_MODEL_REPO      HuggingFace repo id used for the LLM smoke test.
#                       Defaults to taobao-mnn/Qwen2.5-0.5B-Instruct-MNN.
#   LLM_MODEL_URL_BASE  Override the resolve URL prefix (defaults to
#                       https://huggingface.co/<repo>/resolve/main).
#
# Notes:
#   * Replaces project/android/updateTest.sh natively (no shell-out).
#   * Prefers `adbk` over `adb` and manages session lifecycle (--create-session
#     on entry, --delete-session on EXIT, including failure paths).
#   * Mirrors every OpenCL (backend=3) probe with a Vulkan (backend=7) probe;
#     skipped (not failed) when the backend library is absent.

set -euo pipefail
IFS=$'\n\t'
umask 022

# ─────────────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
PUBLIC_MODELS_DIR="${SCRIPT_DIR}/resource/model"
DEVICE_DIR="/data/local/tmp/MNN"
DEVICE_MODEL_DIR="/data/local/tmp/MNN/models"
DEVICE_PUBLIC_MODELS_DIR="/data/local/tmp/MNN/public_models"

# Public smoke-test models. Populated by tools/script/get_model.sh from
# upstream MobileNet/SqueezeNet GitHub repos and the TensorFlow model zoo.
SMOKE_MODELS=(
    MobileNet/v1/mobilenet_v1.caffe.mnn
    MobileNet/v2/mobilenet_v2.caffe.mnn
    SqueezeNet/v1.0/squeezenet_v1.0.caffe.mnn
    SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn
)

LLM_MODEL_REPO="${LLM_MODEL_REPO:-taobao-mnn/Qwen2.5-0.5B-Instruct-MNN}"
LLM_MODEL_URL_BASE="${LLM_MODEL_URL_BASE:-https://huggingface.co/${LLM_MODEL_REPO}/resolve/main}"
LLM_MODEL_NAME="$(basename "${LLM_MODEL_REPO}")"
LLM_MODEL_DIR="${MODELS_DIR}/${LLM_MODEL_NAME}"
# Files that must be present in a complete MNN-format LLM checkout.
LLM_MODEL_FILES=(
    config.json
    llm_config.json
    llm.mnn
    llm.mnn.json
    llm.mnn.weight
    embeddings_bf16.bin
    tokenizer.txt
)

MODE=""
DEVICE=""
ADB_BIN=""
USE_ADBK=0
SESSION_CREATED=0

STAGE_PASS=0
STAGE_FAIL=0
STAGE_SKIP=0
declare -a STAGE_LOG=()

# Per-run log directory; each stage's combined stdout/stderr lands here.
LOG_DIR="${SCRIPT_DIR}/logs/test_ci-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_RST=$'\033[0m'; C_DIM=$'\033[2m'; C_RED=$'\033[31m'
    C_GRN=$'\033[32m'; C_YLW=$'\033[33m'; C_CYN=$'\033[36m'; C_BLD=$'\033[1m'
else
    C_RST=""; C_DIM=""; C_RED=""; C_GRN=""; C_YLW=""; C_CYN=""; C_BLD=""
fi

_ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log_info() { printf "%s %sINFO%s  %s\n" "$(_ts)" "${C_CYN}" "${C_RST}" "$*"; }
log_ok()   { printf "%s %sOK%s    %s\n" "$(_ts)" "${C_GRN}" "${C_RST}" "$*"; }
log_warn() { printf "%s %sWARN%s  %s\n" "$(_ts)" "${C_YLW}" "${C_RST}" "$*" >&2; }
log_err()  { printf "%s %sERROR%s %s\n" "$(_ts)" "${C_RED}" "${C_RST}" "$*" >&2; }
log_step() {
    printf "\n%s%s═══ %s ═══%s\n" "${C_BLD}" "${C_CYN}" "$*" "${C_RST}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage runner
# ─────────────────────────────────────────────────────────────────────────────
# Usage: run_stage <name> -- <cmd...>
# Prints a delimited block, captures status, never aborts the script.
run_stage() {
    local name="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    log_step "stage: ${name}"
    # Mirror combined stdout/stderr into a per-stage log so failures stay
    # diagnosable after the run. Stage names contain '/'; flatten to '_'.
    local logname="${name//\//_}"
    local logfile="${LOG_DIR}/${logname}.log"
    local rc=0
    if "$@" 2>&1 | tee "${logfile}"; then
        rc=0
    else
        # PIPESTATUS[0] is the command's rc; tee almost always returns 0.
        rc=${PIPESTATUS[0]}
    fi
    if [[ $rc -eq 0 ]]; then
        STAGE_PASS=$((STAGE_PASS + 1))
        STAGE_LOG+=("PASS  ${name}")
        log_ok "stage '${name}' passed"
    else
        STAGE_FAIL=$((STAGE_FAIL + 1))
        STAGE_LOG+=("FAIL  ${name} (rc=${rc}, log=${logfile})")
        log_err "stage '${name}' failed (rc=${rc}); log: ${logfile}"
        if [[ $rc -eq 137 ]]; then
            log_warn "rc=137 = SIGKILL — likely OOM-killed. Check 'dmesg | tail' on Linux"
        elif [[ $rc -eq 139 ]]; then
            log_warn "rc=139 = SIGSEGV — process crashed. See ${logfile} for trailing output"
        fi
    fi
    return 0
}

skip_stage() {
    local name="$1" reason="${2:-no reason given}"
    STAGE_SKIP=$((STAGE_SKIP + 1))
    STAGE_LOG+=("SKIP  ${name} (${reason})")
    log_warn "stage '${name}' skipped: ${reason}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Usage / arg parsing
# ─────────────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 <subcommand> [args]

Subcommands:
  local                    Run host-side regression suite.
  android <device_serial>  Build arm64-v8a, push and test on device.

Environment:
  ANDROID_NDK              NDK root (android mode). Defaults to
                           \$HOME/android-ndk-r21 when unset.
  LLM_MODEL_REPO           HuggingFace repo id for the LLM smoke test.
                           Default: ${LLM_MODEL_REPO}.
  LLM_MODEL_URL_BASE       Override resolve URL prefix.

Examples:
  $0 local
  $0 android emulator-5554
  LLM_MODEL_REPO=taobao-mnn/Qwen2-0.5B-Instruct-MNN $0 local
EOF
}

parse_args() {
    if [[ $# -lt 1 ]]; then usage; exit 2; fi
    MODE="$1"; shift
    case "${MODE}" in
        local) ;;
        android)
            if [[ $# -lt 1 || -z "${1:-}" ]]; then
                log_err "android mode requires a device serial"
                usage; exit 2
            fi
            DEVICE="$1"; shift
            ;;
        -h|--help|help) usage; exit 0 ;;
        *) log_err "unknown subcommand: ${MODE}"; usage; exit 2 ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# adb / adbk wrapper
# ─────────────────────────────────────────────────────────────────────────────
detect_adb() {
    local adbk_path adb_path
    adbk_path="$(command -v adbk || true)"
    adb_path="$(command -v adb || true)"
    if [[ -n "${adbk_path}" ]]; then
        ADB_BIN="${adbk_path}"
        USE_ADBK=1
        log_info "using adbk at ${adbk_path}"
    elif [[ -n "${adb_path}" ]]; then
        ADB_BIN="${adb_path}"
        USE_ADBK=0
        log_info "using adb at ${adb_path} (adbk not found)"
    else
        log_err "neither adbk nor adb found on PATH"
        exit 1
    fi
}

ad() { "${ADB_BIN}" -s "${DEVICE}" "$@"; }

ensure_adbk_session() {
    [[ ${USE_ADBK} -eq 1 ]] || return 0
    local user current
    user="${USER:-$(id -un)}"
    if current="$("${ADB_BIN}" --status 2>/dev/null)"; then
        if printf '%s\n' "${current}" | grep -q "${DEVICE}.*Session.*${user}"; then
            log_info "reusing existing adbk session for ${DEVICE}"
            return 0
        fi
    fi
    log_info "creating adbk session for ${DEVICE}"
    if ! "${ADB_BIN}" -s "${DEVICE}" --create-session >/dev/null; then
        log_err "adbk --create-session failed"
        exit 1
    fi
    SESSION_CREATED=1
    log_ok "adbk session created"
}

teardown_adbk_session() {
    [[ ${USE_ADBK} -eq 1 && ${SESSION_CREATED} -eq 1 ]] || return 0
    log_info "deleting adbk session for ${DEVICE}"
    "${ADB_BIN}" -s "${DEVICE}" --delete-session >/dev/null 2>&1 || true
}

verify_device_online() {
    if ! ad shell echo ok >/dev/null 2>&1; then
        log_err "device ${DEVICE} is not reachable via ${ADB_BIN}"
        exit 1
    fi
    log_ok "device ${DEVICE} online"
}

# ─────────────────────────────────────────────────────────────────────────────
# Exit handling
# ─────────────────────────────────────────────────────────────────────────────
print_summary() {
    local total=$((STAGE_PASS + STAGE_FAIL + STAGE_SKIP))
    printf "\n%s════════════════ test_ci.sh summary ════════════════%s\n" "${C_BLD}" "${C_RST}"
    printf "  mode    : %s\n" "${MODE}"
    [[ "${MODE}" == "android" ]] && printf "  device  : %s\n" "${DEVICE}"
    printf "  total   : %d\n" "${total}"
    printf "  %spassed %s : %d\n" "${C_GRN}" "${C_RST}" "${STAGE_PASS}"
    printf "  %sfailed %s : %d\n" "${C_RED}" "${C_RST}" "${STAGE_FAIL}"
    printf "  %sskipped%s : %d\n" "${C_YLW}" "${C_RST}" "${STAGE_SKIP}"
    if [[ ${#STAGE_LOG[@]} -gt 0 ]]; then
        printf "  %sstages%s :\n" "${C_DIM}" "${C_RST}"
        local line
        for line in "${STAGE_LOG[@]}"; do
            printf "    %s\n" "${line}"
        done
    fi
    printf "%s════════════════════════════════════════════════════%s\n" "${C_BLD}" "${C_RST}"
}

_on_exit() {
    local rc=$?
    teardown_adbk_session
    # Suppress the summary on usage / arg-parse errors (no stages attempted).
    local total=$((STAGE_PASS + STAGE_FAIL + STAGE_SKIP))
    if [[ ${total} -gt 0 ]]; then
        print_summary
    fi
    if [[ ${STAGE_FAIL} -gt 0 ]]; then exit 1; fi
    exit "${rc}"
}
trap _on_exit EXIT

# ─────────────────────────────────────────────────────────────────────────────
# LLM model provisioning (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────
# Cache layout: ${MODELS_DIR}/<repo-basename>/{config.json,llm.mnn,...}
# Plus a generated prompt.txt so llm_demo has something to consume.
llm_model_complete() {
    local f
    for f in "${LLM_MODEL_FILES[@]}"; do
        [[ -s "${LLM_MODEL_DIR}/${f}" ]] || return 1
    done
    return 0
}

provision_llm_model() {
    log_step "provisioning LLM model: ${LLM_MODEL_REPO}"
    if llm_model_complete; then
        log_ok "LLM model cache hit at ${LLM_MODEL_DIR}"
    else
        mkdir -p "${LLM_MODEL_DIR}"
        local f url tmp
        for f in "${LLM_MODEL_FILES[@]}"; do
            local dst="${LLM_MODEL_DIR}/${f}"
            if [[ -s "${dst}" ]]; then
                continue
            fi
            url="${LLM_MODEL_URL_BASE}/${f}"
            tmp="${dst}.part"
            log_info "fetching ${f}"
            if ! curl -fL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"; then
                log_err "failed to download ${url}"
                rm -f "${tmp}"
                exit 1
            fi
            mv "${tmp}" "${dst}"
        done
        log_ok "LLM model staged at ${LLM_MODEL_DIR}"
    fi
    # Always (re)write a small prompt for the smoke test.
    local prompt_path="${LLM_MODEL_DIR}/prompt.txt"
    if [[ ! -s "${prompt_path}" ]]; then
        printf 'Hello, who are you?\n' > "${prompt_path}"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Public smoke models (resource/model/) — populated by tools/script/get_model.sh
# from upstream MobileNet/SqueezeNet GitHub repos.
# ─────────────────────────────────────────────────────────────────────────────
public_models_complete() {
    local m
    for m in "${SMOKE_MODELS[@]}"; do
        [[ -s "${PUBLIC_MODELS_DIR}/${m}" ]] || return 1
    done
    return 0
}

# Returns 0 if the public smoke set is ready, 1 if it could not be provisioned
# (e.g. host MNNConvert is missing). Caller decides whether to skip stages.
provision_public_models() {
    log_step "provisioning public smoke models"
    if public_models_complete; then
        log_ok "public model cache hit at ${PUBLIC_MODELS_DIR}"
        return 0
    fi
    if [[ ! -x "${SCRIPT_DIR}/build/MNNConvert" ]]; then
        log_warn "build/MNNConvert missing — cannot run get_model.sh; smoke stages will skip"
        return 1
    fi
    log_info "running tools/script/get_model.sh"
    if ! bash "${SCRIPT_DIR}/tools/script/get_model.sh"; then
        log_warn "get_model.sh failed — smoke stages will skip"
        return 1
    fi
    if ! public_models_complete; then
        log_warn "get_model.sh did not produce all expected .mnn files — smoke stages will skip"
        return 1
    fi
    log_ok "public smoke models ready"
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# NDK helpers
# ─────────────────────────────────────────────────────────────────────────────
ensure_android_ndk() {
    if [[ -n "${ANDROID_NDK:-}" && -d "${ANDROID_NDK}" ]]; then
        log_info "using ANDROID_NDK=${ANDROID_NDK}"
        return 0
    fi
    local fallback="${HOME}/android-ndk-r21"
    if [[ -d "${fallback}" ]]; then
        export ANDROID_NDK="${fallback}"
        log_warn "ANDROID_NDK unset; falling back to ${fallback}"
        return 0
    fi
    log_err "ANDROID_NDK not set and ${fallback} does not exist"
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Local-mode build
# ─────────────────────────────────────────────────────────────────────────────
_host_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    else
        sysctl -n hw.ncpu 2>/dev/null || echo 4
    fi
}

local_build() {
    log_step "configuring + building host (build/) [CPU-only]"
    local build_dir="${SCRIPT_DIR}/build"
    mkdir -p "${build_dir}"
    pushd "${build_dir}" >/dev/null
    # Local mode is CPU-only: host GPU drivers are usually unavailable or
    # unreliable on dev machines, so we omit OpenCL/Vulkan to keep the
    # build fast and the test surface meaningful.
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_TEST=ON \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_BUILD_CONVERTER=ON
    make -j"$(_host_jobs)"
    popd >/dev/null
    log_ok "host build complete"
}

# Source archives for the smoke corpus. Hosts the upstream URLs that
# tools/script/get_model.sh hits. Format per row:
#   <url> <local_basename> <output_mnn_relpath> <framework>
# framework is "CAFFE" (paired prototxt download follows immediately) or
# "TFLITE" (single .tflite payload).
# Total payload ~40 MB.
SMOKE_SOURCES=(
    "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet.caffemodel|mobilenet_v1.caffemodel|MobileNet/v1/mobilenet_v1.caffe.mnn|CAFFE"
    "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_deploy.prototxt|mobilenet_v1.prototxt|MobileNet/v1/mobilenet_v1.caffe.mnn|CAFFE_PROTOTXT"
    "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel|mobilenet_v2.caffemodel|MobileNet/v2/mobilenet_v2.caffe.mnn|CAFFE"
    "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt|mobilenet_v2.prototxt|MobileNet/v2/mobilenet_v2.caffe.mnn|CAFFE_PROTOTXT"
    "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel|squeezenet_v1.0.caffemodel|SqueezeNet/v1.0/squeezenet_v1.0.caffe.mnn|CAFFE"
    "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/deploy.prototxt|squeezenet_v1.0.prototxt|SqueezeNet/v1.0/squeezenet_v1.0.caffe.mnn|CAFFE_PROTOTXT"
    "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel|squeezenet_v1.1.caffemodel|SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn|CAFFE"
    "https://raw.githubusercontent.com/DeepScale/SqueezeNet/b6b5ae2ce884a3866c21efd31e103defde8631ae/SqueezeNet_v1.1/deploy.prototxt|squeezenet_v1.1.prototxt|SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn|CAFFE_PROTOTXT"
)
SMOKE_SOURCES_DIR="${SCRIPT_DIR}/smoke_sources"

# Map MNN-output relpath -> (caffemodel, prototxt) basenames. Mirrors the
# pairing in SMOKE_SOURCES above and is consumed by android conversion.
_smoke_pair_for() {
    case "$1" in
        MobileNet/v1/mobilenet_v1.caffe.mnn)
            printf '%s\n%s\n' mobilenet_v1.caffemodel mobilenet_v1.prototxt ;;
        MobileNet/v2/mobilenet_v2.caffe.mnn)
            printf '%s\n%s\n' mobilenet_v2.caffemodel mobilenet_v2.prototxt ;;
        SqueezeNet/v1.0/squeezenet_v1.0.caffe.mnn)
            printf '%s\n%s\n' squeezenet_v1.0.caffemodel squeezenet_v1.0.prototxt ;;
        SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn)
            printf '%s\n%s\n' squeezenet_v1.1.caffemodel squeezenet_v1.1.prototxt ;;
        *)
            return 1 ;;
    esac
}

# Download upstream caffe sources to the host cache. Skips files already
# present. Returns non-zero if any download fails.
provision_smoke_sources() {
    log_step "fetching smoke-model sources"
    mkdir -p "${SMOKE_SOURCES_DIR}"
    local entry url fname rest
    for entry in "${SMOKE_SOURCES[@]}"; do
        url="${entry%%|*}"
        rest="${entry#*|}"
        fname="${rest%%|*}"
        local dst="${SMOKE_SOURCES_DIR}/${fname}"
        if [[ -s "${dst}" ]]; then
            continue
        fi
        log_info "fetching ${fname}"
        if ! curl -fL --retry 3 --retry-delay 2 -o "${dst}.part" "${url}"; then
            log_err "failed to download ${url}"
            rm -f "${dst}.part"
            return 1
        fi
        mv "${dst}.part" "${dst}"
    done
    log_ok "smoke sources cached at ${SMOKE_SOURCES_DIR}"
    return 0
}

# Detect whether a backend library was built locally.
local_has_lib() {
    local prefix="$1"
    compgen -G "${SCRIPT_DIR}/build/${prefix}.*" >/dev/null
}

# ─────────────────────────────────────────────────────────────────────────────
# Local-mode stages
# ─────────────────────────────────────────────────────────────────────────────
_local_unit() { (cd "${SCRIPT_DIR}/build" && ./run_test.out "$@"); }
_local_v2basic() {
    (cd "${SCRIPT_DIR}/build" && ./MNNV2Basic.out "$@")
}
_local_backendtest() {
    (cd "${SCRIPT_DIR}/build" && ./backendTest.out "$@")
}

# Stage A: load+forward smoke via MNNV2Basic.out (no numeric validation).
local_smoke_a_stages() {
    local backend_id="$1" backend_name="$2" prefix="$3"
    if [[ -n "${prefix}" ]] && ! local_has_lib "${prefix}"; then
        skip_stage "smokeA/${backend_name}" "lib${prefix} not built"
        return 0
    fi
    local m short
    for m in "${SMOKE_MODELS[@]}"; do
        short="${m##*/}"
        if [[ ! -s "${PUBLIC_MODELS_DIR}/${m}" ]]; then
            skip_stage "smokeA/${backend_name}/${short}" "model file missing"
            continue
        fi
        run_stage "smokeA/${backend_name}/${short}" -- \
            _local_v2basic "${PUBLIC_MODELS_DIR}/${m}" 1 0 "${backend_id}"
    done
}

# Stage B: CPU-as-oracle correctness check via backendTest.out.
local_smoke_b_stages() {
    local backend_id="$1" backend_name="$2" prefix="$3" tolerance="${4:-0.05}"
    if ! local_has_lib "${prefix}"; then
        skip_stage "smokeB/${backend_name}" "lib${prefix} not built"
        return 0
    fi
    local m short
    for m in "${SMOKE_MODELS[@]}"; do
        short="${m##*/}"
        if [[ ! -s "${PUBLIC_MODELS_DIR}/${m}" ]]; then
            skip_stage "smokeB/${backend_name}/${short}" "model file missing"
            continue
        fi
        run_stage "smokeB/${backend_name}/${short}" -- \
            _local_backendtest "${PUBLIC_MODELS_DIR}/${m}" "${backend_id}" "${tolerance}"
    done
}

local_run_stages() {
    # Local mode is CPU-only by design — see local_build() comment.
    # Unit tests use built-in op cases; no external corpus needed.
    run_stage "unit/cpu"    -- _local_unit
    run_stage "unit/cpu-mt" -- _local_unit op 0 0 4

    # Stage A on CPU: forward-smoke per public model. Catches model-load /
    # shape-inference regressions. (Stage B is CPU-vs-backend, meaningless
    # here without GPU, so it's skipped in local mode.)
    if [[ ! -x "${SCRIPT_DIR}/build/MNNV2Basic.out" ]]; then
        skip_stage "smokeA" "build/MNNV2Basic.out not built (check MNN_BUILD_TOOLS)"
    elif ! public_models_complete; then
        skip_stage "smokeA" "public smoke models missing under ${PUBLIC_MODELS_DIR} (get_model.sh failed?)"
    else
        local_smoke_a_stages 0 cpu ""
    fi

    # LLM smoke test on the public HF model.
    if [[ -x "${SCRIPT_DIR}/build/llm_demo" ]]; then
        run_stage "llm/${LLM_MODEL_NAME}" -- bash -c \
            "cd '${SCRIPT_DIR}/build' && ./llm_demo '${LLM_MODEL_DIR}/config.json' '${LLM_MODEL_DIR}/prompt.txt'"
    else
        skip_stage "llm/${LLM_MODEL_NAME}" "llm_demo not built"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Android-mode build
# ─────────────────────────────────────────────────────────────────────────────
ANDROID_BUILD_DIR="${SCRIPT_DIR}/project/android/build_64"

android_build() {
    log_step "building MNN for arm64-v8a"
    ensure_android_ndk
    mkdir -p "${ANDROID_BUILD_DIR}"
    pushd "${ANDROID_BUILD_DIR}" >/dev/null
    # ANDROID_EXTRA_CMAKE lets the caller append/override cmake flags, e.g.
    # to bisect a suspected backend regression:
    #   ANDROID_EXTRA_CMAKE="-DMNN_SME2=OFF -DMNN_KLEIDIAI=OFF" \
    #       ./test_ci.sh android <serial>
    local -a extra=()
    if [[ -n "${ANDROID_EXTRA_CMAKE:-}" ]]; then
        # shellcheck disable=SC2206
        extra=(${ANDROID_EXTRA_CMAKE})
    fi
    bash ../build_64.sh \
        -DMNN_BUILD_TRAIN=OFF \
        -DMNN_ARM82=true \
        -DMNN_OPENCL=true \
        -DMNN_VULKAN=true \
        -DMNN_LOW_MEMORY=true \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_BUILD_CONVERTER=ON \
        "${extra[@]}"
    popd >/dev/null
    log_ok "android build complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# Native push (replaces project/android/updateTest.sh; NPU dropped)
# ─────────────────────────────────────────────────────────────────────────────
ANDROID_BIN_LIST=(
    libllm.so
    llm_demo
    libMNN.so
    libMNN_CL.so
    libMNN_Vulkan.so
    libMNN_Express.so
    MNNV2Basic.out
    ModuleBasic.out
    testModel.out
    testModelWithDescribe.out
    backendTest.out
    timeProfile.out
    benchmark.out
    benchmarkExprModels.out
    run_test.out
    MNNConvert
)
DEVICE_SMOKE_SRC_DIR="/data/local/tmp/MNN/smoke_sources"

push_artifacts() {
    log_step "pushing artefacts to ${DEVICE_DIR}"
    ad shell "mkdir -p ${DEVICE_DIR} && rm -rf ${DEVICE_DIR}/output && mkdir -p ${DEVICE_DIR}/output" >/dev/null
    local pushed=0 missing=0
    local rel
    for rel in "${ANDROID_BIN_LIST[@]}"; do
        local src="${ANDROID_BUILD_DIR}/${rel}"
        local dst="${DEVICE_DIR}/$(basename "${rel}")"
        if [[ -e "${src}" ]]; then
            ad push "${src}" "${dst}" >/dev/null
            pushed=$((pushed + 1))
        else
            log_warn "missing artefact: ${rel} (skipped)"
            missing=$((missing + 1))
        fi
    done
    log_ok "pushed ${pushed} artefact(s); ${missing} missing"
}

# Push the cached caffe sources (.caffemodel + .prototxt) to the device,
# then drive the on-device MNNConvert binary to produce .mnn files. Avoids
# requiring a host MNNConvert.
convert_smoke_on_device() {
    log_step "converting smoke models on device (arm64 MNNConvert)"

    if ! ad shell "[ -x ${DEVICE_DIR}/MNNConvert ]" 2>/dev/null; then
        log_warn "device MNNConvert missing; cannot convert smoke models"
        return 1
    fi

    ad shell "mkdir -p ${DEVICE_SMOKE_SRC_DIR} ${DEVICE_PUBLIC_MODELS_DIR}" >/dev/null

    # Push every source file we have cached. Idempotent: skip when the size
    # matches what's already on device.
    local f local_path remote_path
    for f in "${SMOKE_SOURCES_DIR}"/*; do
        [[ -f "${f}" ]] || continue
        local_path="${f}"
        remote_path="${DEVICE_SMOKE_SRC_DIR}/$(basename "${f}")"
        local local_size remote_size
        local_size="$(stat -f%z "${local_path}" 2>/dev/null || stat -c%s "${local_path}" 2>/dev/null || echo 0)"
        remote_size="$(ad shell "stat -c%s ${remote_path} 2>/dev/null || echo 0" 2>/dev/null \
            | tr -d '[:space:]\r' || echo 0)"
        if [[ "${local_size}" != "${remote_size}" ]]; then
            ad push "${local_path}" "${remote_path}" >/dev/null
        fi
    done

    # Convert each model on device. Skip if the .mnn already exists with
    # non-zero size (idempotent on re-runs).
    local m short ok=0 fail=0
    for m in "${SMOKE_MODELS[@]}"; do
        short="${m##*/}"
        local mnn_remote="${DEVICE_PUBLIC_MODELS_DIR}/${short}"
        if ad shell "[ -s ${mnn_remote} ]" 2>/dev/null; then
            log_info "skip convert ${short} (already on device)"
            ok=$((ok + 1))
            continue
        fi
        local pair_caffe pair_proto
        pair_caffe="$(_smoke_pair_for "${m}" | sed -n '1p')"
        pair_proto="$(_smoke_pair_for "${m}" | sed -n '2p')"
        if [[ -z "${pair_caffe}" || -z "${pair_proto}" ]]; then
            log_warn "no source pairing known for ${m}; skipping"
            fail=$((fail + 1))
            continue
        fi
        local cmd="cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ./MNNConvert -f CAFFE \
            --modelFile ${DEVICE_SMOKE_SRC_DIR}/${pair_caffe} \
            --prototxt ${DEVICE_SMOKE_SRC_DIR}/${pair_proto} \
            --MNNModel ${mnn_remote} \
            --bizCode 0000 --keepInputFormat=0"
        log_info "converting ${short}"
        if ad shell "${cmd}" >/dev/null 2>&1; then
            ok=$((ok + 1))
        else
            log_warn "conversion failed for ${short}"
            fail=$((fail + 1))
        fi
    done
    log_ok "on-device conversion complete (${ok} ok, ${fail} failed)"
    [[ ${fail} -eq 0 ]]
}

push_llm_model() {
    log_step "pushing LLM model ${LLM_MODEL_NAME} to device"
    local remote="${DEVICE_MODEL_DIR}/${LLM_MODEL_NAME}"
    ad shell "mkdir -p ${DEVICE_MODEL_DIR}" >/dev/null

    local local_count remote_count
    local_count="$(find "${LLM_MODEL_DIR}" -type f | wc -l | tr -d '[:space:]')"
    remote_count="$(ad shell "find ${remote} -type f 2>/dev/null | wc -l" 2>/dev/null \
        | tr -d '[:space:]\r' || echo 0)"
    if [[ "${remote_count}" == "${local_count}" && "${local_count}" -gt 0 ]]; then
        log_info "skip push (already on device, ${local_count} files)"
        return 0
    fi
    ad shell "rm -rf ${remote}" >/dev/null
    ad push "${LLM_MODEL_DIR}" "${remote}" >/dev/null
    log_ok "LLM model pushed (${local_count} files)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Android on-device stages
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: the script-wide IFS is set to $'\n\t', so unquoted "$*" would join
# args with newlines. Embedded in an `adb shell` command string those
# newlines split into separate remote commands, breaking the run with rc=127
# on the trailing tokens. Force a local space-only IFS for the join.
_remote_run_test() {
    local IFS=' '
    ad shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ./run_test.out $*"
}

_remote_v2basic() {
    local IFS=' '
    ad shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ./MNNV2Basic.out $*"
}

_remote_backendtest() {
    local IFS=' '
    ad shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ./backendTest.out $*"
}

_remote_benchmark() {
    local IFS=' '
    ad shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ./benchmark.out $*"
}

# Per-backend benchmark over the public smoke model set.
# benchmark.out args: <models_folder> <loop> <warmup> <forwardtype> <numberThread|gpuMode> <precision>
# argv[5] (numberThread) is reused as gpuMode for OpenCL/Vulkan, same as run_test.out.
android_benchmarks() {
    local loop=10 warmup=2
    if ! ad shell "[ -d ${DEVICE_PUBLIC_MODELS_DIR} ]" 2>/dev/null; then
        skip_stage "bench" "public_models dir absent on device"
        return 0
    fi
    run_stage "bench/cpu"    -- _remote_benchmark "${DEVICE_PUBLIC_MODELS_DIR}" "${loop}" "${warmup}" 0 4 1
    run_stage "bench/opencl" -- _remote_benchmark "${DEVICE_PUBLIC_MODELS_DIR}" "${loop}" "${warmup}" 3 132 1
    run_stage "bench/vulkan" -- _remote_benchmark "${DEVICE_PUBLIC_MODELS_DIR}" "${loop}" "${warmup}" 7 4 1
}

# Stage A on-device: forward smoke per (backend × model).
android_smoke_a_stages() {
    local backend_id="$1" backend_name="$2"
    local m short remote
    for m in "${SMOKE_MODELS[@]}"; do
        short="${m##*/}"
        remote="${DEVICE_PUBLIC_MODELS_DIR}/${short}"
        run_stage "smokeA/${backend_name}/${short}" -- \
            _remote_v2basic "${remote}" 1 0 "${backend_id}"
    done
}

# Stage B on-device: CPU-vs-backend numeric comparison.
android_smoke_b_stages() {
    local backend_id="$1" backend_name="$2" tolerance="${3:-0.05}"
    local m short remote
    for m in "${SMOKE_MODELS[@]}"; do
        short="${m##*/}"
        remote="${DEVICE_PUBLIC_MODELS_DIR}/${short}"
        run_stage "smokeB/${backend_name}/${short}" -- \
            _remote_backendtest "${remote}" "${backend_id}" "${tolerance}"
    done
}

# Unit-test matrix (mirrors test.sh:android_unit_test 64 0 plus Vulkan).
#
# IMPORTANT: argv[4] of run_test.out has DIFFERENT semantics per backend.
#   - CPU (type 0)        → thread count (we use 1 or 4).
#   - OpenCL (type 3)     → gpuMode bitmask (MNN_GPU_TUNING_* | MNN_GPU_MEMORY_*).
#                           132 = TUNING_WIDE (4) | MEMORY_IMAGE (128) — the
#                           recommended OpenCL default. Plain 4 leaves memory
#                           mode unset, which on some drivers segfaults.
#   - Vulkan (type 7)     → gpuMode bitmask, but only TUNING_* bits are valid.
#                           4 = TUNING_WIDE.
android_unit_tests() {
    run_stage "unit/cpu/all"            -- _remote_run_test all 0 0 1 64 0
    run_stage "unit/cpu/op-mt"          -- _remote_run_test op 0 0 4 multi64 0
    run_stage "unit/cpu/op-fp16-conv"   -- _remote_run_test op/convolution 0 2 4 fp16multi64 0
    run_stage "unit/cpu/op-fp16-col2im" -- _remote_run_test op/col2im 0 2 4 fp16col2im64 0
    run_stage "unit/cpu/op-fp16-roi"    -- _remote_run_test op/R 0 2 4 fp16roipooling64 0
    run_stage "unit/opencl/op"          -- _remote_run_test op 3 1 132 64 0
    run_stage "unit/vulkan/op"          -- _remote_run_test op 7 1 4 64 0
}

# Low-memory armv8 matrix (mirrors test.sh:android_unit_test_low_memory_armv8).
android_low_memory_tests() {
    local tag=64
    run_stage "lowmem/dyn-p1-t1"  -- _remote_run_test op/lowMemory 0 1 1 "${tag}" 2
    run_stage "lowmem/dyn-p2-t1"  -- _remote_run_test op/lowMemory 0 2 1 "${tag}" 2
    run_stage "lowmem/dyn-p1-t4"  -- _remote_run_test op/lowMemory 0 1 4 "${tag}" 2
    run_stage "lowmem/dyn-p2-t4"  -- _remote_run_test op/lowMemory 0 2 4 "${tag}" 2
    run_stage "lowmem/wdeq-p1"    -- _remote_run_test op/lowMemory 0 1 1 "${tag}"
    run_stage "lowmem/wdeq-p2"    -- _remote_run_test op/lowMemory 0 2 1 "${tag}"
    run_stage "lowmem/i8i4-d1-p2" -- _remote_run_test op/convolution/weighti8i4conv2d 0 2 4 "${tag}" 2 1
    run_stage "lowmem/i8i4-d1-p1" -- _remote_run_test op/convolution/weighti8i4conv2d 0 1 4 "${tag}" 2 1
    run_stage "lowmem/i8i4-d2-p2" -- _remote_run_test op/convolution/weighti8i4conv2d 0 2 4 "${tag}" 2 2
    run_stage "lowmem/i8i4-d2-p1" -- _remote_run_test op/convolution/weighti8i4conv2d 0 1 4 "${tag}" 2 2
}

android_llm_test() {
    local remote="${DEVICE_MODEL_DIR}/${LLM_MODEL_NAME}"
    if ad shell "[ -f ${remote}/config.json ]" 2>/dev/null; then
        run_stage "llm/${LLM_MODEL_NAME}" -- ad shell \
            "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ./llm_demo ${remote}/config.json ${remote}/prompt.txt"
    else
        skip_stage "llm/${LLM_MODEL_NAME}" "model dir absent on device"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Drivers
# ─────────────────────────────────────────────────────────────────────────────
drive_local() {
    log_info "mode=local script_dir=${SCRIPT_DIR}"
    provision_llm_model
    if [[ ! -x "${SCRIPT_DIR}/build/run_test.out" ]]; then
        local_build
    else
        log_info "reusing existing build/ (delete it to force a rebuild)"
    fi
    # Try to populate the public smoke set after the build (MNNConvert is
    # produced by the build). Stages downstream gate on availability.
    provision_public_models || true
    local_run_stages
}

drive_android() {
    log_info "mode=android device=${DEVICE} script_dir=${SCRIPT_DIR}"
    detect_adb
    ensure_adbk_session
    verify_device_online
    provision_llm_model
    # Cache caffe sources on host (small download). The conversion itself
    # runs on device with the arm64 MNNConvert we build below.
    local smoke_ok=1
    provision_smoke_sources || smoke_ok=0
    android_build
    push_artifacts
    push_llm_model
    if [[ ${smoke_ok} -eq 1 ]]; then
        convert_smoke_on_device || smoke_ok=0
    fi
    android_unit_tests
    android_low_memory_tests
    if [[ ${smoke_ok} -eq 1 ]]; then
        android_smoke_a_stages 0 cpu
        android_smoke_a_stages 3 opencl
        android_smoke_a_stages 7 vulkan
        android_smoke_b_stages 3 opencl 0.05
        android_smoke_b_stages 7 vulkan 0.05
        android_benchmarks
    else
        skip_stage "smokeA" "smoke source download or on-device conversion failed"
        skip_stage "smokeB" "smoke source download or on-device conversion failed"
        skip_stage "bench"  "smoke source download or on-device conversion failed"
    fi
    android_llm_test
}

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
main() {
    parse_args "$@"
    case "${MODE}" in
        local)   drive_local ;;
        android) drive_android ;;
    esac
}

main "$@"
