#!/usr/bin/env bash
# test_ci.sh - Focused Android-arm64 + local-host CI driver for MNN.
#
# Subcommands:
#   ./test_ci.sh local                  Run host-side regression: build + the
#                                       built-in unit-test suite (CPU) and the
#                                       LLM smoke test.
#   ./test_ci.sh android <serial>       Build for arm64-v8a, push artefacts and
#                                       the LLM model to /data/local/tmp, then
#                                       run the on-device matrix.
#
# Environment:
#   ANDROID_NDK         (required for android mode) NDK root. Falls back to
#                       $HOME/android-ndk-r21 when unset.
#   LLM_MODEL_DIR       Path to an existing MNN-format LLM model on disk. When
#                       set, that directory is used as-is and NO download is
#                       attempted. Defaults to models/<repo-basename>/.
#   LLM_MODEL_REPO      Model repo id used for the LLM smoke test.
#                       Defaults to taobao-mnn/Qwen2.5-0.5B-Instruct-MNN.
#   LLM_MODEL_SOURCE    Download source when LLM_MODEL_DIR is not provided:
#                       'huggingface' (default) or 'modelscope' (CDN reachable
#                       from mainland China, where huggingface.co is blocked).
#   LLM_MODEL_URL_BASE  Override the resolve URL prefix outright. Wins over
#                       LLM_MODEL_SOURCE. Defaults to the source's resolve URL.
#
# Notes:
#   * Replaces project/android/updateTest.sh natively (no shell-out).
#   * Prefers `adbk` over `adb` and manages session lifecycle (--create-session
#     on entry, --delete-session on EXIT, including failure paths).
#   * Mirrors every OpenCL (backend=3) probe with a Vulkan (backend=7) probe;
#     skipped (not failed) when the backend library is absent.
#   * LLM model provisioning is lazy: the download (or LLM_MODEL_DIR check) is
#     deferred until the llm stage actually runs, so unit / smoke / bench
#     stages proceed even with no network. A provisioning failure skips the
#     llm stage rather than aborting the run.

set -euo pipefail
IFS=$'\n\t'
umask 022

# ─────────────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
# Declarative stage list — see test_stages.json for the schema and rationale
# behind each entry. Editing the JSON is the supported way to add, drop, or
# reconfigure unit/lowmem stages. Smoke and bench stages stay in shell since
# they iterate over external model files rather than a fixed parameter grid.
STAGES_JSON_FILE="${SCRIPT_DIR}/test_stages.json"
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

# Capture whether the caller pinned LLM_MODEL_REPO before applying the default,
# so the ModelScope org remap below only rewrites the project's own default.
if [[ -n "${LLM_MODEL_REPO:-}" ]]; then LLM_MODEL_REPO_USER_SET=1; else LLM_MODEL_REPO_USER_SET=0; fi
LLM_MODEL_REPO_DEFAULT="$(python3 -c "import json; print(json.load(open('${STAGES_JSON_FILE}')).get('llm', {}).get('model_repo', 'taobao-mnn/Qwen2.5-0.5B-Instruct-MNN'))" 2>/dev/null || true)"
LLM_MODEL_REPO="${LLM_MODEL_REPO:-${LLM_MODEL_REPO_DEFAULT:-taobao-mnn/Qwen2.5-0.5B-Instruct-MNN}}"

# Resolve the download URL prefix. An explicit LLM_MODEL_URL_BASE always wins;
# otherwise it is derived from LLM_MODEL_SOURCE. HuggingFace serves under
# resolve/main, ModelScope under resolve/master. The MNN team mirrors its
# HuggingFace 'taobao-mnn/*' models under the 'MNN/*' org on ModelScope, so the
# built-in default's org is remapped for ModelScope (an explicitly-set
# LLM_MODEL_REPO is left verbatim). NB: this runs before the log_* helpers are
# defined, so errors print with a plain echo.
LLM_MODEL_SOURCE="${LLM_MODEL_SOURCE:-huggingface}"
if [[ -z "${LLM_MODEL_URL_BASE:-}" ]]; then
    case "${LLM_MODEL_SOURCE}" in
        huggingface|hf)
            LLM_MODEL_URL_BASE="https://huggingface.co/${LLM_MODEL_REPO}/resolve/main" ;;
        modelscope|ms)
            LLM_MODEL_MS_REPO="${LLM_MODEL_REPO}"
            if [[ ${LLM_MODEL_REPO_USER_SET} -eq 0 && "${LLM_MODEL_MS_REPO}" == taobao-mnn/* ]]; then
                LLM_MODEL_MS_REPO="MNN/${LLM_MODEL_MS_REPO#taobao-mnn/}"
            fi
            LLM_MODEL_URL_BASE="https://modelscope.cn/models/${LLM_MODEL_MS_REPO}/resolve/master" ;;
        *)
            echo "ERROR: unknown LLM_MODEL_SOURCE='${LLM_MODEL_SOURCE}' (want: huggingface | modelscope)" >&2
            exit 2 ;;
    esac
fi

# LLM_MODEL_DIR may be pre-set by the caller to point at an existing on-disk
# MNN-format model, in which case no download is ever attempted. When it is
# left unset we fall back to the per-repo cache under models/ and may download.
if [[ -n "${LLM_MODEL_DIR:-}" ]]; then
    LLM_MODEL_DIR_PROVIDED=1
    LLM_MODEL_NAME="$(basename "${LLM_MODEL_DIR}")"
else
    LLM_MODEL_DIR_PROVIDED=0
    LLM_MODEL_NAME="$(basename "${LLM_MODEL_REPO}")"
    LLM_MODEL_DIR="${MODELS_DIR}/${LLM_MODEL_NAME}"
fi

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
FILTER="all"
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
  android <device_serial> [filter]
                           Build arm64-v8a, push and test on device.
                           filter (optional, default 'all'):
                             all      — every stage (cpu+opencl+vulkan+lowmem+llm+smoke).
                             cpu      — CPU unit + lowmem + llm only.
                             opencl   — OpenCL unit (IMAGE + BUFFER) + opencl smoke.
                             opencl-image  — only OpenCL IMAGE-mem stage.
                             opencl-buffer — only OpenCL BUFFER-mem stage.
                             vulkan   — Vulkan unit + vulkan smoke.
                             gpu      — opencl + vulkan unit + smoke.
                             unit     — all unit/op stages, no lowmem/smoke/llm.
                             lowmem   — only lowmem matrix.
                             android-ci
                                      — bench + smoke (cpu/opencl/vulkan) + llm only;
                                        skips unit-test and lowmem stages.

Environment:
  ANDROID_NDK              NDK root (android mode). Defaults to
                           \$HOME/android-ndk-r21 when unset.
  LLM_MODEL_DIR            Use an existing on-disk MNN LLM model; no download.
                           Default: models/${LLM_MODEL_NAME}.
  LLM_MODEL_REPO           Model repo id for the LLM smoke test.
                           Default: ${LLM_MODEL_REPO}.
  LLM_MODEL_SOURCE         Download source: huggingface (default) | modelscope.
  LLM_MODEL_URL_BASE       Override resolve URL prefix (wins over _SOURCE).

  The LLM model is fetched lazily — only when the llm stage runs — so the
  unit / smoke / bench stages proceed even with no network.

Examples:
  $0 local
  $0 android emulator-5554
  $0 android R5CY71BJJ9D opencl-buffer
  $0 android R5CY71BJJ9D cpu
  $0 android R5CY71BJJ9D android-ci      # bench + smoke + llm only
  LLM_MODEL_DIR=/path/to/MNN-model $0 local             # use a local model, no download
  LLM_MODEL_SOURCE=modelscope $0 android R5CY71BJJ9D    # fetch from ModelScope
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
            if [[ $# -ge 1 && -n "${1:-}" ]]; then
                FILTER="$1"; shift
            fi
            case "${FILTER}" in
                all|cpu|opencl|opencl-image|opencl-buffer|vulkan|gpu|unit|lowmem|android-ci) ;;
                *) log_err "unknown filter: ${FILTER}"; usage; exit 2 ;;
            esac
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
# LLM model provisioning (HuggingFace / ModelScope / local dir)
# ─────────────────────────────────────────────────────────────────────────────
# Cache layout: ${MODELS_DIR}/<repo-basename>/{config.json,llm.mnn,...}
# Plus a generated prompt.txt so llm_demo has something to consume.
#
# Provisioning is lazy — invoked only once a run reaches the llm stage — and
# returns non-zero on failure (instead of aborting the whole script) so the
# caller can skip just the llm stage. When LLM_MODEL_DIR is caller-supplied the
# directory is used as-is and nothing is downloaded.
llm_model_complete() {
    local f
    for f in "${LLM_MODEL_FILES[@]}"; do
        [[ -s "${LLM_MODEL_DIR}/${f}" ]] || return 1
    done
    return 0
}

provision_llm_model() {
    log_step "provisioning LLM model: ${LLM_MODEL_NAME}"
    if llm_model_complete; then
        log_ok "LLM model present at ${LLM_MODEL_DIR}"
    elif [[ ${LLM_MODEL_DIR_PROVIDED} -eq 1 ]]; then
        log_err "LLM_MODEL_DIR=${LLM_MODEL_DIR} is missing required files; not downloading (caller-supplied path)"
        return 1
    else
        log_info "downloading from ${LLM_MODEL_SOURCE} (${LLM_MODEL_URL_BASE})"
        mkdir -p "${LLM_MODEL_DIR}"
        local f url tmp dst
        for f in "${LLM_MODEL_FILES[@]}"; do
            dst="${LLM_MODEL_DIR}/${f}"
            if [[ -s "${dst}" ]]; then
                continue
            fi
            url="${LLM_MODEL_URL_BASE}/${f}"
            tmp="${dst}.part"
            log_info "fetching ${f}"
            if ! curl -fL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"; then
                log_err "failed to download ${url}"
                rm -f "${tmp}"
                return 1
            fi
            mv "${tmp}" "${dst}"
        done
        log_ok "LLM model staged at ${LLM_MODEL_DIR}"
    fi
    # Ensure a small prompt exists for the smoke test. Best-effort: a read-only
    # caller-supplied model dir is fine as long as it already ships a prompt.
    local prompt_path="${LLM_MODEL_DIR}/prompt.txt"
    if [[ ! -s "${prompt_path}" ]]; then
        if ! printf 'Hello, who are you?\n' > "${prompt_path}" 2>/dev/null; then
            log_err "no prompt.txt in ${LLM_MODEL_DIR} and the directory is not writable"
            return 1
        fi
    fi
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Public smoke models (resource/model/) — Caffe MobileNet/SqueezeNet pairs
# fetched from upstream GitHub repos and converted with our local MNNConvert.
# We deliberately do NOT use tools/script/get_model.sh: it pulls extra TFLite
# tarballs from URLs that frequently 404 / 503, producing noisy gzip + parser
# errors (and an MNNConvert SIGSEGV on a corrupt .tflite payload) even when
# the four .mnn files we actually need have already been converted.
# ─────────────────────────────────────────────────────────────────────────────
public_models_complete() {
    local m
    for m in "${SMOKE_MODELS[@]}"; do
        [[ -s "${PUBLIC_MODELS_DIR}/${m}" ]] || return 1
    done
    return 0
}

# Convert a caffemodel/prototxt pair to ${PUBLIC_MODELS_DIR}/<relpath>.
# Args: 1=caffemodel basename, 2=prototxt basename, 3=output relpath
_local_convert_caffe_pair() {
    local caffemodel="$1" prototxt="$2" relpath="$3"
    local src_caffe="${SMOKE_SOURCES_DIR}/${caffemodel}"
    local src_proto="${SMOKE_SOURCES_DIR}/${prototxt}"
    local dst="${PUBLIC_MODELS_DIR}/${relpath}"
    if [[ -s "${dst}" ]]; then
        return 0
    fi
    if [[ ! -s "${src_caffe}" || ! -s "${src_proto}" ]]; then
        log_err "missing source: ${src_caffe} or ${src_proto}"
        return 1
    fi
    mkdir -p "$(dirname "${dst}")"
    log_info "converting ${relpath}"
    "${SCRIPT_DIR}/build/MNNConvert" \
        -f CAFFE \
        --modelFile "${src_caffe}" \
        --prototxt "${src_proto}" \
        --MNNModel "${dst}" \
        --bizCode 0000 \
        --keepInputFormat=0 >/dev/null
}

# Returns 0 if the public smoke set is ready, 1 otherwise. Caller decides
# whether to skip downstream stages.
provision_public_models() {
    log_step "provisioning public smoke models"
    if public_models_complete; then
        log_ok "public model cache hit at ${PUBLIC_MODELS_DIR}"
        return 0
    fi
    if [[ ! -x "${SCRIPT_DIR}/build/MNNConvert" ]]; then
        log_warn "build/MNNConvert missing — smoke stages will skip"
        return 1
    fi
    if ! provision_smoke_sources; then
        log_warn "smoke source download failed — smoke stages will skip"
        return 1
    fi
    local rc=0
    _local_convert_caffe_pair mobilenet_v1.caffemodel mobilenet_v1.prototxt \
        MobileNet/v1/mobilenet_v1.caffe.mnn || rc=1
    _local_convert_caffe_pair mobilenet_v2.caffemodel mobilenet_v2.prototxt \
        MobileNet/v2/mobilenet_v2.caffe.mnn || rc=1
    _local_convert_caffe_pair squeezenet_v1.0.caffemodel squeezenet_v1.0.prototxt \
        SqueezeNet/v1.0/squeezenet_v1.0.caffe.mnn || rc=1
    _local_convert_caffe_pair squeezenet_v1.1.caffemodel squeezenet_v1.1.prototxt \
        SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn || rc=1
    if [[ ${rc} -ne 0 ]]; then
        log_warn "one or more conversions failed — smoke stages will skip"
        return 1
    fi
    if ! public_models_complete; then
        log_warn "expected .mnn files missing after conversion — smoke stages will skip"
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
    # On macOS, CMake fails to auto-pick the SDK when CMAKE_CXX_COMPILER
    # resolves to /usr/bin/c++ (the Apple shim), leaving CMAKE_OSX_SYSROOT
    # empty and the OBJECT-library targets (e.g. MNNARM64) unable to find
    # <vector>/<map>. Pass the active SDK path explicitly.
    #
    # Additionally, partially-upgraded Command Line Tools installs leave a
    # stale /Library/Developer/CommandLineTools/usr/include/c++/v1 with only
    # a few legacy files. clang's internal-isystem hits that dir first and
    # stops searching, so the SDK's complete libc++ headers are never seen.
    # Prepend the SDK's libc++ via CPLUS_INCLUDE_PATH to force a hit.
    local -a platform_args=()
    if [[ "$(uname -s)" == "Darwin" ]]; then
        local sdkpath
        sdkpath="$(xcrun --show-sdk-path 2>/dev/null || true)"
        if [[ -n "${sdkpath}" ]]; then
            platform_args+=("-DCMAKE_OSX_SYSROOT=${sdkpath}")
            local sdk_libcxx="${sdkpath}/usr/include/c++/v1"
            if [[ -d "${sdk_libcxx}" && ! -f "/Library/Developer/CommandLineTools/usr/include/c++/v1/vector" ]]; then
                export CPLUS_INCLUDE_PATH="${sdk_libcxx}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
                log_warn "host CommandLineTools libc++ is incomplete; prepending ${sdk_libcxx} to CPLUS_INCLUDE_PATH"
            fi
        fi
    fi
    # Local mode is CPU-only: host GPU drivers are usually unavailable or
    # unreliable on dev machines, so we omit OpenCL/Vulkan to keep the
    # build fast and the test surface meaningful.
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_TEST=ON \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON \
        -DMNN_BUILD_CONVERTER=ON \
        "${platform_args[@]}"
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

# ─────────────────────────────────────────────────────────────────────────────
# Local-mode runners — invoked by the JSON dispatch table; one wrapper per
# binary, named symmetrically with the remote/_remote_* counterparts.
# ─────────────────────────────────────────────────────────────────────────────
_local_run_test() {
    (cd "${SCRIPT_DIR}/build" && ./run_test.out "$@");
}
_local_v2basic() {
    (cd "${SCRIPT_DIR}/build" && ./MNNV2Basic.out "$@")
}
_local_backendtest() {
    (cd "${SCRIPT_DIR}/build" && ./backendTest.out "$@")
}

local_run_stages() {
    # Local mode is CPU-only by design — see local_build() comment.
    # Unit and smoke stages flow through the JSON config (section: "local").
    _run_json_stages local _local_run_test

    # Stage A on CPU: forward-smoke per public model. Stage B (CPU-vs-
    # backend) is meaningless without a GPU build, so the local section of
    # the JSON only declares smoke_a_stages.
    if [[ ! -x "${SCRIPT_DIR}/build/MNNV2Basic.out" ]]; then
        skip_stage "smokeA" "build/MNNV2Basic.out not built (check MNN_BUILD_TOOLS)"
    elif ! public_models_complete; then
        skip_stage "smokeA" "public smoke models missing under ${PUBLIC_MODELS_DIR} (get_model.sh failed?)"
    else
        _run_json_model_stages smokeA local "${PUBLIC_MODELS_DIR}"
    fi

    # LLM smoke test. Model provisioning is lazy: it happens here, after the
    # unit + smoke stages have already run, so those proceed even with no
    # network. A failed provision skips just this stage.
    if [[ ! -x "${SCRIPT_DIR}/build/llm_demo" ]]; then
        skip_stage "llm/${LLM_MODEL_NAME}" "llm_demo not built"
    elif ! provision_llm_model; then
        skip_stage "llm/${LLM_MODEL_NAME}" "model unavailable (see provisioning log above)"
    else
        run_stage "llm/${LLM_MODEL_NAME}" -- bash -c \
            "cd '${SCRIPT_DIR}/build' && ./llm_demo '${LLM_MODEL_DIR}/config.json' '${LLM_MODEL_DIR}/prompt.txt'"
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
    # build_64.sh hard-codes `make -j4`; respin with all cores so subsequent
    # incremental work uses full parallelism.
    make -j"$(_host_jobs)"
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
    # MNNConvert dynamically links against libMNNConvertDeps.so which is
    # built under tools/converter/. Without this push the on-device caffe->
    # mnn conversion fails ("library libMNNConvertDeps.so not found"),
    # causing smokeA / smokeB / bench to be skipped.
    tools/converter/libMNNConvertDeps.so
)
DEVICE_SMOKE_SRC_DIR="/data/local/tmp/MNN/smoke_sources"

push_artifacts() {
    log_step "pushing artefacts to ${DEVICE_DIR}"
    ad shell "mkdir -p ${DEVICE_DIR} && rm -rf ${DEVICE_DIR}/output && mkdir -p ${DEVICE_DIR}/output" >/dev/null
    local pushed=0 missing=0
    local rel src dst
    for rel in "${ANDROID_BIN_LIST[@]}"; do
        src="${ANDROID_BUILD_DIR}/${rel}"
        dst="${DEVICE_DIR}/$(basename "${rel}")"
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
        # Flat layout on device: benchmark.out's findModelFiles() does a
        # non-recursive readdir() of its dir argument and trips over any
        # subdirectories, so all .mnn files must sit at the same level.
        # _emit_json_smoke_stages is section-aware and uses basename for
        # android paths to match.
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
    # MNN_TEST_SKIP is honored by MNNTestSuite::run() to omit named tests
    # from the in-process loop. Used by the OpenCL BUFFER stage to drop
    # ops that hit Mali driver bugs in BUFFER-mode loop kernels.
    local skip_env=""
    if [[ -n "${MNN_TEST_SKIP:-}" ]]; then
        skip_env="export MNN_TEST_SKIP='${MNN_TEST_SKIP}' && "
    fi
    ad shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ${skip_env}./run_test.out $*"
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

# ─────────────────────────────────────────────────────────────────────────────
# JSON dispatch — every stage in test_stages.json is materialised to a row
# (name|filter|binary|argv) and dispatched through one of two binary maps.
# ─────────────────────────────────────────────────────────────────────────────

# Map JSON 'binary' string -> shell function that invokes the on-device tool.
_remote_for_binary() {
    case "$1" in
        run_test)    echo _remote_run_test ;;
        v2basic)     echo _remote_v2basic ;;
        backendtest) echo _remote_backendtest ;;
        benchmark)   echo _remote_benchmark ;;
        *)           return 1 ;;
    esac
}

# Map JSON 'binary' string -> shell function that invokes the host tool.
# Local mode has no benchmark.out (not built); benchmark stages live in
# JSON's android section only, so this map intentionally omits it.
_local_for_binary() {
    case "$1" in
        run_test)    echo _local_run_test ;;
        v2basic)     echo _local_v2basic ;;
        backendtest) echo _local_backendtest ;;
        *)           return 1 ;;
    esac
}

# Materialize per-model smoke stage rows. One TAB-separated row per
# (smoke_a or smoke_b stage) × (model). Columns:
#   1: stage name (with model short-name suffix)
#   2: filter tag
#   3: shell function name (_remote_v2basic / _remote_backendtest)
#   4: shell-quoted argv (with {model} / {models_dir} substituted)
_emit_json_smoke_stages() {
    local section_root="$1"      # "android" or "local"
    local stages_key="$2"        # "smoke_a_stages" or "smoke_b_stages"
    local models_dir="$3"        # device or host public_models dir
    SMOKE_SECTION_ROOT="${section_root}" SMOKE_STAGES_KEY="${stages_key}" \
    SMOKE_MODELS_DIR="${models_dir}" \
        python3 - "${STAGES_JSON_FILE}" <<'PY'
import json, os, shlex, sys
root       = os.environ["SMOKE_SECTION_ROOT"]
key        = os.environ["SMOKE_STAGES_KEY"]
models_dir = os.environ["SMOKE_MODELS_DIR"]
with open(sys.argv[1]) as f:
    data = json.load(f)
node   = data.get(root, {})
models = node.get("smoke_models", []) or data.get("android", {}).get("smoke_models", [])
stages = node.get(key, [])
for m in models:
    short = m.split("/")[-1]
    # Local mode keeps host's nested layout (resource/model/MobileNet/v1/...).
    # Android mode keeps a flat layout under DEVICE_PUBLIC_MODELS_DIR because
    # benchmark.out's findModelFiles() is non-recursive (see benchmark.cpp:54).
    device_path = f"{models_dir}/{m}" if root == "local" else f"{models_dir}/{short}"
    for st in stages:
        argv = [
            a.replace("{model}", device_path).replace("{models_dir}", models_dir)
            for a in st["args"]
        ]
        name = f"{st['name']}/{short}"
        quoted = " ".join(shlex.quote(a) for a in argv)
        print(f"{name}|{st['filter']}|{st['binary']}|{quoted}")
PY
}

# Materialize bench stage rows (one per backend, models_dir substituted).
_emit_json_bench_stages() {
    local section_root="$1"      # "android"
    local models_dir="$2"
    BENCH_SECTION_ROOT="${section_root}" BENCH_MODELS_DIR="${models_dir}" \
        python3 - "${STAGES_JSON_FILE}" <<'PY'
import json, os, shlex, sys
root       = os.environ["BENCH_SECTION_ROOT"]
models_dir = os.environ["BENCH_MODELS_DIR"]
with open(sys.argv[1]) as f:
    data = json.load(f)
for st in data.get(root, {}).get("bench_stages", []):
    argv = [a.replace("{models_dir}", models_dir) for a in st["args"]]
    print(f"{st['name']}|{st['filter']}|{st['binary']}|{' '.join(shlex.quote(a) for a in argv)}")
PY
}

# Driver for smoke A / smoke B / bench JSON-defined stages. Buffers rows
# into an array first so adb's stdin can't close the read loop early
# (the classic `while read … | tee | adb` gotcha).
_run_json_model_stages() {
    local kind="$1"                # smokeA | smokeB | bench
    local section_root="${2:-android}"
    local models_dir="${3:-${DEVICE_PUBLIC_MODELS_DIR}}"
    local -a rows=()
    local row
    case "${kind}" in
        smokeA)
            while IFS= read -r row; do rows+=("${row}"); done \
                < <(_emit_json_smoke_stages "${section_root}" smoke_a_stages "${models_dir}") ;;
        smokeB)
            while IFS= read -r row; do rows+=("${row}"); done \
                < <(_emit_json_smoke_stages "${section_root}" smoke_b_stages "${models_dir}") ;;
        bench)
            while IFS= read -r row; do rows+=("${row}"); done \
                < <(_emit_json_bench_stages "${section_root}" "${models_dir}") ;;
        *) return 1 ;;
    esac
    local name filt binary argv runner
    for row in "${rows[@]}"; do
        IFS='|' read -r name filt binary argv <<<"${row}"
        [[ -z "${name}" ]] && continue
        if ! _filter_runs "${filt}"; then
            continue
        fi
        if [[ "${section_root}" == "local" ]]; then
            runner="$(_local_for_binary "${binary}")" || continue
        else
            runner="$(_remote_for_binary "${binary}")" || continue
        fi
        # ${argv} is shell-quoted by Python's shlex.quote, so eval-driven
        # word-splitting is required to honour the original token
        # boundaries. \$name defers expansion to eval's parse pass.
        eval "run_stage \"\$name\" -- ${runner} ${argv}"
    done
}

android_benchmarks() {
    if ! ad shell "[ -d ${DEVICE_PUBLIC_MODELS_DIR} ]" 2>/dev/null; then
        skip_stage "bench" "public_models dir absent on device"
        return 0
    fi
    _run_json_model_stages bench android "${DEVICE_PUBLIC_MODELS_DIR}"
}

# Backwards-compatible wrappers — drive_android still calls these.
android_smoke_a_stages() {
    _run_json_model_stages smokeA android "${DEVICE_PUBLIC_MODELS_DIR}"
}

android_smoke_b_stages() {
    _run_json_model_stages smokeB android "${DEVICE_PUBLIC_MODELS_DIR}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Filter logic — translates the user-supplied --filter (FILTER global) into a
# per-stage admission decision based on the row's tag.
#
# Unit-test matrix (mirrors test.sh:android_unit_test 64 0 plus Vulkan).
#
# IMPORTANT: argv[4] of run_test.out has DIFFERENT semantics per backend.
#   - CPU (type 0)        -> thread count (we use 1 or 4).
#   - OpenCL (type 3)     -> gpuMode bitmask (MNN_GPU_TUNING_* | MNN_GPU_MEMORY_*).
#                            132 = TUNING_WIDE (4) | MEMORY_IMAGE (128) — the
#                            recommended OpenCL default. Plain 4 leaves memory
#                            mode unset, which on some drivers segfaults.
#   - Vulkan (type 7)     -> gpuMode bitmask, but only TUNING_* bits are valid.
#                            4 = TUNING_WIDE.
# ─────────────────────────────────────────────────────────────────────────────
_filter_runs() {
    # arg1 = backend tag (cpu|opencl-image|opencl-buffer|vulkan|lowmem|smoke|llm)
    # returns 0 (truthy in bash if-statements) if the stage should run.
    local tag="$1"
    case "${FILTER}" in
        all)            return 0 ;;
        cpu)            [[ "${tag}" == "cpu" || "${tag}" == "lowmem" || "${tag}" == "llm" ]] ;;
        opencl)         [[ "${tag}" == "opencl-image" || "${tag}" == "opencl-buffer" || "${tag}" == "smoke-opencl" ]] ;;
        opencl-image)   [[ "${tag}" == "opencl-image" ]] ;;
        opencl-buffer)  [[ "${tag}" == "opencl-buffer" ]] ;;
        vulkan)         [[ "${tag}" == "vulkan" || "${tag}" == "smoke-vulkan" ]] ;;
        gpu)            [[ "${tag}" == "opencl-image" || "${tag}" == "opencl-buffer" || "${tag}" == "vulkan" || "${tag}" == "smoke-opencl" || "${tag}" == "smoke-vulkan" ]] ;;
        unit)           [[ "${tag}" == "cpu" || "${tag}" == "opencl-image" || "${tag}" == "opencl-buffer" || "${tag}" == "vulkan" ]] ;;
        lowmem)         [[ "${tag}" == "lowmem" ]] ;;
        # android-ci: bench (filter=cpu inside bench_stages) + smokeA/B
        # (cpu, smoke-opencl, smoke-vulkan) + llm. Unit/lowmem are skipped at
        # the drive_android level by gating android_tests, so accepting "cpu"
        # here only lets bench and smokeA/cpu rows through.
        android-ci)     [[ "${tag}" == "cpu" || "${tag}" == "smoke-opencl" || "${tag}" == "smoke-vulkan" || "${tag}" == "llm" ]] ;;
    esac
}

# Materialize each stage from a $section.stages array in test_stages.json
# into one shell-quoted line per stage. Columns are pipe-delimited because
# `read` collapses runs of whitespace IFS chars (tab/space/newline) into a
# single separator — using '|' (non-whitespace) keeps empty fields like
# 'skip' addressable instead of losing them.
#   1: name
#   2: filter tag
#   3: skip-list (comma-joined; empty if none)
#   4: positional argv for the chosen binary, already shell-quoted
_emit_json_stages() {
    local section="${1:-android}"
    JSON_SECTION="${section}" python3 - "${STAGES_JSON_FILE}" <<'PY'
import json, os, shlex, sys
section = os.environ["JSON_SECTION"]
with open(sys.argv[1]) as f:
    data = json.load(f)
node = data.get(section, {})
for st in node.get("stages", []):
    name   = st["name"]
    filt   = st["filter"]
    skip   = ",".join(st.get("skip") or [])
    argv = [
        str(st["prefix"]),
        str(st["type"]),
        str(st["precision"]),
        str(st["threadOrGpuMode"]),
        str(st["tag"]),
    ]
    if st.get("memory") is not None:
        argv.append(str(st["memory"]))
        if st.get("dynamicOption") is not None:
            argv.append(str(st["dynamicOption"]))
            if st.get("kleidiAi") is not None:
                argv.append(str(st["kleidiAi"]))
    quoted = " ".join(shlex.quote(a) for a in argv)
    print(f"{name}|{filt}|{skip}|{quoted}")
PY
}

# Iterate JSON-defined stages from the named section and dispatch each one.
# Honours --runs filter via `_filter_runs` and the per-stage skip list via
# MNN_TEST_SKIP. The stage list is captured up-front into an array so the
# inner adb-shell calls in run_stage can't eat the read loop's stdin
# (classic gotcha with `while read … done < <(generator)` + ssh/adb).
_run_json_stages() {
    local section="${1:-android}"
    local runner="${2:-_remote_run_test}"
    local -a rows=()
    local row
    while IFS= read -r row; do
        rows+=("${row}")
    done < <(_emit_json_stages "${section}")
    local name filt skip argv
    for row in "${rows[@]}"; do
        IFS='|' read -r name filt skip argv <<<"${row}"
        [[ -z "${name}" ]] && continue
        if ! _filter_runs "${filt}"; then
            continue
        fi
        # ${argv} is shell-quoted by Python's shlex.quote, so eval-driven
        # word-splitting is required to honour the original token
        # boundaries. \$name defers expansion to eval's parse pass.
        if [[ -n "${skip}" ]]; then
            MNN_TEST_SKIP="${skip}" eval "run_stage \"\$name\" -- ${runner} ${argv}"
        else
            eval "run_stage \"\$name\" -- ${runner} ${argv}"
        fi
    done
}

android_tests() {
    _run_json_stages android _remote_run_test
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
    # Cache caffe sources on host (small download). The conversion itself
    # runs on device with the arm64 MNNConvert we build below.
    local smoke_ok=1
    provision_smoke_sources || smoke_ok=0
    android_build
    push_artifacts
    if [[ ${smoke_ok} -eq 1 ]]; then
        convert_smoke_on_device || smoke_ok=0
    fi
    # Unit/op + lowmem matrix. android-ci skips these so the on-device run
    # stays focused on bench / smoke / llm.
    if [[ "${FILTER}" != "android-ci" ]]; then
        android_tests
    fi
    if [[ ${smoke_ok} -eq 1 ]]; then
        # Each driver iterates the JSON-defined backend list and applies
        # _filter_runs internally, so a single call covers cpu/opencl/vulkan.
        android_smoke_a_stages
        android_smoke_b_stages
        android_benchmarks
    else
        skip_stage "smokeA" "smoke source download or on-device conversion failed"
        skip_stage "smokeB" "smoke source download or on-device conversion failed"
        skip_stage "bench"  "smoke source download or on-device conversion failed"
    fi

    # LLM stage. Provisioning + device push are lazy: deferred to here so the
    # unit / smoke / bench stages run even with no network, and so a missing
    # model skips only this stage instead of aborting the run.
    if _filter_runs llm; then
        if provision_llm_model; then
            push_llm_model
            android_llm_test
        else
            skip_stage "llm/${LLM_MODEL_NAME}" "model unavailable (see provisioning log above)"
        fi
    fi
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
