#!/bin/bash
set -eo pipefail

DSP_ARCH="$1"
if [ -z "$DSP_ARCH" ]; then
    DSP_ARCH="v79"
fi
DSP_ARCH_SUFFIX=$(echo "$DSP_ARCH" | tr '[:lower:]' '[:upper:]')

HTP_OPS_SDK_ENV="${HTP_OPS_SDK_ENV:-${HOME}/third/hexagon/setup_sdk_env.source}"
if [[ ! -f "${HTP_OPS_SDK_ENV}" ]]; then
    echo "Hexagon SDK environment file not found: ${HTP_OPS_SDK_ENV}"
    exit 1
fi

if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    python() {
        command python3 "$@"
    }
    export -f python
fi

source "${HTP_OPS_SDK_ENV}"
set -u

if [[ -n "${HTP_OPS_CMAKE_ROOT:-}" ]]; then
    export CMAKE_ROOT_PATH="${HTP_OPS_CMAKE_ROOT}"
fi
if [[ "${HTP_OPS_BUILD_ANDROID:-1}" == "1" ]]; then
    build_cmake android
fi
HEXAGON_BUILD_ARGS=(hexagon "DSP_ARCH=${DSP_ARCH}" "HTP_OPS_PWL_VARIANT=${HTP_OPS_PWL_VARIANT:-learned8}")
if [[ "${HTP_OPS_USE_MAKE:-0}" == "1" ]]; then
    HEXAGON_BUILD_ARGS+=("-gMake")
fi
build_cmake "${HEXAGON_BUILD_ARGS[@]}"
mkdir -p outputs
rm -f outputs/*.so
if [[ "${HTP_OPS_BUILD_ANDROID:-1}" == "1" ]]; then
    cp android_ReleaseG_aarch64/libMNN_htpops.so outputs/
fi
cp hexagon_ReleaseG_toolv19_$DSP_ARCH/libMNN_htpops_skel.so outputs/libMNN_htpops_skel${DSP_ARCH_SUFFIX}.so
if [ "$DSP_ARCH_SUFFIX" = "V79" ]; then
    cp hexagon_ReleaseG_toolv19_$DSP_ARCH/libMNN_htpops_skel.so outputs/libMNN_htpops_skel.so
fi

HEXAGON_STRIP_BIN="${HEXAGON_STRIP:-${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-strip}"
if [[ -x "${HEXAGON_STRIP_BIN}" ]]; then
    "${HEXAGON_STRIP_BIN}" --strip-debug outputs/libMNN_htpops_skel${DSP_ARCH_SUFFIX}.so
    if [[ "${DSP_ARCH_SUFFIX}" == "V79" ]]; then
        cp outputs/libMNN_htpops_skel${DSP_ARCH_SUFFIX}.so outputs/libMNN_htpops_skel.so
    fi
else
    echo "Warning: Hexagon strip tool not found; skeleton keeps debug information"
fi

# Check for unexpected undefined symbols in the DSP dynamic library
if ! ./check_so_symbols.sh outputs/libMNN_htpops_skel${DSP_ARCH_SUFFIX}.so; then
    echo "Build failed due to unresolved symbols in DSP library."
    exit 1
fi
