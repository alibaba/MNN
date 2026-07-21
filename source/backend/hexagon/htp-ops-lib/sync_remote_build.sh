#!/bin/bash
set -euo pipefail

# Usage: bash sync_remote_build.sh [DSP_ARCH]
# Required env: REMOTE_SSH. Optional env: DSP_ARCH, default v79.
REMOTE_SSH="${REMOTE_SSH:-}"
DSP_ARCH="${1:-${DSP_ARCH:-v79}}"

if [[ -z "${REMOTE_SSH}" ]]; then
    echo "REMOTE_SSH is not set"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_HTP_DIR="${SCRIPT_DIR}"
LOCAL_HEXAGON_DIR="$(cd "${LOCAL_HTP_DIR}/.." && pwd)"
LOCAL_MNN_DIR="$(cd "${LOCAL_HTP_DIR}/../../../.." && pwd)"
LOCAL_SCHEMA_DIR="${LOCAL_HEXAGON_DIR}/schema"
LOCAL_FLATBUFFERS_INCLUDE_DIR="${LOCAL_MNN_DIR}/3rd_party/flatbuffers/include"
LOCAL_OUTPUT_DIR="${LOCAL_HTP_DIR}/outputs"
BUILD_ID="$(date +%s)_$$_${RANDOM}_${RANDOM}"
REMOTE_SOURCE_ROOT="/tmp/cache_ssid_dsp_source_${BUILD_ID}"
REMOTE_HEXAGON_DIR="${REMOTE_SOURCE_ROOT}/source/backend/hexagon"
REMOTE_HTP_DIR="${REMOTE_HEXAGON_DIR}/htp-ops-lib"
LOCAL_ARCHIVE="$(mktemp -t htp-ops-lib.XXXXXX).zip"
LOCAL_STAGE="$(mktemp -d -t htp-ops-lib-stage.XXXXXX)"
REMOTE_ARCHIVE="${REMOTE_SOURCE_ROOT}/htp-ops-lib.zip"

cleanup() {
    rm -f "${LOCAL_ARCHIVE}"
    rm -rf "${LOCAL_STAGE}"
    if [[ -n "${REMOTE_SSH}" && -n "${REMOTE_SOURCE_ROOT}" ]]; then
        ssh "${REMOTE_SSH}" "rm -rf '${REMOTE_SOURCE_ROOT}'" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ ! -d "${LOCAL_SCHEMA_DIR}" ]]; then
    echo "Local schema dir not found: ${LOCAL_SCHEMA_DIR}"
    exit 1
fi
if [[ ! -d "${LOCAL_FLATBUFFERS_INCLUDE_DIR}" ]]; then
    echo "Local flatbuffers include dir not found: ${LOCAL_FLATBUFFERS_INCLUDE_DIR}"
    exit 1
fi

rm -rf "${LOCAL_OUTPUT_DIR}"

echo "[1/6] prepare remote temporary source dir: ${REMOTE_SSH}:${REMOTE_SOURCE_ROOT}"
ssh "${REMOTE_SSH}" "rm -rf '${REMOTE_SOURCE_ROOT}' && mkdir -p '${REMOTE_SOURCE_ROOT}'"

echo "[2/6] create local zip archive"
mkdir -p "${LOCAL_STAGE}/source/backend/hexagon" "${LOCAL_STAGE}/3rd_party/flatbuffers"
cp -R "${LOCAL_HTP_DIR}" "${LOCAL_STAGE}/source/backend/hexagon/htp-ops-lib"
cp -R "${LOCAL_SCHEMA_DIR}" "${LOCAL_STAGE}/source/backend/hexagon/schema"
cp -R "${LOCAL_FLATBUFFERS_INCLUDE_DIR}" "${LOCAL_STAGE}/3rd_party/flatbuffers/include"
(cd "${LOCAL_STAGE}" && zip -qr "${LOCAL_ARCHIVE}" .)

echo "[3/6] upload local htp-ops-lib zip to remote"
scp "${LOCAL_ARCHIVE}" "${REMOTE_SSH}:${REMOTE_ARCHIVE}"

echo "[4/6] unzip remote htp-ops-lib"
ssh "${REMOTE_SSH}" "cd '${REMOTE_SOURCE_ROOT}' && unzip -oq '${REMOTE_ARCHIVE}' && rm -f '${REMOTE_ARCHIVE}'"

echo "[5/6] build remote htp so with bash build.sh ${DSP_ARCH}"
ssh "${REMOTE_SSH}" "cd '${REMOTE_HTP_DIR}' && bash build.sh '${DSP_ARCH}'"

echo "[6/6] prepare local outputs dir and download outputs/*.so"
mkdir -p "${LOCAL_OUTPUT_DIR}"
rm -f "${LOCAL_OUTPUT_DIR}"/*.so

scp "${REMOTE_SSH}:${REMOTE_HTP_DIR}/outputs/*.so" "${LOCAL_OUTPUT_DIR}/"

echo "done"
echo "local outputs: ${LOCAL_OUTPUT_DIR}"

if [[ -n "${ANDROID_SERIAL:-}" ]]; then
    adb -s "${ANDROID_SERIAL}" push "${LOCAL_OUTPUT_DIR}"/*.so /data/local/tmp/MNN/
else
    DEVICE_COUNT="$(adb devices | awk 'NR > 1 && $2 == "device" {count++} END {print count + 0}')"
    if [[ "${DEVICE_COUNT}" == "1" ]]; then
        adb push "${LOCAL_OUTPUT_DIR}"/*.so /data/local/tmp/MNN/
    else
        echo "skip adb push: set ANDROID_SERIAL to select one device"
    fi
fi
