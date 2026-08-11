#!/bin/bash
# cmp_probe.sh <tensor_name>
#
# 截断 ONNX 到 <tensor_name>，push 到设备，分别在 QNN(fwd=5) 与 OpenCL(fwd=3) 上以 fp16 跑，
# 并排打印两者相对 fp32 参考的 diff。用于“步骤 3”区分 QNN 真 bug 与 HTP fp16 精度：
#   - 某点 QNN 突跳而 OpenCL 不跳  -> 该算子真 bug
#   - QNN/OpenCL 平滑同步、Pool 处下降 -> 随机精度噪声
#   - QNN 比 OpenCL 大 ~2.5x/层并放大 -> HTP fp16 累加（OpenCL fp16 用 fp32 累加器）
#
# 注: 若设备没装 OpenCL(libMNN_CL.so),用 qnn_probe.sh 的 CPU-fp16 列同样能当 fp16 地板。
set -e
ROOT=${MNN_ROOT:-/Users/qian/Documents/mnn/AliNNPrivate}
BUILD=${MNN_BUILD:-$ROOT/build}
MODEL=${MNN_ONNX:-$BUILD/src_model.onnx}   # ⚠️ 必须在 onnx/ 目录【之外】(见 qnn_probe.sh 注释)
DEVDIR=${MNN_DEVDIR:-/data/local/tmp/MNN}
VENV=${MNN_VENV:-/Users/qian/venvs/qian-env/bin/activate}

NAME="$1"
[ -z "$NAME" ] && { echo "usage: $0 <tensor_name>"; exit 1; }
source "$VENV" 2>/dev/null || true
cd "$BUILD"

python3 "$ROOT/tools/script/testMNNFromOnnx.py" "$MODEL" "$NAME" >/dev/null 2>&1
cp -f convert_cache.mnn onnx/test.mnn 2>/dev/null || true
# QNN 在线路径要求 shapeMutable=false(见 reference 案例 1)
python3 - <<PY
import json
p='onnx/input.json'; d=json.load(open(p)); d['shapeMutable']=False
json.dump(d, open(p,'w'), indent=2)
PY
adb push onnx/test.mnn onnx/input*.txt onnx/input.json "onnx/${NAME}.txt" "$DEVDIR/onnx/" >/dev/null 2>&1

Q=$(adb shell "cd $DEVDIR && rm -f .tempcache && export LD_LIBRARY_PATH=. && ./ModuleBasic.out onnx/test.mnn onnx 0 5 1 4 2 2>&1 | grep 'diff rate' | head -1")
C=$(adb shell "cd $DEVDIR && export LD_LIBRARY_PATH=. && ./ModuleBasic.out onnx/test.mnn onnx 0 3 1 4 2 2>&1 | grep 'diff rate' | head -1")
printf "%-8s QNN-fp16=%-10s OpenCL-fp16=%-10s\n" "$NAME" \
    "$(echo "$Q" | sed -E 's/.*diff rate = //')" \
    "$(echo "$C" | sed -E 's/.*diff rate = //')"
