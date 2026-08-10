#!/bin/bash
# qnn_probe.sh <tensor_name> [<tensor_name> ...]
#
# 截断 ONNX 到 <tensor_name>，用 onnxruntime 生成参考，转成 MNN，push 到设备，
# 分别在 QNN(fwd=5,fp16) / CPU(fwd=0,fp32) / CPU(fwd=0,fp16) 上跑并并排打印 diff。
# 用于二分定位 QNN 首个出错算子:
#   - 某点 QNN-fp16 ≫ CPU-fp16 且是突跳  -> 该算子 QNN 实现 bug(深挖它)
#   - QNN-fp16 与 CPU-fp16 同步平滑增长   -> fp16 精度累积(见 reference 案例 2)
#   - CPU-fp32 应始终 ≈1e-3 以下(可信基线;若不满足说明参考/转换有问题)
#
# 依赖：主机构建目录 $BUILD 下有 MNNConvert；python 有 onnx/onnxruntime/numpy；
#       设备 /data/local/tmp/MNN/ 下已就绪 ModuleBasic.out 与 android build_64 产出的 libMNN.so。
# 用前按实际环境改下面几个变量。
set -e
ROOT=${MNN_ROOT:-/Users/qian/Documents/mnn/AliNNPrivate}
BUILD=${MNN_BUILD:-$ROOT/build}                 # 主机构建目录（有 MNNConvert）
MODEL=${MNN_ONNX:-$BUILD/src_model.onnx}        # 待测 onnx。⚠️ 必须放在 onnx/ 目录【之外】,
                                                #   否则 testMNNFromOnnx 拷成 onnx/test.onnx 时 SameFileError。
                                                #   常规做法: cp build/onnx/test.onnx build/src_model.onnx
DEVDIR=${MNN_DEVDIR:-/data/local/tmp/MNN}
VENV=${MNN_VENV:-/Users/qian/venvs/qian-env/bin/activate}

[ -z "$1" ] && { echo "usage: $0 <tensor_name> [<tensor_name> ...]"; exit 1; }
source "$VENV" 2>/dev/null || true
cd "$BUILD"

probe() {
  local NAME="$1"
  python3 "$ROOT/tools/script/testMNNFromOnnx.py" "$MODEL" "$NAME" >/tmp/qnn_probe_trunc.log 2>&1 || true
  if ! grep -q TEST_SUCCESS /tmp/qnn_probe_trunc.log; then
    printf "%-24s TRUNC/CPU-FAIL: %s\n" "$NAME" "$(grep -iE 'error|not found|TESTERROR' /tmp/qnn_probe_trunc.log | head -1)"
    return
  fi
  cp -f convert_cache.mnn onnx/test.mnn 2>/dev/null || true
  # ⚠️ QNN 在线路径要求 shapeMutable=false,否则模型输入不被拷入、首个算子吃全零(见 reference 案例 1)。
  python3 - <<PY
import json
p='onnx/input.json'; d=json.load(open(p)); d['shapeMutable']=False
json.dump(d, open(p,'w'), indent=2)
PY
  # 多输入模型: push 所有 input*.txt(单输入时就是 input.txt)。参考文件名为 <tensor>.txt。
  adb push onnx/test.mnn onnx/input*.txt onnx/input.json "onnx/${NAME}.txt" "$DEVDIR/onnx/" >/dev/null 2>&1
  f(){ echo "$1" | sed -E 's/.*diff rate = //'; }
  local Q C32 C16
  Q=$(adb shell   "cd $DEVDIR && rm -f .tempcache && export LD_LIBRARY_PATH=. && ./ModuleBasic.out onnx/test.mnn onnx 0 5 1 4 2 2>&1 | grep 'diff rate' | head -1")
  C32=$(adb shell "cd $DEVDIR && export LD_LIBRARY_PATH=. && ./ModuleBasic.out onnx/test.mnn onnx 0 0 1 4 1 2>&1 | grep 'diff rate' | head -1")
  C16=$(adb shell "cd $DEVDIR && export LD_LIBRARY_PATH=. && ./ModuleBasic.out onnx/test.mnn onnx 0 0 1 4 2 2>&1 | grep 'diff rate' | head -1")
  printf "%-24s QNNfp16=%-11s CPUfp32=%-11s CPUfp16=%-11s\n" "$NAME" "$(f "$Q")" "$(f "$C32")" "$(f "$C16")"
}

for t in "$@"; do probe "$t"; done
