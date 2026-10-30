cd ../../../project/android
mkdir -p build_64
cd build_64
../build_64.sh "\
-DMNN_LOW_MEMORY=true \
-DMNN_CPU_WEIGHT_DEQUANT_GEMM=true \
-DMNN_BUILD_LLM=true \
-DMNN_SUPPORT_TRANSFORMER_FUSE=true \
-DMNN_ARM82=true \
-DMNN_USE_LOGCAT=true \
-DMNN_OPENCL=true \
-DLLM_SUPPORT_VISION=true \
-DMNN_BUILD_OPENCV=true \
-DMNN_IMGCODECS=true \
-DLLM_SUPPORT_AUDIO=true \
-DMNN_BUILD_AUDIO=true \
-DMNN_BUILD_DIFFUSION=ON \
-DMNN_SEP_BUILD=OFF \
-DBUILD_PLUGIN=ON \
-DMNN_QNN=OFF \
-DMNN_HEXAGON=ON \
-DCMAKE_SHARED_LINKER_FLAGS='-Wl,-z,max-page-size=16384' \
-DCMAKE_INSTALL_PREFIX=."
make install
cd ../../../apps/Android/MnnLlmChat/

# Package the Hexagon FastRPC stub (libMNN_htpops.so) + DSP skel (libMNN_htpops_skel*.so) into the
# APK. They are built separately with the Qualcomm Hexagon SDK (see
# source/backend/hexagon/htp-ops-lib) and are not tracked in git. Override the source dir with
# HEXAGON_OPS_DIR. When absent we skip: libMNN.so still builds and the hexagon backend simply
# stays unavailable at runtime (registerHexagon dlopen fails -> falls back to cpu).
HEXAGON_OPS_DIR="${HEXAGON_OPS_DIR:-../../../source/backend/hexagon/htp-ops-lib/outputs}"
JNI_DIR="app/src/main/jniLibs/arm64-v8a"
if ls "$HEXAGON_OPS_DIR"/libMNN_htpops*.so >/dev/null 2>&1; then
    mkdir -p "$JNI_DIR"
    cp -f "$HEXAGON_OPS_DIR"/libMNN_htpops*.so "$JNI_DIR"/
    echo "Hexagon libs packaged from $HEXAGON_OPS_DIR:"
    ls -1 "$JNI_DIR"/libMNN_htpops*.so
else
    echo "Hexagon libs not found in $HEXAGON_OPS_DIR; skipping (hexagon backend disabled at runtime)."
fi

./gradlew assembleStandardDebug
