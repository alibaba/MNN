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
-DMNN_QNN=ON \
-DMNN_WITH_PLUGIN=ON \
-DBUILD_PLUGIN=ON \
-DCMAKE_SHARED_LINKER_FLAGS='-Wl,-z,max-page-size=16384' \
-DCMAKE_INSTALL_PREFIX=."
make install
cd ../../../apps/Android/MnnLlmChat/
./gradlew assembleStandardDebug
