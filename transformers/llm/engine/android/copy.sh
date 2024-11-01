cd ../../../../
if not -d build 
then
    mkdir build
fi
if not -d "build/phone" 
then
    mkdir build/phone
fi
if not -d "build/phone/libMNN.so"
then
    cd build/phone
    export ANDROID_NDK=~/NDK/android-ndk
    cmake ../.. -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DMNN_USE_LOGCAT=true -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_USE_SSE=OFF -DMNN_SUPPORT_BF16=OFF -DMNN_BUILD_TEST=ON -DANDROID_NATIVE_API_LEVEL=android-30  -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_ARM82=ON -DMNN_OPENCL=ON -DLLM_SUPPORT_VISION=ON
    make -j20
    cd ../..
fi

cp build/phone/libMNN.so transformers/llm/engine/android/app/src/main/jni/libs/arm64-v8a
cp build/phone/libMNN_Express.so transformers/llm/engine/android/app/src/main/jni/libs/arm64-v8a
cp build/phone/libllm.so transformers/llm/engine/android/app/src/main/jni/libs/arm64-v8a
cp build/phone/libMNN_CL.so transformers/llm/engine/android/app/src/main/jni/libs/arm64-v8a
cp build/phone/tools/cv/libMNNOpenCV.so transformers/llm/engine/android/app/src/main/jni/libs/arm64-v8a

cd transformers/llm/engine/android/