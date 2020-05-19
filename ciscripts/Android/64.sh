./schema/generate.sh
cd project/android
mkdir build_64
cd build_64
cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_USE_LOGCAT=true \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. \
-DMNN_VULKAN=ON -DMNN_OPENCL=ON -DMNN_ARM82=ON

make -j4
