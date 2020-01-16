set -e
schema/generate.sh
cd project/android
rm -rf build_32
mkdir build_32
cd build_32
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="armeabi-v7a" -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release -DANDROID_NATIVE_API_LEVEL=android-21  -DANDROID_TOOLCHAIN=clang -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. -DMNN_VULKAN=ON -DMNN_OPENMP=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_OPENGL=ON -DMNN_OPENCL=ON ../../../
make -j8
