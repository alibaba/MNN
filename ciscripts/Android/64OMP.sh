./schema/generate.sh
cd project/android
mkdir build_64
cd build_64
../build_64.sh -DMNN_VULKAN=ON -DMNN_OPENMP=ON -DMNN_USE_THREAD_POOL=OFF
