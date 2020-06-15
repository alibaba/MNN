#!/bin/bash
make -j16
adb push ./libMNN.so /data/local/tmp/MNN/libMNN.so
adb push ./libMNN_CL.so /data/local/tmp/MNN/libMNN_CL.so
adb push ./libMNN_Vulkan.so /data/local/tmp/MNN/libMNN_Vulkan.so
adb push ./libMNN_GL.so /data/local/tmp/MNN/libMNN_GL.so
adb push ./libMNN_Express.so /data/local/tmp/MNN/libMNN_Express.so
adb push ./libMNN_Arm82.so /data/local/tmp/MNN/libMNN_Arm82.so
adb push ./MNNV2Basic.out /data/local/tmp/MNN/MNNV2Basic.out
adb shell "cd /data/local/tmp/MNN && rm -r output"
adb shell "cd /data/local/tmp/MNN && mkdir output"
adb push ./unitTest.out /data/local/tmp/MNN/unitTest.out
adb push ./testModel.out /data/local/tmp/MNN/testModel.out
adb push ./testModelWithDescrisbe.out /data/local/tmp/MNN/testModelWithDescrisbe.out
adb push ./backendTest.out /data/local/tmp/MNN/backendTest.out
adb push ./timeProfile.out /data/local/tmp/MNN/timeProfile.out

adb push ./train.out /data/local/tmp/MNN/train.out
adb push ./benchmark.out /data/local/tmp/MNN/benchmark.out
adb push ./benchmarkExprModels.out /data/local/tmp/MNN/benchmarkExprModels.out
