#!/bin/bash
make -j16
adb push ./libMNN.so /data/local/tmp/MNN/libMNN.so
adb push ./source/backend/opencl/libMNN_CL.so /data/local/tmp/MNN/libMNN_CL.so
adb push ./source/backend/vulkan/libMNN_Vulkan.so /data/local/tmp/MNN/libMNN_Vulkan.so
adb push ./source/backend/opengl/libMNN_GL.so /data/local/tmp/MNN/libMNN_GL.so
adb push ./express/libMNN_Express.so /data/local/tmp/MNN/libMNN_Express.so
adb push ./source/backend/arm82/libMNN_Arm82.so /data/local/tmp/MNN/libMNN_Arm82.so
adb push ./MNNV2Basic.out /data/local/tmp/MNN/MNNV2Basic.out
adb shell "cd /data/local/tmp/MNN && rm -r output"
adb shell "cd /data/local/tmp/MNN && mkdir output"
adb push ./unitTest.out /data/local/tmp/MNN/unitTest.out
adb push ./testModel.out /data/local/tmp/MNN/testModel.out
adb push ./testModelWithDescrisbe.out /data/local/tmp/MNN/testModelWithDescrisbe.out
adb push ./backendTest.out /data/local/tmp/MNN/backendTest.out
adb push ./timeProfile.out /data/local/tmp/MNN/timeProfile.out

