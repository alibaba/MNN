#!/bin/bash
DIR=MNN

make -j16
adb push ./libllm.so /data/local/tmp/MNN/libllm.so
adb push ./llm_demo /data/local/tmp/MNN/llm_demo
adb push ./diffusion_demo /data/local/tmp/MNN/diffusion_demo
adb push ./libMNN.so /data/local/tmp/$DIR/libMNN.so
adb push ./libMNN_CL.so /data/local/tmp/$DIR/libMNN_CL.so
adb push ./libMNN_Vulkan.so /data/local/tmp/$DIR/libMNN_Vulkan.so
adb push ./libMNN_GL.so /data/local/tmp/$DIR/libMNN_GL.so
adb push source/backend/hiai/libMNN_NPU.so /data/local/tmp/$DIR/libMNN_NPU.so
adb push ./libMNN_Express.so /data/local/tmp/$DIR/libMNN_Express.so
adb push ./MNNV2Basic.out /data/local/tmp/$DIR/MNNV2Basic.out
adb push ./ModuleBasic.out /data/local/tmp/$DIR/ModuleBasic.out
adb shell "cd /data/local/tmp/$DIR && rm -r output"
adb shell "cd /data/local/tmp/$DIR && mkdir output"
adb push ./unitTest.out /data/local/tmp/$DIR/unitTest.out
adb push ./testModel.out /data/local/tmp/$DIR/testModel.out
adb push ./testModelWithDescribe.out /data/local/tmp/$DIR/testModelWithDescribe.out
adb push ./backendTest.out /data/local/tmp/$DIR/backendTest.out
adb push ./timeProfile.out /data/local/tmp/$DIR/timeProfile.out

adb push ./train.out /data/local/tmp/$DIR/train.out
adb push ./benchmark.out /data/local/tmp/$DIR/benchmark.out
adb push ./benchmarkExprModels.out /data/local/tmp/$DIR/benchmarkExprModels.out
adb push ./run_test.out /data/local/tmp/$DIR/run_test.out
