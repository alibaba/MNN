#!/bin/bash
adb push ./input_0.txt /data/local/tmp/MNN/input_0.txt
adb push ./input_1.txt /data/local/tmp/MNN/input_1.txt
adb push ./temp.bin /data/local/tmp/MNN/temp.bin

./MNNV2Basic.out temp.bin

adb pull /data/local/tmp/MNN/output.txt output_android.txt
