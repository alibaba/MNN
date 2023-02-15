#!/bin/bash
adb push ./input_0.txt /data/local/tmp/MNN/input_0.txt
adb push ./input_1.txt /data/local/tmp/MNN/input_1.txt
adb push ./temp.bin /data/local/tmp/MNN/temp.bin

#./MNNV2Basic.out temp.bin

adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH&&./MNNV2Basic.out temp.bin $1 $2 $3 $4 $5 $6"

adb pull /data/local/tmp/MNN/output.txt output_android.txt
