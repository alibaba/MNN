#!/bin/bash
adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH&&./MNNV2Basic.out temp.bin $1 $2 $3 $4 $5 $6"

