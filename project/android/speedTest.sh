#!/bin/bash
make && adb push speedTest.out /data/local/tmp/speedTest.out && adb shell "/data/local/tmp/speedTest.out"
