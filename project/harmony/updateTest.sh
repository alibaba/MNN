#!/bin/bash
DIR=yanxing

make -j16
hdc file send ./libMNN.so /data/local/tmp/$DIR/libMNN.so
hdc file send ./libMNN_Express.so /data/local/tmp/$DIR/libMNN_Express.so
hdc file send ./MNNV2Basic.out /data/local/tmp/$DIR/MNNV2Basic.out
hdc file send ./ModuleBasic.out /data/local/tmp/$DIR/ModuleBasic.out
# hdc shell "cd /data/local/tmp/$DIR && rm -r output"
# hdc shell "cd /data/local/tmp/$DIR && mkdir output"
hdc file send ./unitTest.out /data/local/tmp/$DIR/unitTest.out
hdc file send ./testModel.out /data/local/tmp/$DIR/testModel.out
hdc file send ./testModelWithDescribe.out /data/local/tmp/$DIR/testModelWithDescribe.out
hdc file send ./backendTest.out /data/local/tmp/$DIR/backendTest.out
hdc file send ./timeProfile.out /data/local/tmp/$DIR/timeProfile.out
hdc file send ./run_test.out /data/local/tmp/$DIR/run_test.out
