#!/bin/bash
echo '===> Generating Schema...'
FLATBUFFER_PATH=../../3rd_party/flatbuffers
FLATBUFFER_INCLUDE_PATH=$FLATBUFFER_PATH/include/flatbuffers
SCHMEMA_PATH=../../schema
FLAT_PARAMTER='-c -b --reflect-types --gen-mutable --reflect-names --gen-object-api'

# check is flatbuffer installed or not
FLATC=$FLATBUFFER_PATH/tmp/flatc

if [ ! -e $FLATC ]; then
  echo "*** building flatc ***"

  # make tmp dir
  pushd $FLATBUFFER_PATH > /dev/null
  [ ! -d tmp ] && mkdir tmp
  cd tmp && rm -rf *

  # build
  env -i bash -l -c "cmake .. && cmake --build . --target flatc -- -j4"

  # dir recover
  popd > /dev/null
fi

# determine directory to use
DIR="default"
if [ -d "$SCHMEMA_PATH/private" ]; then
  DIR="private"
fi

# # generate MNN flatbuffer header file
IR_PATH=source/IR
find $SCHMEMA_PATH/$DIR/*.fbs | xargs $FLATC $FLAT_PARAMTER -o $IR_PATH
cp -r $FLATBUFFER_INCLUDE_PATH $IR_PATH

# generate tflite flatbuffer header file
TFLITE_PATH=source/tflite/schema
$FLATC $FLAT_PARAMTER -o $TFLITE_PATH $TFLITE_PATH/schema.fbs
# mv $TFLITE_PATH/schema_v3_generated.h $TFLITE_PATH/schema_generated.h
echo '===> Done!'
