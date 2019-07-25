#$erroractionpreference = "stop"

echo '===> Generating Schema...'
Set-Variable -Name "FLATBUFFER_PATH" -Value "../../3rd_party/flatbuffers"
Set-Variable -Name "FLATBUFFER_INCLUDE_PATH" -Value "$FLATBUFFER_PATH/include/flatbuffers"
Set-Variable -Name "SCHMEMA_PATH" -Value "../../schema"
Set-Variable -Name "FLAT_PARAMTER" -Value "-c -b --reflect-types --gen-mutable --reflect-names --gen-object-api"

# check is flatbuffer installed or not
Set-Variable -Name "FLATC" -Value "$FLATBUFFER_PATH/tmp/flatc.exe"
if (-Not (Test-Path $FLATC -PathType Leaf)) {
  echo "*** building flatc ***"

  # make tmp dir
  pushd $FLATBUFFER_PATH
  if (-Not (Test-Path "tmp" -PathType Container)) {
    mkdir tmp
  }
  (cd tmp) -and (rm -rf *)

  # build
  cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --target flatc

  # dir recover
  popd
}

# determine directory to use
Set-Variable -Name "DIR" -Value "default"
if (Test-Path "$SCHMEMA_PATH/private" -PathType Container) {
  Set-Variable -Name "DIR" -Value "private"
}

# # generate MNN flatbuffer header file
Set-Variable -Name "IR_PATH" -Value "source/IR"
Get-ChildItem $SCHMEMA_PATH/$DIR/*.fbs | %{Invoke-Expression "$FLATC $FLAT_PARAMTER -o $IR_PATH $_"}
robocopy /xc /xn /xo /e $FLATBUFFER_INCLUDE_PATH $IR_PATH/flatbuffers

# generate tflite flatbuffer header file
Set-Variable -Name "TFLITE_PATH" -Value "source/tflite/schema"
Invoke-Expression "$FLATC $FLAT_PARAMTER -o $TFLITE_PATH $TFLITE_PATH/schema.fbs"
# mv $TFLITE_PATH/schema_v3_generated.h $TFLITE_PATH/schema_generated.h
echo '===> Done!'
