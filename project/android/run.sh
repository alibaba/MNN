#!/bin/sh

set -e
# set -x

function showUsage () {
  echo "Usage: $0 [options] model_path [input_path] [-- [options-pass-to-MNNV2Basic.out]]"
  echo "options: "
  echo "  -h | --help: show this message and exit"
  echo "  -b | --build: build before run"
  echo "  -c | --clean: clean the build dir before build"
  echo "  --32: run in 32bits mode"
  echo "  --64: run in 64bits mode"
  echo "  --only-gpu: only run in gpu mode"
  echo "  --only-cpu: only run in cpu mode"
}

clean=0
build=0
enable_gpu=1
enable_cpu=1
remain_args=()
sub_args=()
bits=32
while [ "$1" != "" ]; do
  case $1 in
    -h | --help )
      showUsage
      exit 0
      ;;
    -b | --build )
      build=1
      ;;
    -c | --clean )
      clean=1
      ;;
    --32 )
      bits=32
      ;;
    --64)
      bits=64
      ;;
    --only-gpu )
      enable_gpu=1
      enable_cpu=0
      ;;
    --only-cpu )
      enable_cpu=1
      enable_gpu=0
      ;;
    -- )
      shift
      while [ "$1" != "" ]; do
        sub_args+=($1)
        shift
      done
      break
      ;;
    -* )
      echo "Unknown options: $1"
      showUsage
      exit 1
      ;;
    * )
      remain_args+=($1)
      ;;
  esac
  shift
done

if [ ${#remain_args[@]} -lt 1 ]; then
  showUsage
  exit -1
fi

model_path=${remain_args[0]}
if [ ! -f $model_path ]; then
  echo "The model \`$model_path' did not exist."
  exit -1
fi

model_name=$(basename $model_path)

if [ ${#remain_args[@]} -gt 1 ]; then
  input_path=${remain_args[1]}
  if [ ! -f $input_path ]; then
    echo "The input \`$input_path' did not exist."
    exit -1
  fi
  input_name=$(basename $input_path)
fi

if [ $clean -eq 1 ]; then
  build=1
  rm -fr build$bits
fi

mkdir -p build$bits
if [ $build -eq 1 ]; then
  pushd build$bits
    ../build_$bits.sh
  popd
fi

WORKING_DIR=/data/local/tmp/MNNv2
adb shell mkdir -p $WORKING_DIR
echo "push exe and library files"
adb push build$bits/MNNV2Basic.out $WORKING_DIR
adb push build$bits/libMNN.so $WORKING_DIR
adb push build$bits/project/android/OpenCL/libMNN_CL.so $WORKING_DIR

echo "push model and input file"
adb push $model_path $WORKING_DIR

adb shell LD_LIBRARY_PATH=$WORKING_DIR $WORKING_DIR/MNNV2Basic.out \
  $WORKING_DIR/$model_name \
  ${sub_args[@]}

