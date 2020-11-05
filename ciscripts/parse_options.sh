#!/usr/bin/env bash

# Valid platform:
#   - arm_android_32
#   - arm_android_64
#   - arm_linux_32
#   - arm_linux_64
#   - x86_linux
platform="all"

# Option to build with opencl.
use_opencl=0

# Option to build with opengl.
use_opengl=0

# Option to build with vulkan.
use_vulkan=0

# Option to build with openmp multithreads library.
use_openmp=0

build_threads=1

# Option to clear the build history.
clean=0

USE_OPENCL=OFF
USE_VULKAN=OFF
USE_OPENGL=OFF
USE_OPENMP=OFF
USE_THREAD_POOL=ON

function print_usage {
  echo -e "Usgae: ./build.sh"
  echo -e "  --platform=x: Specify build platform x. "
  echo -e "      All valid platforms are \"arm_android_32\", \"arm_android_64\",
                \"arm_linux_32\", \"arm_linux_64\", \"x86_linux\", \"all\"."
  echo -e "      The default is \"all\"."
  echo -e "  --use_openmp=true|false: Build with openmp or not."
  echo -e "      The default is false."
  echo -e "  --use_opencl=true|false: Build with opencl or not."
  echo -e "      The default is false."
  echo -e "  --use_opengl=true|false: Build with opengl or not."
  echo -e "      The default is false."
  echo -e "  --use_vulkan=true|false: Build with vulkan or not."
  echo -e "      The default is false."
  echo -e "  --job=n: Build with n threads. Default is 1."
}

function parse_platform {
  platform=`echo "$1" | awk -F '=' '{print $2}'`
}

function parse_nthreads {
  build_threads=`echo "$1" | awk -F '=' '{print $2}'`
}

function parse_bool {
  val=`echo "$1" | awk -F '=' '{print $2}'`
  if [ $val == "true" ] || [ $val == "1" ]; then
    return 1;
  else
    return 0;
  fi
}

[ -z "${1:-}" ] && print_usage && exit 1;

while true; do
  [ -z "${1:-}" ] && break;
  case "$1" in
    --platform=*) parse_platform "$1"; shift 1;
      ;;
    --use_openmp=*) parse_bool "$1"; use_openmp=$?; shift 1;
      ;;
    --use_openmp) use_openmp=true; shift 1;
      ;;
    --use_opencl=*) parse_bool "$1"; use_opencl=$?; shift 1;
      ;;
    --use_opencl) use_opencl=true; shift 1;
      ;;
    --use_opengl=*) parse_bool "$1"; use_opengl=$?; shift 1;
      ;;
    --use_opengl) use_opengl=true; shift 1;
      ;;
    --use_vulkan=*) parse_bool "$1"; use_vulkan=$?; shift 1;
      ;;
    --use_vulkan) use_vulkan=true; shift 1;
      ;;
    --job=*) parse_nthreads "$1"; shift 1;
      ;;
    clean) clean=1; shift 1;
      ;;
    *) break;
  esac
done

if [ $use_opencl == 1 ]; then
  USE_OPENCL=ON
fi
if [ $use_opengl == 1 ]; then
  USE_OPENGL=ON
fi
if [ $use_vulkan == 1 ]; then
  USE_VULKAN=ON
fi
if [ $use_openmp == 1 ]; then
  USE_OPENMP=ON
  USE_THREAD_POOL=OFF
fi

true;
