#!/bin/bash
# Copyright (c) 2023 Huawei Device Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
cd $(dirname $0)/../

export TOOLCHAIN_PATH=../../../../../vendor/hisi/npu/bin/misc/native/

rm -rf build/
mkdir build && cd build

${TOOLCHAIN_PATH}/build-tools/cmake/bin/cmake .. \
    -DOHOS_STL=c++_static \
    -DCMAKE_TOOLCHAIN_FILE=../../../../../vendor/hisi/npu/build/core/cmake/toolchain/ohos-ndk.toolchain.cmake \
	-DFFRT_EXAMPLE=ON \
	-DFFRT_BENCHMARKS=ON \
	-DFFRT_TEST_ENABLE=ON \
	-DFFRT_ST_ENABLE=ON \
	-DFFRT_HLT_ENABLE=ON \
	-DFFRT_PERF_EVENT_ENABLE=ON \

make -j ffrt
make -j ffrt_st
make -j