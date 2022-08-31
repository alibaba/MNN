# TVM Mobile Benchmark

this folder contain the patch of benchmark needed, ios and android tuning and inference python script of tvm.

## Build TVM
First get tvm code from github:

    git clone https://github.com/apache/tvm.git

Checkout to the benchmark version:

    cd tvm && git checkout df6ccaec3bff66f8879c97df279cf096a6e73079

Apply the patch:

    git apply benchmark.patch

Then build tvm from source by [ref](https://tvm.apache.org/docs/install/from_source.html).

## Android Benchmark

Build an Android RPC apk at ``tvm/apps/android_rpc`` by [ref](https://github.com/apache/tvm/tree/main/apps/android_rpc), and run:

```bash
# Specify the RPC tracker
export TVM_TRACKER_HOST=0.0.0.0
export TVM_TRACKER_PORT=[PORT]
# Specify the standalone Android C++ compiler
export TVM_NDK_CC=/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++

# start RPC tracker
python -m tvm.exec.rpc_tracker --port 9090 --host 0.0.0.0
# start tuning and inference
python android_tuning.py <test_model_file>
```

## iOS Benchmark

Build an iOS RPC apk at ``tvm/apps/ios_rpc`` by [ref](https://github.com/apache/tvm/tree/main/apps/ios_rpc), and run:

```bash
export TVM_IOS_RPC_PROXY_HOST=0.0.0.0
export TVM_IOS_RPC_ROOT=${TVM_HOME}/your/testscript/path
export TVM_IOS_CODESIGN='Apple Development: xxxxx(xxxxx)'
export TVM_IOS_RPC_DESTINATION='platform=iOS,id=xxxxx-xxxxxxx'

# start RPC tracker
python -m tvm.exec.rpc_tracker --host=192.168.101.49  --port=9190 --no-fork
# start RPC proxy
python -m tvm.exec.rpc_proxy --no-fork --host 0.0.0.0 --port 9090 --tracker 0.0.0.0:9190
# tuning and inference
# change the `network` in ios_tuning.py to specify test model
python ios_tuning.py
```

## CUDA Benchmark
After building tvm from source, run the script:
	python tvmc_cuda_test.py

## MNN Semi-Search Time
The time of MNN resizeSession contains all semi-search time using.
Computing the time of resizeSession in MNN as below:
```bash
git apply mnn_semisearch_time.patch
# then run the benckmark script of MNN 
# get the output info: ### MNN Semi-Search Time is : xxx ms
```
