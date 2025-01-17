## Qualcomm AI Engine Backend (QNN)


MNN refers to [mllm](https://github.com/UbiquitousLearning/mllm/) project for Qualcomm NPU direct supports.

### 0. Setup
Currently the qualcomm compilation to be performed on Linux platform.

The QNN backend relies on the Qualcomm QNN framework and Hexagon SDK to compile LLM-specific operators. Please download them using [QPM3 1.0.118.3](https://qpm.qualcomm.com/#/main/tools/details/QPM3). `qpm-cli` tool shall be used from terminal (or gui version alternatively). Instruction can be referred to `https://docs.qualcomm.com/bundle/publicresource/topics/80-77512-1/hexagon-dsp-sdk-install-addons-linux.html?product=1601111740010422`.

Version requirements:
* Qualcomm Neural Processing SDK : [Linux Latest](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk) (about 1GB)
* Hexagon SDK: [Linux 5.5](https://qpm.qualcomm.com/#/main/tools/details/HexagonSDK5.x)  (Some accounts may have no permission to access this SDK and may need to contact Qualcomm for support.) (about 1.7GB)

After downloading and installing the two SDKs, copy the SDK directories into the following paths:
* source/backend/qnn/3rd_party/qnn_ai/
* source/backend/qnn/3rd_party/Hexagon/ 

<b>We prepared the download and setup script for you in `codegen/qnn/setup.sh`.</b> Execute the script and then everything is settled for you.

```bash
# login with your Qualcomm account
qpm-cli --login
# execute the script in MNN root
sh codegen/qnn/setup.sh
```

(It may take up several minutes for downloading and extracting the components. If it exits with error, simply reexecute the script.)

### 1. Compilation

~~~bash
export MNN_ROOT=$(pwd)
export QNN_SDK_ROOT=${MNN_ROOT}/source/backend/qnn/3rd_party/qnn_ai/
export ANDROID_NDK_ROOT=~/NDK/android-ndk
export PATH=$PATH:$ANDROID_NDK_ROOT

source source/backend/qnn/3rd_party/Hexagon/setup_sdk_env.source
source $QNN_SDK_ROOT/bin/envsetup.sh
~~~

You may need to install `libtinfo5` as well.
~~~bash
sudo apt-get install libtinfo5
~~~