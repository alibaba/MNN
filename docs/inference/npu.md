# NPU 及相应后端使用说明

目前 MNN 支持通过如下后端调用部分手机上的NPU能力：
- QNN
- Hexagon
- CoreML
- NNAPI
- HIAI
- RKNN

## 可选 NPU 插件接入

QNN / NeuroPilot / HiAI 支持独立插件，各后端提供配置、诊断、动态加载和
严格 Session API：

- QNN：`MNNQnnBackend.h`、`MNNQnnPlugin.hpp`、`MNNQnnSession.hpp`；
- NeuroPilot：`MNNNeuroPilotBackend.h`、`MNNNeuroPilotPlugin.hpp`、`MNNNeuroPilotSession.hpp`；
- HiAI：`MNNHiAIBackend.h`、`MNNHiAIIO.h`、`MNNHiAIPlugin.hpp`、`MNNHiAISession.hpp`。

配置通过 `BackendConfig::sharedContext` 传入。调用方加载目标芯片对应的插件，
再调用该后端的 `create*Session`。NeuroPilot 在线模式与 NNAPI/CoreML 编译互斥。

## QNN

### QNN后端整体介绍

- MNN通过调用QNN SDK的CPP API构建了MNN-QNN后端，以期在能够使用高通NPU的设备上取得推理加速。
- 我们支持了两种运行模式：
  - 在线构图模式，在线编译和序列化QNN计算图。
    - 支持静态形状的常规模型的推理。
  - 离线构图模式则先借助MNN的离线工具缓存QNN计算图的序列化产物，接着在运行时直接读取产物，可以节省初始化时间。
    - 支持静态形状/有限形状组合的常规模型的推理。
    - 可支持部分llm模型的推理加速。

### 准备工作

#### 开发环境
- Host
  - 在线构图模式：无要求。
  - 离线构图模式：一台x86_64，Linux的机器（链路中的部分QNN工具必须在此环境中运行）。
- Device
  - 一台可以使用高通NPU的设备；为便于陈述，下文假设这是一台Android系统的设备。

#### 明确硬件架构

QNN后端的部分使用步骤（如生成离线产物，确定QNN的NPU库依赖等）需要指定device的硬件架构对应的SOC ID以及HEXAGON ARCH。对于一些常见的硬件架构，我们列举如下供你参考：

| 硬件    | SOC ID | HEXAGON ARCH |
| :------ | :----- | :----------- |
| 8 Gen 1 | 36     | 69           |
| 8 Gen 2 | 43     | 73           |
| 8 Gen 3 | 57     | 75           |
| 8 Elite | 69     | 79           |

对于其他的硬件架构，你可以参考高通官网的设备支持列表。

#### 获得QNN依赖

MNN-QNN后端依赖QNN SDK中的`include/QNN`与`lib`，可通过以下步骤获取依赖：
- [注册高通账号](https://myaccount.qualcomm.com/signup)
- 访问Qualcomm AI Engine Direct SDK（即QNN SDK），下载SDK，并解压。比如`/home/xiaying/third/qnn/qairt/2.38.0.250901`
- 修改`~/.bashrc` ，增加SDK路径到环境变量, 然后运行 `source ~/.bashrc` 或者重启终端。eg：

```
export QNN_SDK_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
export QNN_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
export HEXAGON_SDK_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
```

### 在线构图模式，推理常规模型
在线构图模式的使用步骤与其他后端基本一致，主要包含以下三部分。

#### Host，交叉编译Device侧的MNN库及AI应用程序
- 参考[“主库编译”](../compile/engine.md#主库编译)，配置Android系统的编译环境及CMake变量。
- 添加额外的CMake变量并编译：`-DMNN_QNN=ON`、`-DMNN_QNN_CONVERT_MODE=OFF`、`-DMNN_WITH_PLUGIN=OFF`。

#### 推送资源至Device

参考下面的指令，将以下资源推送到Device侧
- AI应用程序。
- 交叉编译得到的Device侧的MNN库。
- QNN库（`libQnnHtp.so`、`libQnnHtpV${HEXAGON_ARCH}Stub.so`、`libQnnHtpV${HEXAGON_ARCH}Skel.so`、`libQnnHtpPrepare.so`）。
- MNN模型。
```
HEXAGON_ARCH="75" # modify this variable according to your environment
MNN_ROOT_PATH="/YOUR/MNN/ROOT/PATH" # modify this variable according to your environment
BUILD_ANDROID_PATH="/your/build/andorid/path" # modify this variable according to your environment
ANDROID_WORKING_DIR="/data/local/tmp" # modify this variable according to your environment

# push mnn libs
cd ${BUILD_ANDROID_PATH}
find . -name "*.so" | while read solib; do
    adb push $solib ${ANDROID_WORKING_DIR}
done
cd -

# push your AI exe
adb push /your/AI/exe ${ANDROID_WORKING_DIR}

# push QNN libs
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV${HEXAGON_ARCH}Stub.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v${HEXAGON_ARCH}/unsigned/libQnnHtpV${HEXAGON_ARCH}Skel.so ${ANDROID_WORKING_DIR}
# The following lib is only needed in the online case.
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so ${ANDROID_WORKING_DIR}

# push MNN models
adb push model.mnn ${ANDROID_WORKING_DIR}
```

#### Device，链接并运行
- 链接QNN库
  - 为了动态链接到QNN HTP相关的库，需要在环境变量`ADSP_LIBRARY_PATH`中添加QNN HTP库所在的目录（部分机型上有效）。如果这样也没法成功链接，可将可执行文件，QNN HTP库推送至同一目录，cd到对应目录后，再运行可执行文件，参考如下指令。
```
adb shell "cd ${ANDROID_WORKING_DIR} && export LD_LIBRARY_PATH=.:${ANDROID_LD_LIBRARY_PATH} && export ADSP_LIBRARY_PATH=.:${ANDROID_ADSP_LIBRARY_PATH} && ./your/mnn/qnn/ai/exe"
```
- 配置MNN
  - Backend Type 显式设置为 `MNN_FORWARD_QNN`，即 16；不再占用 NNAPI/NeuroPilot 的 5。
  - 在使用Module API推理时，需要设定`Module::Config`中的`shapeMutable`字段为`false`。

### 离线构图模式，推理常规模型
相较于在线构图模式，离线构图模式额外包含一次编译（构建生成离线产物需要的MNN库）以及一个模型转换步骤（将原始的MNN模型转化成QNN产物），具体如下。

#### Host，编译生成离线模式产物需要的的MNN库及相应MNN离线工具
- 添加额外的CMake变量并编译：`-DMNN_QNN=ON`、`-DMNN_QNN_CONVERT_MODE=ON`、`-DMNN_WITH_PLUGIN=OFF`、`-DMNN_BUILD_TOOLS=ON`。

#### Host，生成QNN离线构图产物
调用`MNN2QNNModel`工具，针对Device的硬件架构，生成QNN离线产物（`model_${SOC_ID}_${HEXAGON_ARCH}.bin`）以及替代模型（`model_${SOC_ID}_${HEXAGON_ARCH}.mnn`），具体可参考[该工具的用法](../tools/convert.md#mnn2qnnmodel)。

#### Host，交叉编译Device侧的MNN库及AI应用程序
- 参考[“主库编译”](../compile/engine.md#主库编译)，配置Android系统的编译环境及CMake变量。
- 添加额外的CMake变量并编译：`-DMNN_QNN=ON`、`-DMNN_QNN_CONVERT_MODE=OFF`、`-DMNN_WITH_PLUGIN=ON`。

#### 推送资源至Device
与[在线构图模式的情况](#推送资源至device)类似，但有以下两点不同：
- 依赖的QNN库变为`libQnnHtp.so`、`libQnnHtpV${HEXAGON_ARCH}Stub.so`、`libQnnHtpV${HEXAGON_ARCH}Skel.so`、`libQnnSystem.so`（不再依赖`libQnnHtpPrepare.so`，而是依赖`libQnnSystem.so`）。
- 不再使用原始的MNN模型，而是需要QNN离线产物（`model_${SOC_ID}_${HEXAGON_ARCH}.bin`）以及替代模型（`model_${SOC_ID}_${HEXAGON_ARCH}.mnn`）。

#### Device，链接并运行
- 配置MNN
    - 指定backend type为0（CPU）。读取并推理QNN离线产物的功能被封装在Plugin算子内，该算子被注册在CPU后端，因此，此时需要指定backend type为CPU。
    - 在Device侧，如果你的离线产物和你的应用的工作目录不一致，那么你需要在程序中通过`Executor::RuntimeManager::setExternalPath`接口设定离线产物所在的目录。
- 链接QNN库
    - 离线构图模式对于链接的要求和在线构图模式一致。


## Hexagon

### Hexagon后端整体介绍

- 与上面的QNN后端不同，`MNN_HEXAGON`不经过QNN构图，而是MNN自带的算子库直接运行在高通cDSP（Hexagon）上，内核由HVX / HMX手写实现，host与DSP之间通过FastRPC的stub / skel通信，不依赖QNN SDK的运行库。
- 后端代码在`source/backend/hexagon/`，DSP侧算子库在`source/backend/hexagon/htp-ops-lib/`。
- 两条高通路径互相独立，同一个模型只会走其中一条。Hexagon后端是一个普通的MNN后端，Backend Type为`MNN_FORWARD_HEXAGON`，即10；LLM场景下通过`config.json`的`"backend_type": "hexagon"`选择。
- 当前主要面向端侧LLM，Attention / KVCache / RoPE / 量化MatMul等主链路算子已覆盖，同时也实现了一部分常规CV算子。
- 不支持的算子由`HexagonBackend::onCreate`返回`nullptr`，Pipeline会自动回退到`backupType`（一般是CPU），机制与OpenCL / Vulkan一致。因此模型能跑通不代表整张图都在DSP上执行。
- 运行期后端强制`numThread = 1`，并行由DSP侧的worker完成，host侧的线程数设置对它无效。

### 准备工作

#### 开发环境

- Host
  - 编译DSP侧算子库需要一台x86_64 Linux机器。Hexagon SDK的主机版本只有Linux与Windows，没有macOS，而`build.sh` / `sync_remote_build.sh`都是bash脚本，所以实际按Linux主机准备；在Mac上开发时，skel必须通过ssh到远程Linux机器编译，仓库里的`sync_remote_build.sh`就是为此准备的（见下文）。
  - 编译MNN主库本身不需要Hexagon SDK，普通的Android交叉编译环境即可。
  - 远程编译方式下，本机还需要`ssh`、`scp`、`zip`，远程机器需要`unzip`；本机装了`adb`的话可以顺带把产物推到设备。
- Device
  - 一台搭载骁龙8 Gen 2及以上SoC的Android设备，即Hexagon v73及以上。
  - 实现依赖FP16 HMX，部分中低端机型即使SoC达标也可能没有该单元，此时后端不可用。

#### 明确硬件架构

DSP侧的skel按Hexagon arch分别编译，一份产物不能跨代复用，所以编译和部署前都要先确定设备的arch。常见机型的对应关系如下：

| 硬件 | SoC | DSP arch | skel文件名 |
| :------ | :----- | :----------- | :---------- |
| 8 Gen 2 | SM8550 | v73 | `libMNN_htpops_skelV73.so` |
| 8 Gen 3 | SM8650 | v75 | `libMNN_htpops_skelV75.so` |
| 8 Elite | SM8750 | v79 | `libMNN_htpops_skelV79.so` |
| — | SM8850 | v81 | `libMNN_htpops_skelV81.so` |

8 Gen 1（SM8450，v69）及更早的设备不支持。Android App侧可以用`Build.SOC_MODEL`读到SoC型号（API 31 / Android 12 及以上才有该字段），`HexagonModule.kt`里已经内置了这张表。

#### 获得Hexagon SDK依赖

只有编译DSP侧算子库需要SDK，步骤如下：

- [注册高通账号](https://myaccount.qualcomm.com/signup)，下载Hexagon SDK 6.x并解压（我们验证过的版本是6.0.0.2）。SDK一般装在前面提到的那台Linux机器上。
- 在Linux机器上准备好SDK环境文件路径，`build.sh`通过`HTP_OPS_SDK_ENV`找到它（默认`${HOME}/third/hexagon/setup_sdk_env.source`）：

```
export HTP_OPS_SDK_ENV=/home/yourname/third/hexagon/Hexagon_SDK/6.0.0.2/setup_sdk_env.source
export HEXAGON_SDK_ROOT=/home/yourname/third/hexagon/Hexagon_SDK/6.0.0.2
```

- 同一台机器上还需要Android NDK，用于编译aarch64侧的stub；NDK路径可以用`HTP_OPS_ANDROID_ROOT`指定。

SDK的许可不允许我们把工具链产物提交进仓库，所以`htp-ops-lib/outputs/`下的`.so`不入库，每个人都要自己编一次。

### 编译DSP算子库

产物有两个，分别在CPU和cDSP上运行，可以用`ldd`区分：

- `libMNN_htpops.so`：aarch64 stub，由`libMNN.so`在运行时按soname `dlopen`。
- `libMNN_htpops_skelV${ARCH}.so`：DSP skel，由FastRPC按arch后缀查找并加载到cDSP。

只有v79会额外复制一份不带后缀的`libMNN_htpops_skel.so`，其他arch部署时必须使用带`V${ARCH}`后缀的文件名。

#### 方式一，在装有SDK的Linux机器上编译

```
cd source/backend/hexagon/htp-ops-lib
bash build.sh v75 # modify this variable according to your device
```

不传arch时默认v79。产物输出到`htp-ops-lib/outputs/`。脚本结束时会调用`check_so_symbols.sh`检查skel的未定义符号，检查不通过会直接以非0退出——这类符号会让DSP侧`dlopen`失败，并在FastRPC上返回error 44（`AEE_EFAILED`）。

脚本从`android_ReleaseG_aarch64/`和`hexagon_ReleaseG_toolv19_${DSP_ARCH}/`取产物，也就是假定工具链为Hexagon Tools 19.x（对应SDK 6.x）；换用其他大版本的SDK时需要同步修改脚本里的目录名。

`build.sh`支持下列环境变量，都有默认值，正常情况不需要设置：

| 变量 | 作用 |
| :--- | :--- |
| `HTP_OPS_SDK_ENV` | SDK `setup_sdk_env.source`的路径 |
| `HTP_OPS_ANDROID_ROOT` | 编译stub用的Android NDK根目录 |
| `HTP_OPS_CMAKE_ROOT` | CMake根目录 |
| `HTP_OPS_BUILD_ANDROID` | 置0跳过aarch64 stub，只编skel |
| `HTP_OPS_USE_MAKE` | 置1让SDK走gMake而非ninja |
| `HTP_OPS_PWL_VARIANT` | 分段线性激活的系数变体，默认`learned8` |
| `HTP_OPS_MM_PHASE_PROFILE` | 置`ON`打开fp16 HMX matmul的分阶段profiling |
| `HEXAGON_NM` | `check_so_symbols.sh`使用的nm工具路径 |

#### 方式二，本机通过ssh在远程Linux机器上编译

在Mac等没有SDK的主机上开发时用这个方式。`sync_remote_build.sh`会把`htp-ops-lib`、`source/backend/hexagon/schema`和`3rd_party/flatbuffers/include`打包上传到远程临时目录，按传入的arch逐个编译，再把所有产物下载回本机的`outputs/`，最后清理远程目录。

```
cd source/backend/hexagon/htp-ops-lib
export REMOTE_SSH=yourname@hexagon-build-server # modify this variable according to your environment
export HTP_OPS_SDK_ENV=/home/yourname/third/hexagon/Hexagon_SDK/6.0.0.2/setup_sdk_env.source # 该路径在远程机器上生效
bash sync_remote_build.sh v73 v75 v79
```

- 支持一次传多个arch，不传则用`DSP_ARCH`（默认v79）。
- `HTP_OPS_SDK_ENV`、`HTP_OPS_CMAKE_ROOT`等变量会原样透传到远程的`build.sh`，所以填的是远程机器上的路径。
- 产物下载目录可用`HTP_OPS_OUTPUT_DIR`覆盖，默认`htp-ops-lib/outputs/`。
- 如果本机装了`adb`，脚本会顺带把产物推到设备的`/data/local/tmp/MNN/`；接了多台设备时需要用`ANDROID_SERIAL`指定其中一台，否则会跳过推送。

#### 无设备时的环境自检

在装有SDK的机器上，先跑过一次`build.sh <arch>`，再执行`bash run_sim_so.sh v79`：它会在Hexagon模拟器里跑一段HVX代码，确认当前SDK能为该arch正常生成并执行HVX指令。这只是工具链自检，不测试算子功能，需要`HEXAGON_SDK_ROOT`，并且依赖上一步产生的`hexagon_ReleaseG_toolv19_${DSP_ARCH}/libMNN_htpops_skel.so`。

### Host，交叉编译Device侧的MNN库及AI应用程序

- 参考[“主库编译”](../compile/engine.md#主库编译)，配置Android系统的编译环境及CMake变量。
- 添加额外的CMake变量并编译：`-DMNN_HEXAGON=ON`。可选项的含义见[“CMake选项”](../compile/cmake.md)：
  - `-DMNN_HEXAGON_ASAN=ON`：打开后端内部的内存一致性检查，排查越界问题时使用。
  - `-DMNN_HEXAGON_APP_POWER_HOLD=ON`：在命令flush之间保持DSP电源投票、空闲后释放，面向App进程；命令行工具和服务端不需要。
- 如果要跑在16 KB page size的Android设备上，另外加上`-DCMAKE_SHARED_LINKER_FLAGS='-Wl,-z,max-page-size=16384'`。这是Android的通用要求，与Hexagon后端无关，NDK r28及以上默认已按16 KB对齐；`apps/Android/MnnLlmChat/build.sh`里已经带上了。
- LLM场景下还需要LLM相关的编译开关（`-DMNN_BUILD_LLM=ON`、`-DMNN_LOW_MEMORY=ON`、`-DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON`、`-DMNN_SUPPORT_TRANSFORMER_FUSE=ON`等），完整的命令行可以参考`apps/Android/MnnLlmChat/build.sh`。

主库编译不需要Hexagon SDK：stub是运行时`dlopen`的，没有链接期依赖。如果设备上找不到`libMNN_htpops.so`，`registerHexagon`不会注册该后端，推理自动回退到CPU，不会崩溃。

### 推送资源至Device

参考下面的指令，将以下资源推送到Device侧

- AI应用程序（或`llm_demo` / `llm_bench`）。
- 交叉编译得到的Device侧的MNN库。
- Hexagon库（`libMNN_htpops.so`、`libMNN_htpops_skelV${HEXAGON_ARCH}.so`）。
- MNN模型。

```
HEXAGON_ARCH="75" # modify this variable according to your environment
BUILD_ANDROID_PATH="/your/build/android/path" # modify this variable according to your environment
HTP_OPS_OUTPUT_PATH="/your/MNN/root/path/source/backend/hexagon/htp-ops-lib/outputs" # modify this variable according to your environment
ANDROID_WORKING_DIR="/data/local/tmp/MNN" # modify this variable according to your environment

# push mnn libs
cd ${BUILD_ANDROID_PATH}
find . -name "*.so" | while read solib; do
    adb push $solib ${ANDROID_WORKING_DIR}
done
cd -

# push hexagon libs
adb push ${HTP_OPS_OUTPUT_PATH}/libMNN_htpops.so ${ANDROID_WORKING_DIR}
adb push ${HTP_OPS_OUTPUT_PATH}/libMNN_htpops_skelV${HEXAGON_ARCH}.so ${ANDROID_WORKING_DIR}

# push your AI exe
adb push /your/AI/exe ${ANDROID_WORKING_DIR}

# push MNN models
adb push model ${ANDROID_WORKING_DIR}
```

### Device，链接并运行

- 链接Hexagon库
  - `LD_LIBRARY_PATH`需要包含`libMNN_htpops.so`所在目录，供动态链接器加载stub。
  - `ADSP_LIBRARY_PATH`需要包含`libMNN_htpops_skelV${HEXAGON_ARCH}.so`所在目录，供FastRPC加载skel。两个都必须设置，这也是目前无法省掉的环境配置。
  - 后端以unsigned PD打开cDSP会话（domain 3），skel不需要签名。

```
adb shell "cd ${ANDROID_WORKING_DIR} && export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=${ANDROID_WORKING_DIR} && ./your/mnn/hexagon/ai/exe"
```

- 配置MNN
  - Backend Type显式设置为`MNN_FORWARD_HEXAGON`，即10。
  - LLM场景下把模型`config.json`的`backend_type`改成`hexagon`即可，导出约束（4-bit对称量化、Transformer C4导出）和`llm_demo` / `llm_bench`的用法见[“Android Hexagon 后端”](../transformers/llm.md#android-hexagon-后端)。

### App（MnnLlmChat）集成

`apps/Android/MnnLlmChat/build.sh`已经把上面的流程串好了：

1. 以`-DMNN_HEXAGON=ON -DMNN_HEXAGON_APP_POWER_HOLD=ON`交叉编译主库；
2. 把`htp-ops-lib/outputs/libMNN_htpops*.so`拷进`app/src/main/jniLibs/arm64-v8a`，随APK一起打包，来源目录可用`HEXAGON_OPS_DIR`覆盖；产物不存在时跳过打包，App照常构建，运行时回退CPU；
3. 运行期由`HexagonModule.kt`按`Build.SOC_MODEL`判断设备是否在支持列表内、确认stub已打包，并把`ADSP_LIBRARY_PATH` / `LD_LIBRARY_PATH`指向App的native lib目录。

所以App场景下只需要先跑一次`htp-ops-lib`的编译（本机或远程），其余是自动的。

### 常见问题

- 速度和纯CPU一样：说明没有走上DSP。依次检查`libMNN_htpops.so`是否在`LD_LIBRARY_PATH`中、skel的arch后缀是否与设备匹配、`ADSP_LIBRARY_PATH`是否设置。
- FastRPC返回error 44（`AEE_EFAILED`）：skel里有未定义的符号，重跑`check_so_symbols.sh outputs/libMNN_htpops_skelV${HEXAGON_ARCH}.so`看具体是哪些。
- 改了DSP侧实现但设备行为没变：修改`htp-ops-lib/src/dsp/`下的代码后必须重新编译并重新推送skel，否则设备仍在跑旧的DSP实现。
- 想知道哪些算子回退到了CPU：看日志里的`[MNN::Hexagon] create unsupported: type=..., name=...`，由`HexagonExecutionFactory`打印，每一条对应一个回退算子。按模型提issue时请附上这份清单。
- 在16 KB page size的设备上后端不可用：`libMNN_htpops.so`同样需要16 KB对齐，用`readelf -l libMNN_htpops.so`确认LOAD段的Align是`0x4000`。`htp-ops-lib`的android目标没有显式设置该链接参数，实际结果取决于编译用的NDK版本（r28及以上默认已对齐）。stub加载失败时后端不会注册，表现为整体回退CPU。

### 延伸阅读

- `source/backend/hexagon/htp-ops-lib/README.md`：FP16 HMX的实现背景、PWL激活，以及该算子库对应的论文。
- [“Hexagon HVX FP16 PWL 激活函数”](../perf/hexagon_pwl_activations.md)：分段线性激活的设计与实测。


## CoreML
适用于 Mac / iOS / iPad

### CoreML 后端编译
1. 编译 MNN 时打开编译宏 MNN_COREML ：-DMNN_COREML=ON
2. 编译App / 可执行程序时，增加链接 CoreML.framework

### CoreML 后端使用
backend type设置成：MNN_FORWARD_NN

## NNAPI
适用于 Android 系统，高通/联发科芯片

### NNAPI 后端编译
打开编译宏 MNN_NNAPI 即可
```
cd ${MNN}
cd project/android
mkdir build && cd build
../build_64.sh -DMNN_USE_LOGCAT=ON -DMNN_NNAPI=ON
```

### NNAPI 后端使用
backend type设置成：MNN_FORWARD_NN


## 华为 HIAI
适用于 Android 系统， Kirlin芯片

### HIAI 环境准备
1. 从如下链接下载 DDK
https://developer.huawei.com/consumer/cn/doc/hiai-Library/ddk-download-0000001053590180



2. 拷贝相对应的so和include文件到 hiai/3rdParty 目录下，如果没有3rdParty目录，新建一个：

```
mkdir ${MNN}/source/backend/hiai/3rdParty
cp -r ${DDK}/lib ${MNN}/source/backend/hiai/3rdParty/armeabi-v7a
cp -r ${DDK}/lib64 ${MNN}/source/backend/hiai/3rdParty/arm64-v8a
cp -r ${DDK}/include ${MNN}/source/backend/hiai/3rdParty/include
```

### HIAI 编译执行
1. cmake 参数打开npu开关： -DMNN_NPU=ON
2. backend type设置成：MNN_FORWARD_USER_0
3. 根据构建参数选择产物：
   - `MNN_NPU_BACKENDS_SHARED=ON`：生成 `libMNN_Backend_HiAI.so`，
     通过 `MNN::loadHiAIBackendPlugin` 显式加载。
   - `MNN_NPU_BACKENDS_SHARED=OFF`、`MNN_SEP_BUILD=ON`：生成 `libMNN_NPU.so`。
   - 两者均为 `OFF`：HiAI 后端编入 `libMNN.so`。
4. 创建 HiAI Runtime 时，通过 `dlopen()` 加载运行库并用 `dlsym()` 解析接口。
   HCL V600 路径的 HiAI C++ 图接口从 `libhiai_ir.so` 解析；V320 完整路径还需
   `libhiai.so` 和 `libhiai_ir_build.so`。这些库不是上述 MNN 产物的
   `DT_NEEDED` 依赖。
5. 将 HiAI 运行库放在动态链接器可搜索的路径中，或通过
   `MNNHiAIBackendConfigV1::runtimeLibraryDirectory` 指定目录，再调用
   `MNN::createHiAISession`。库加载失败会报告 Runtime 初始化错误。

## RKNN
适用于 Rockchip RKNPU 平台。当前接入方式不是在线逐算子构图，而是同一份 ONNX 在 Host 侧同时生成：
- 包装后的 `.mnn`
- sidecar `.rknn`

其中 `.mnn` 内部保留 `Input + Plugin(type="RKNN")` 包装图，运行时由 MNN 的 CPU Plugin 框架调用 RKNN C API 执行 `.rknn`。

### RKNN 后端整体介绍

- Host 侧通过 `MNNConvert --rknn` 完成双产物生成，不走 `compilefornpu` 的 `MNN -> NPU` 逐算子编译链路。
- Device 侧通过 MNN 的 CPU Plugin 框架调用 RKNN C API 加载 `.rknn` 并执行；应用侧 Session backend 仍使用 `MNN_FORWARD_CPU`。
- RKNN backend 读取 runtime 库路径、转换脚本路径、目标平台等信息时，不做硬编码，全部从环境变量读取；缺失时直接报 `MNN_ERROR`。

更完整的 RKNN 说明、包内容、示例代码与板端运行方式，请参考：
- `source/backend/rknn/README.md`
  - 包含 Host 转换、aarch64 交叉编译、Plugin 运行机制、独立示例代码、板端包内容与运行方式。

### 编译

#### Host，编译带 RKNN 转换能力的 MNNConvert

需要开启：
- `-DMNN_BUILD_CONVERTER=ON`
- `-DMNN_RKNN_CONVERT_MODE=ON`

示例：

```bash
cmake -S ${MNN_ROOT} -B ${BUILD_DIR} \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_RKNN_CONVERT_MODE=ON

cmake --build ${BUILD_DIR} --target MNNConvert -j8
```

#### Device/Runtime，编译带 RKNN backend 的 MNN

需要开启：
- `-DMNN_RKNN=ON`
- `-DRKNN_API_INCLUDE_DIR=/path/to/rknn_api/include`

示例：

```bash
cmake -S ${MNN_ROOT} -B ${BUILD_DIR} \
  -DMNN_RKNN=ON \
  -DRKNN_API_INCLUDE_DIR=/path/to/rknn_api/include

cmake --build ${BUILD_DIR} --target MNN -j8
```

### Host，生成 RKNN 包装模型

调用 `MNNConvert --rknn` 前，必须设置以下环境变量：

- `MNN_RKNN_TARGET`
  - 例如 `rv1126b`
- `MNN_RKNN_PYTHON`
  - RKNN Toolkit 所在 Python 解释器
- `MNN_RKNN_SCRIPT`
  - ONNX 转 `.rknn` 的脚本路径
- `MNN_RKNN_OUTPUT_DIR`
  - `.rknn` 和 manifest 的输出目录

示例：

```bash
export MNN_RKNN_TARGET=rv1126b
export MNN_RKNN_PYTHON=/path/to/python
export MNN_RKNN_SCRIPT=/path/to/to_rknn.py
export MNN_RKNN_OUTPUT_DIR=/path/to/output/sidecar

${BUILD_DIR}/MNNConvert \
  -f ONNX \
  --modelFile model.onnx \
  --MNNModel model.mnn \
  --rknn
```

执行成功后会生成：
- `model.mnn`
  - RKNN wrapper 模型
- `${MNN_RKNN_OUTPUT_DIR}/model_<target>.rknn`
- `${MNN_RKNN_OUTPUT_DIR}/model.rknn.bundle.json`

### Device，运行

运行时必须设置：
- `MNN_RKNN_RUNTIME_LIB`
  - 指向目标板上的 `librknnrt.so`

并在创建 Session 时选择：
- backend type = `MNN_FORWARD_CPU`

**注意：在 RK 板上执行任何真正调用 NPU 的命令时，必须使用 `sudo`。**

如果 `.rknn` 路径在 wrapper `.mnn` 中是相对路径，则需要确保模型外部路径设置正确，使 MNN 能解析 sidecar 所在目录。

### 当前限制

- 当前 RKNN 路径执行 `Plugin(type="RKNN")` 节点，不支持逐算子 RKNN backend。
- 当前实现走 host buffer copy 路径，尚未做 zero-copy。
- 当前输出路径按 `float32` 处理。
- 当前主目标是板端运行；PC 侧如果没有可用的 x86 `librknnrt.so`，则不能直接用 MNN runtime 在 Host 上模拟执行 RKNN backend。
