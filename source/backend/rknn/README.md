# RKNN Backend

This directory contains the RKNN integration for MNN.

This file intentionally keeps the instructions generic.
For one machine-specific, real-path compilation and deployment example, see the external project README used in this integration workflow.

Current design:
- Converter side generates two artifacts from the same ONNX model:
  - a wrapper `.mnn` model containing `Plugin(type="RKNN")`
  - a sidecar `.rknn` model plus bundle manifest
- Runtime side executes `Plugin("RKNN")` through the MNN CPU Plugin framework.
- There is no `MNN_FORWARD_USER_2` RKNN runtime path anymore.
- Application-side session backend remains `MNN_FORWARD_CPU`.

## 1. Host build for `MNNConvert --rknn`

Build a host `MNNConvert` with plugin support and RKNN converter support enabled:

```bash
cmake -S /path/to/MNN-Agent -B /path/to/MNN-Agent/build-linux \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_WITH_PLUGIN=ON \
  -DMNN_RKNN=ON \
  -DMNN_RKNN_CONVERT_MODE=ON \
  -DRKNN_API_INCLUDE_DIR=/path/to/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include

cmake --build /path/to/MNN-Agent/build-linux --target MNN MNNConvert -j8
```

## 2. Generate wrapper `.mnn` + sidecar `.rknn`

Before running `MNNConvert --rknn`, export these environment variables:

```bash
export MNN_RKNN_TARGET=rv1126b
export MNN_RKNN_PYTHON=/path/to/python
export MNN_RKNN_SCRIPT=/path/to/to_rknn.py
export MNN_RKNN_OUTPUT_DIR=/path/to/output/sidecar
```

Example:

```bash
/path/to/MNN-Agent/build-linux/MNNConvert \
  -f ONNX \
  --modelFile /path/to/model.onnx \
  --MNNModel /path/to/model.mnn \
  --rknn
```

Expected outputs:
- `/path/to/model.mnn`
- `${MNN_RKNN_OUTPUT_DIR}/model_<target>.rknn`
- `${MNN_RKNN_OUTPUT_DIR}/model.rknn.bundle.json`

The generated wrapper `.mnn` contains:
- `Input` ops for original inputs
- one `Plugin(type="RKNN")` op
- plugin attrs including:
  - `model_path`
  - `bundle_manifest`
  - `target`
  - `inputs`
  - `outputs`
  - `o_0`, `o_1`, ... for output shape metadata

Important:
- `model_path` and `bundle_manifest` are emitted as relative file names.
- The validated deployment layout is: wrapper `.mnn`, sidecar `.rknn`, and bundle `.json` in the same target directory.

## 3. Cross compile runtime for Linux aarch64 / ARMv8

Example cross build using the system `aarch64-linux-gnu` toolchain.
This builds the target-side runtime libraries; `MNNConvert` itself is usually only needed on the host.

```bash
cmake -S /path/to/MNN-Agent -B /path/to/MNN-Agent/build-linux-aarch64-gnu \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
  -DCMAKE_C_FLAGS='-march=armv8-a' \
  -DCMAKE_CXX_FLAGS='-march=armv8-a' \
  -DMNN_WITH_PLUGIN=ON \
  -DMNN_RKNN=ON \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_TOOLS=ON \
  -DRKNN_API_INCLUDE_DIR=/path/to/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include

cmake --build /path/to/MNN-Agent/build-linux-aarch64-gnu --target MNN MNN_Express -j8
```

Notes:
- `MNN_WITH_PLUGIN=ON` is required because RKNN is implemented as a Plugin op.
- `MNN_RKNN=ON` pulls in the RKNN Plugin kernels.
- `RKNN_API_INCLUDE_DIR` must point to the directory containing `rknn_api.h`.
- The RKNN runtime library is loaded at runtime via `dlopen`, not linked as a hard dependency.

## 4. Target runtime usage

On the target board, export the RKNN runtime library path:

```bash
export MNN_RKNN_RUNTIME_LIB=/path/to/librknnrt.so
```

The wrapper `.mnn` should be deployed together with its sidecar `.rknn` and bundle manifest in the same directory on target.

Important:
- On RK boards, commands that actually execute NPU code should be run with `sudo`.

Runtime behavior:
- MNN loads the wrapper `.mnn`
- `Plugin(type="RKNN")` is created by the CPU Plugin framework
- the plugin loads the `.rknn` sidecar using RKNN C API
- application-side MNN backend is still `MNN_FORWARD_CPU`
- if the RKNN model expects `NHWC` but the incoming MNN tensor is `NCHW`, the plugin converts layout automatically
- if the incoming tensor is already `NHWC`, no extra layout conversion is done
- backend-side RKNN profiling can be enabled through the public hint path:
  - `Interpreter::setSessionHint(Interpreter::RKNN_PROFILE, 1)` or `RuntimeManager::setHint(Interpreter::RKNN_PROFILE, 1)`
  - retrieve the exported profile text through `getSessionInfo(..., Interpreter::BACKEND_PROFILE, &ptr)` or `RuntimeManager::getInfo(Interpreter::BACKEND_PROFILE, &ptr)`
  - because the profile is exposed as plain text, applications can print it or write it directly to a file

## 5. Current limitations

- This is a sidecar-subgraph path, not a per-op RKNN backend.
- Current implementation uses host buffer copies; zero-copy is not implemented.
- Current output copy path assumes float32 outputs from RKNN runtime.
- Input layout auto-conversion currently handles the common `NCHW -> NHWC` case for 4D tensors only, and only when the RKNN model explicitly expects `NHWC`.
- Host-side PC simulation through MNN runtime requires an x86 RKNN runtime library; usually this path is meant for target boards.

## 6. Code examples

### 6.1 Minimal C++ example with `Interpreter`

This example loads the wrapper `.mnn` generated by `MNNConvert --rknn` and runs it through the normal CPU backend. Internally, the `Plugin("RKNN")` node will call the RKNN C API.

```cpp
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

int main() {
    const char* model_path = "/data/local/tmp/rejshand_epoch200_b1_nogridsample.mnn";

    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model_path));
    if (!net) {
        std::fprintf(stderr, "createFromFile failed\n");
        return 1;
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;

    MNN::BackendConfig backendConfig;
    config.backendConfig = &backendConfig;

    auto session = net->createSession(config);
    if (!session) {
        std::fprintf(stderr, "createSession failed\n");
        return 1;
    }

    auto input = net->getSessionInput(session, "image");
    if (!input) {
        std::fprintf(stderr, "getSessionInput failed\n");
        return 1;
    }

    net->resizeTensor(input, {1, 3, 224, 224});
    net->resizeSession(session);

    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    std::memset(hostInput.host<float>(), 0, hostInput.size());
    input->copyFromHostTensor(&hostInput);

    if (net->runSession(session) != 0) {
        std::fprintf(stderr, "runSession failed\n");
        return 1;
    }

    auto uv = net->getSessionOutput(session, "uv");
    auto vertices = net->getSessionOutput(session, "vertices");
    if (!uv || !vertices) {
        std::fprintf(stderr, "getSessionOutput failed\n");
        return 1;
    }

    MNN::Tensor uvHost(uv, MNN::Tensor::CAFFE);
    MNN::Tensor verticesHost(vertices, MNN::Tensor::CAFFE);
    uv->copyToHostTensor(&uvHost);
    vertices->copyToHostTensor(&verticesHost);

    auto uvPtr = uvHost.host<float>();
    auto vPtr = verticesHost.host<float>();
    std::printf("uv[0] = %f, %f\n", uvPtr[0], uvPtr[1]);
    std::printf("vertices[0] = %f, %f, %f\n", vPtr[0], vPtr[1], vPtr[2]);
    return 0;
}
```

Typical build command on target:

```bash
aarch64-linux-gnu-g++ -O2 -std=c++11 demo_rknn_mnn.cpp \
  -I/path/to/MNN-Agent/include \
  -L/path/to/mnn/libs -lMNN -o demo_rknn_mnn
```

At runtime on board:

```bash
export LD_LIBRARY_PATH=/path/to/mnn/libs:$LD_LIBRARY_PATH
export MNN_RKNN_RUNTIME_LIB=/path/to/librknnrt.so
./demo_rknn_mnn
```

### 6.2 Minimal `Module` example

If you prefer the Express / Module API, load the same wrapper `.mnn` with `MNN_FORWARD_CPU`.

```cpp
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>

#include <cstdio>
#include <memory>
#include <vector>

using namespace MNN::Express;

int main() {
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;

    std::shared_ptr<MNN::Executor::RuntimeManager> rtmgr(MNN::Executor::RuntimeManager::createRuntimeManager(config));
    if (!rtmgr) {
        std::fprintf(stderr, "createRuntimeManager failed\n");
        return 1;
    }

    std::vector<std::string> inputs = {"image"};
    std::vector<std::string> outputs = {"uv", "vertices"};
    auto module = Module::load(inputs, outputs, "/data/local/tmp/rejshand_epoch200_b1_nogridsample.mnn", rtmgr);
    if (!module) {
        std::fprintf(stderr, "Module::load failed\n");
        return 1;
    }

    auto image = _Input({1, 3, 224, 224}, NCHW, halide_type_of<float>());
    auto imagePtr = image->writeMap<float>();
    for (int i = 0; i < 1 * 3 * 224 * 224; ++i) {
        imagePtr[i] = 0.0f;
    }

    auto outputsVar = module->onForward({image});
    if (outputsVar.size() != 2) {
        std::fprintf(stderr, "unexpected output size: %zu\n", outputsVar.size());
        return 1;
    }

    auto uvInfo = outputsVar[0]->getInfo();
    auto verticesInfo = outputsVar[1]->getInfo();
    if (!uvInfo || !verticesInfo) {
        std::fprintf(stderr, "output info is null\n");
        return 1;
    }

    auto uv = outputsVar[0]->readMap<float>();
    auto vertices = outputsVar[1]->readMap<float>();
    std::printf("uv[0] = %f, %f\n", uv[0], uv[1]);
    std::printf("vertices[0] = %f, %f, %f\n", vertices[0], vertices[1], vertices[2]);
    return 0;
}
```

Runtime requirements are the same:

```bash
export LD_LIBRARY_PATH=/path/to/mnn/libs:$LD_LIBRARY_PATH
export MNN_RKNN_RUNTIME_LIB=/path/to/librknnrt.so
./demo_rknn_module
```

## 7. Notes

- Keep this README generic. Put machine-specific paths, standalone example source files, and one-off deployment commands in the external example project README instead.
- The standalone example program is intentionally kept outside the MNN source tree.
