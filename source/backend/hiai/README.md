# HiAI backend plugin

HiAI owns its optional plugin ABI in `include/MNNHiAIBackend.h`, its loader in
`include/MNNHiAIPlugin.hpp`, strict Session creation in
`include/MNNHiAISession.hpp`, and native-handle I/O contract in
`include/MNNHiAIIO.h`. These headers are installed under `MNN/`.

Build `libMNN_Backend_HiAI.so` with `MNN_NPU=ON` and
`MNN_NPU_BACKENDS_SHARED=ON`. Load it through
`MNN::loadHiAIBackendPlugin`, put `MNNHiAIBackendConfigV1` in
`BackendConfig::sharedContext`, set both schedule types to
`MNN_FORWARD_USER_0`, and call `MNN::createHiAISession`.

The plugin keeps the baseline automatic HiAI registration path separate from
the explicit plugin path. HiAI runtime libraries are resolved with
`dlopen`/`dlsym`; they are not `DT_NEEDED` dependencies of public `libMNN.so`.
The application owns diagnostics and native I/O contexts for the complete
Runtime/Session lifetime.
