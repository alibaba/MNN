#include "HexagonExecution.hpp"
#ifdef MNN_HEXAGON_ASAN
#include "backend/hexagon/backend/HexagonBackend.hpp"
#include "backend/hexagon/backend/HexagonRuntime.hpp"
#endif

namespace MNN {

HexagonExecution::HexagonExecution(Backend* backend) : Execution(backend) {
}

ErrorCode HexagonExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mCmd.clear();
    mCmd.reserve(32);
    auto code = onBuildCmd(inputs, outputs, mCmd);
#ifdef MNN_HEXAGON_ASAN
    auto hexBackend = static_cast<HexagonBackend*>(backend());
    auto runtime = static_cast<const HexagonRuntime*>(hexBackend->getRuntime());
    if (runtime != nullptr && !runtime->asanCheckAllBuffers("after HexagonExecution::onResize")) {
        return INVALID_VALUE;
    }
#endif
    return code;
}

ErrorCode HexagonExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (!mValid) {
        return NOT_SUPPORT;
    }

#ifdef MNN_HEXAGON_ASAN
    auto hexBackend = static_cast<HexagonBackend*>(backend());
    auto runtime = static_cast<const HexagonRuntime*>(hexBackend->getRuntime());
    if (runtime != nullptr && !runtime->asanCheckAllBuffers("before HexagonExecution::onExecute")) {
        return INVALID_VALUE;
    }
#endif
    for (auto& cmd : mCmd) {
        cmd.execute();
    }
#ifdef MNN_HEXAGON_ASAN
    if (runtime != nullptr && !runtime->asanCheckAllBuffers("after HexagonExecution::onExecute")) {
        return INVALID_VALUE;
    }
#endif
    return NO_ERROR;
}

} // namespace MNN
