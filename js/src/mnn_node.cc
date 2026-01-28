#include <napi.h>
#include <MNN/Interpreter.hpp>
#include "interpreter.h"
#include "session.h"
#include "tensor.h"
#include "utils.h"
#include "llm.h"

namespace mnn_node {

// Module initialization
Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    // Export classes
    InterpreterWrap::Init(env, exports);
    SessionWrap::Init(env, exports);
    TensorWrap::Init(env, exports);
    LlmWrap::Init(env, exports);
    
    // Export version
    exports.Set("version", Napi::String::New(env, MNN::getVersion()));
    
    // Export forward types
    Napi::Object forwardType = Napi::Object::New(env);
    forwardType.Set("CPU", Napi::Number::New(env, MNN_FORWARD_CPU));
    forwardType.Set("METAL", Napi::Number::New(env, MNN_FORWARD_METAL));
    forwardType.Set("OPENCL", Napi::Number::New(env, MNN_FORWARD_OPENCL));
    forwardType.Set("OPENGL", Napi::Number::New(env, MNN_FORWARD_OPENGL));
    forwardType.Set("VULKAN", Napi::Number::New(env, MNN_FORWARD_VULKAN));
    forwardType.Set("NN", Napi::Number::New(env, MNN_FORWARD_NN));
    forwardType.Set("CUDA", Napi::Number::New(env, MNN_FORWARD_CUDA));
    exports.Set("ForwardType", forwardType);
    
    // Export error codes
    Napi::Object errorCode = Napi::Object::New(env);
    errorCode.Set("NO_ERROR", Napi::Number::New(env, MNN::NO_ERROR));
    errorCode.Set("OUT_OF_MEMORY", Napi::Number::New(env, MNN::OUT_OF_MEMORY));
    errorCode.Set("NOT_SUPPORT", Napi::Number::New(env, MNN::NOT_SUPPORT));
    errorCode.Set("COMPUTE_SIZE_ERROR", Napi::Number::New(env, MNN::COMPUTE_SIZE_ERROR));
    errorCode.Set("NO_EXECUTION", Napi::Number::New(env, MNN::NO_EXECUTION));
    exports.Set("ErrorCode", errorCode);
    
    return exports;
}

NODE_API_MODULE(mnn, InitAll)

} // namespace mnn_node
