//
//  Common definitions for MNN Node.js bindings
//
#ifndef MNN_NODE_COMMON_H
#define MNN_NODE_COMMON_H

#include <napi.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <string>
#include <vector>

namespace mnn_node {

// Forward declarations
class InterpreterWrap;
class SessionWrap;
class TensorWrap;

// Helper macros for error handling
#define THROW_ERROR(env, msg) \
    Napi::Error::New(env, msg).ThrowAsJavaScriptException()

#define THROW_TYPE_ERROR(env, msg) \
    Napi::TypeError::New(env, msg).ThrowAsJavaScriptException()

#define CHECK_ARG_COUNT(info, expected) \
    if (info.Length() < expected) { \
        THROW_ERROR(info.Env(), "Expected at least " #expected " arguments"); \
        return info.Env().Undefined(); \
    }

// Constants
static const int MAX_DIMS = 6;

} // namespace mnn_node

#endif // MNN_NODE_COMMON_H
