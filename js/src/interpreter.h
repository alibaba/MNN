//
//  Interpreter wrapper for MNN Node.js bindings
//
#ifndef MNN_NODE_INTERPRETER_H
#define MNN_NODE_INTERPRETER_H

#include "common.h"

namespace mnn_node {

class InterpreterWrap : public Napi::ObjectWrap<InterpreterWrap> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    InterpreterWrap(const Napi::CallbackInfo& info);
    ~InterpreterWrap();

private:
    // Static methods to create interpreter
    static Napi::Value CreateFromFile(const Napi::CallbackInfo& info);
    static Napi::Value CreateFromBuffer(const Napi::CallbackInfo& info);
    
    // Instance methods
    Napi::Value CreateSession(const Napi::CallbackInfo& info);
    Napi::Value ResizeSession(const Napi::CallbackInfo& info);
    Napi::Value RunSession(const Napi::CallbackInfo& info);
    Napi::Value GetSessionInput(const Napi::CallbackInfo& info);
    Napi::Value GetSessionOutput(const Napi::CallbackInfo& info);
    Napi::Value Release(const Napi::CallbackInfo& info);
    Napi::Value GetModelVersion(const Napi::CallbackInfo& info);

    MNN::Interpreter* interpreter_;
    
    static Napi::FunctionReference constructor;
};

} // namespace mnn_node

#endif // MNN_NODE_INTERPRETER_H
