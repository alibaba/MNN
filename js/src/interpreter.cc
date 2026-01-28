//
//  Interpreter wrapper implementation
//
#include "interpreter.h"
#include "session.h"
#include "tensor.h"
#include "utils.h"

namespace mnn_node {

Napi::FunctionReference InterpreterWrap::constructor;

Napi::Object InterpreterWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Interpreter", {
        StaticMethod("createFromFile", &InterpreterWrap::CreateFromFile),
        StaticMethod("createFromBuffer", &InterpreterWrap::CreateFromBuffer),
        InstanceMethod("createSession", &InterpreterWrap::CreateSession),
        InstanceMethod("resizeSession", &InterpreterWrap::ResizeSession),
        InstanceMethod("runSession", &InterpreterWrap::RunSession),
        InstanceMethod("getSessionInput", &InterpreterWrap::GetSessionInput),
        InstanceMethod("getSessionOutput", &InterpreterWrap::GetSessionOutput),
        InstanceMethod("release", &InterpreterWrap::Release),
        InstanceMethod("getModelVersion", &InterpreterWrap::GetModelVersion),
    });
    
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    
    exports.Set("Interpreter", func);
    return exports;
}

InterpreterWrap::InterpreterWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<InterpreterWrap>(info), interpreter_(nullptr) {
}

InterpreterWrap::~InterpreterWrap() {
    if (interpreter_ != nullptr) {
        MNN::Interpreter::destroy(interpreter_);
        interpreter_ = nullptr;
    }
}

Napi::Value InterpreterWrap::CreateFromFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        THROW_ERROR(env, "Expected at least 1 argument");
        return env.Undefined();
    }
    
    if (!info[0].IsString()) {
        THROW_TYPE_ERROR(env, "Path must be a string");
        return env.Undefined();
    }
    
    std::string path = info[0].As<Napi::String>().Utf8Value();
    MNN::Interpreter* net = MNN::Interpreter::createFromFile(path.c_str());
    
    if (net == nullptr) {
        THROW_ERROR(env, "Failed to create interpreter from file");
        return env.Undefined();
    }
    
    Napi::Object obj = constructor.New({});
    InterpreterWrap* wrap = Napi::ObjectWrap<InterpreterWrap>::Unwrap(obj);
    wrap->interpreter_ = net;
    
    return obj;
}

Napi::Value InterpreterWrap::CreateFromBuffer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        THROW_ERROR(env, "Expected at least 1 argument");
        return env.Undefined();
    }
    
    void* bufferData = nullptr;
    size_t size = 0;
    
    if (info[0].IsArrayBuffer()) {
        Napi::ArrayBuffer arrayBuffer = info[0].As<Napi::ArrayBuffer>();
        bufferData = arrayBuffer.Data();
        size = arrayBuffer.ByteLength();
    } else if (info[0].IsTypedArray()) {
        Napi::TypedArray typedArray = info[0].As<Napi::TypedArray>();
        Napi::ArrayBuffer arrayBuffer = typedArray.ArrayBuffer();
        bufferData = static_cast<uint8_t*>(arrayBuffer.Data()) + typedArray.ByteOffset();
        size = typedArray.ByteLength();
    } else if (info[0].IsBuffer()) { // Node.js Buffer
         Napi::Buffer<uint8_t> buffer = info[0].As<Napi::Buffer<uint8_t>>();
         bufferData = buffer.Data();
         size = buffer.Length();
    } else {
         THROW_TYPE_ERROR(env, "Buffer be an ArrayBuffer, TypedArray or Buffer");
         return env.Undefined();
    }

    MNN::Interpreter* net = MNN::Interpreter::createFromBuffer(bufferData, size);
    if (net == nullptr) {
        THROW_ERROR(env, "Failed to create interpreter from buffer");
        return env.Undefined();
    }
    
    Napi::Object obj = constructor.New({});
    InterpreterWrap* wrap = Napi::ObjectWrap<InterpreterWrap>::Unwrap(obj);
    wrap->interpreter_ = net;
    
    return obj;
}

Napi::Value InterpreterWrap::CreateSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (interpreter_ == nullptr) {
        THROW_ERROR(env, "Interpreter is released");
        return env.Undefined();
    }
    
    MNN::ScheduleConfig config;
    if (info.Length() > 0 && info[0].IsObject()) {
        GetScheduleConfig(info[0].As<Napi::Object>(), config);
    }
    
    MNN::Session* session = interpreter_->createSession(config);
    if (session == nullptr) {
        THROW_ERROR(env, "Failed to create session");
        return env.Undefined();
    }
    
    return SessionWrap::NewInstance(env, session);
}

Napi::Value InterpreterWrap::ResizeSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        THROW_ERROR(env, "Expected at least 1 argument");
        return env.Undefined();
    }
    
    if (interpreter_ == nullptr) {
        THROW_ERROR(env, "Interpreter is released");
        return env.Undefined();
    }
    
    Napi::Object sessionObj = info[0].As<Napi::Object>();
    SessionWrap* sessionWrap = Napi::ObjectWrap<SessionWrap>::Unwrap(sessionObj);
    MNN::Session* session = sessionWrap->GetSession();
    
    interpreter_->resizeSession(session);
    return env.Undefined();
}

Napi::Value InterpreterWrap::RunSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        THROW_ERROR(env, "Expected at least 1 argument");
        return env.Undefined();
    }
    
    if (interpreter_ == nullptr) {
        THROW_ERROR(env, "Interpreter is released");
        return env.Undefined();
    }
    
    Napi::Object sessionObj = info[0].As<Napi::Object>();
    SessionWrap* sessionWrap = Napi::ObjectWrap<SessionWrap>::Unwrap(sessionObj);
    MNN::Session* session = sessionWrap->GetSession();
    
    MNN::ErrorCode code = interpreter_->runSession(session);
    return Napi::Number::New(env, static_cast<int>(code));
}

Napi::Value InterpreterWrap::GetSessionInput(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        THROW_ERROR(env, "Expected at least 1 argument");
        return env.Undefined();
    }
    
    if (interpreter_ == nullptr) {
        THROW_ERROR(env, "Interpreter is released");
        return env.Undefined();
    }
    
    Napi::Object sessionObj = info[0].As<Napi::Object>();
    SessionWrap* sessionWrap = Napi::ObjectWrap<SessionWrap>::Unwrap(sessionObj);
    MNN::Session* session = sessionWrap->GetSession();
    
    const char* name = nullptr;
    std::string nameStr;
    if (info.Length() > 1 && info[1].IsString()) {
        nameStr = info[1].As<Napi::String>().Utf8Value();
        name = nameStr.c_str();
    }
    
    MNN::Tensor* tensor = interpreter_->getSessionInput(session, name);
    if (tensor == nullptr) {
        return env.Null();
    }
    
    // Return Tensor wrapper (not owned by wrapper, owned by MNN)
    return TensorWrap::NewInstance(env, tensor, false);
}

Napi::Value InterpreterWrap::GetSessionOutput(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        THROW_ERROR(env, "Expected at least 1 argument");
        return env.Undefined();
    }
    
    if (interpreter_ == nullptr) {
        THROW_ERROR(env, "Interpreter is released");
        return env.Undefined();
    }
    
    Napi::Object sessionObj = info[0].As<Napi::Object>();
    SessionWrap* sessionWrap = Napi::ObjectWrap<SessionWrap>::Unwrap(sessionObj);
    MNN::Session* session = sessionWrap->GetSession();
    
    const char* name = nullptr;
    std::string nameStr;
    if (info.Length() > 1 && info[1].IsString()) {
        nameStr = info[1].As<Napi::String>().Utf8Value();
        name = nameStr.c_str();
    }
    
    MNN::Tensor* tensor = interpreter_->getSessionOutput(session, name);
    if (tensor == nullptr) {
        return env.Null();
    }
    
    // Return Tensor wrapper (not owned by wrapper, owned by MNN)
    return TensorWrap::NewInstance(env, tensor, false);
}

Napi::Value InterpreterWrap::Release(const Napi::CallbackInfo& info) {
    if (interpreter_ != nullptr) {
        MNN::Interpreter::destroy(interpreter_);
        interpreter_ = nullptr;
    }
    return info.Env().Undefined();
}

Napi::Value InterpreterWrap::GetModelVersion(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (interpreter_ == nullptr) {
        THROW_ERROR(env, "Interpreter is released");
        return env.Undefined();
    }
    const char* version = interpreter_->getModelVersion();
    return Napi::String::New(env, version ? version : "");
}

} // namespace mnn_node
