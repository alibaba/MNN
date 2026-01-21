//
//  Session wrapper implementation
//
#include "session.h"

namespace mnn_node {

Napi::FunctionReference SessionWrap::constructor;

Napi::Object SessionWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Session", {});
    
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    // Don't set instance data if we have multiple classes, as it's singular.
    // Instead we rely on static member.
    
    exports.Set("Session", func);
    return exports;
}

SessionWrap::SessionWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<SessionWrap>(info), session_(nullptr) {
}

Napi::Object SessionWrap::NewInstance(Napi::Env env, MNN::Session* session) {
    Napi::Object obj = constructor.New({});
    SessionWrap* wrap = Napi::ObjectWrap<SessionWrap>::Unwrap(obj);
    wrap->session_ = session;
    return obj;
}

SessionWrap::~SessionWrap() {
    // Session is managed by Interpreter, don't delete here
}

} // namespace mnn_node
