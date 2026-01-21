//
//  Session wrapper for MNN Node.js bindings
//
#ifndef MNN_NODE_SESSION_H
#define MNN_NODE_SESSION_H

#include "common.h"

namespace mnn_node {

class SessionWrap : public Napi::ObjectWrap<SessionWrap> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    SessionWrap(const Napi::CallbackInfo& info);
    ~SessionWrap();
    
    // Get the wrapped session
    MNN::Session* GetSession() const { return session_; }
    
    static Napi::Object NewInstance(Napi::Env env, MNN::Session* session);

private:
    MNN::Session* session_;
    
    static Napi::FunctionReference constructor;
    
    friend class InterpreterWrap;
};

} // namespace mnn_node

#endif // MNN_NODE_SESSION_H
