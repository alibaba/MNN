#ifndef MNN_NODE_LLM_H
#define MNN_NODE_LLM_H

#include <napi.h>
#include <llm/llm.hpp>

namespace mnn_node {

class LlmWrap : public Napi::ObjectWrap<LlmWrap> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    LlmWrap(const Napi::CallbackInfo& info);
    ~LlmWrap();

private:
    static Napi::FunctionReference constructor;
    
    // JS Methods
    Napi::Value Load(const Napi::CallbackInfo& info);
    Napi::Value ApplyChatTemplate(const Napi::CallbackInfo& info);
    Napi::Value Response(const Napi::CallbackInfo& info);
    Napi::Value Generate(const Napi::CallbackInfo& info);
    Napi::Value SetConfig(const Napi::CallbackInfo& info);
    
    // Static factory for MNN.llm.create
    static Napi::Value Create(const Napi::CallbackInfo& info);

    MNN::Transformer::Llm* llm_ = nullptr;
};

} // namespace mnn_node

#endif // MNN_NODE_LLM_H
