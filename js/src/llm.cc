#include "llm.h"
#include <sstream>
#include <iostream>

namespace mnn_node {

Napi::FunctionReference LlmWrap::constructor;

Napi::Object LlmWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Llm", {
        InstanceMethod("load", &LlmWrap::Load),
        InstanceMethod("response", &LlmWrap::Response),
        InstanceMethod("generate", &LlmWrap::Generate),
        InstanceMethod("applyChatTemplate", &LlmWrap::ApplyChatTemplate),
        InstanceMethod("setConfig", &LlmWrap::SetConfig),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    // Create the MNN.llm object
    Napi::Object llmObj = Napi::Object::New(env);
    llmObj.Set("create", Napi::Function::New(env, Create, "create"));
    
    exports.Set("llm", llmObj);

    return exports;
}

LlmWrap::LlmWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LlmWrap>(info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
         Napi::TypeError::New(env, "String expected").ThrowAsJavaScriptException();
         return;
    }
    std::string configPath = info[0].As<Napi::String>();
    llm_ = MNN::Transformer::Llm::createLLM(configPath);
    if (!llm_) {
        Napi::Error::New(env, "Failed to create LLM with path: " + configPath).ThrowAsJavaScriptException();
    }
}

LlmWrap::~LlmWrap() {
    if (llm_) {
        MNN::Transformer::Llm::destroy(llm_);
        llm_ = nullptr;
    }
}

Napi::Value LlmWrap::Create(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Usage: create(configPath)").ThrowAsJavaScriptException();
        return env.Null();
    }
    // Call constructor
    return constructor.New({info[0]});
}

Napi::Value LlmWrap::Load(const Napi::CallbackInfo& info) {
    if (!llm_) return info.Env().Null();
    
    // Running load might take time, ideally async but simpler for now
    bool res = llm_->load();
    if (!res) {
        Napi::Error::New(info.Env(), "Failed to load LLM").ThrowAsJavaScriptException();
    }
    return Napi::Boolean::New(info.Env(), res);
}

Napi::Value LlmWrap::Response(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!llm_) return env.Null();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Usage: response(query, [history])").ThrowAsJavaScriptException();
        return env.Null();
    }
    std::string query = info[0].As<Napi::String>();
    
    // Handle history argument (default true)
    bool history = true;
    if (info.Length() > 1 && info[1].IsBoolean()) {
        history = info[1].As<Napi::Boolean>().Value();
    }
    
    if (!history) {
        llm_->reset();
    }
    
    std::stringstream ss;
    llm_->response(query, &ss);
    
    return Napi::String::New(env, ss.str());
}

Napi::Value LlmWrap::Generate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!llm_) return env.Null();
    
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Usage: generate(tokens)").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    std::vector<int> input_ids;
    if (info[0].IsArray()) {
        Napi::Array arr = info[0].As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); i++) {
            Napi::Value v = arr[i];
            if (v.IsNumber()) {
                input_ids.push_back(v.As<Napi::Number>().Int32Value());
            }
        }
    } else {
         Napi::TypeError::New(env, "Array of integers expected").ThrowAsJavaScriptException();
         return env.Null();
    }
    
    std::vector<int> output_ids = llm_->generate(input_ids);
    
    Napi::Array res = Napi::Array::New(env, output_ids.size());
    for (size_t i = 0; i < output_ids.size(); i++) {
        res[i] = Napi::Number::New(env, output_ids[i]);
    }
    return res;
}

Napi::Value LlmWrap::ApplyChatTemplate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!llm_) return env.Null();
    if (info.Length() < 1 || !info[0].IsString()) return env.Null();
    std::string content = info[0].As<Napi::String>();
    std::string res = llm_->apply_chat_template(content);
    return Napi::String::New(env, res);
}

Napi::Value LlmWrap::SetConfig(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!llm_) return env.Null();
    if (info.Length() < 1 || !info[0].IsString()) return env.Null();
    std::string content = info[0].As<Napi::String>();
    llm_->set_config(content);
    return env.Null();
}

} // namespace mnn_node
