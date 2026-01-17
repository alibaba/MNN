//
//  Tensor wrapper for MNN Node.js bindings
//
#ifndef MNN_NODE_TENSOR_H
#define MNN_NODE_TENSOR_H

#include "common.h"

namespace mnn_node {

class TensorWrap : public Napi::ObjectWrap<TensorWrap> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    TensorWrap(const Napi::CallbackInfo& info);
    ~TensorWrap();
    
    // Create from existing tensor (not owned)
    static Napi::Object NewInstance(Napi::Env env, MNN::Tensor* tensor, bool own = false);
    
    // Get the wrapped tensor
    MNN::Tensor* GetTensor() const { return tensor_; }
    
private:
    // Instance methods
    Napi::Value GetShape(const Napi::CallbackInfo& info);
    Napi::Value GetDataType(const Napi::CallbackInfo& info);
    Napi::Value GetData(const Napi::CallbackInfo& info);
    Napi::Value CopyFrom(const Napi::CallbackInfo& info);
    Napi::Value GetHost(const Napi::CallbackInfo& info);
    Napi::Value GetElementSize(const Napi::CallbackInfo& info);
    
    MNN::Tensor* tensor_;
    bool owner_;  // Whether this wrapper owns the tensor
    
    static Napi::FunctionReference constructor;
};

} // namespace mnn_node

#endif // MNN_NODE_TENSOR_H
