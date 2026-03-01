//
//  Tensor wrapper implementation
//
#include "tensor.h"
#include "utils.h"
#include <cstring>

namespace mnn_node {

Napi::FunctionReference TensorWrap::constructor;

Napi::Object TensorWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Tensor", {
        InstanceMethod("getShape", &TensorWrap::GetShape),
        InstanceMethod("getDataType", &TensorWrap::GetDataType),
        InstanceMethod("getData", &TensorWrap::GetData),
        InstanceMethod("copyFrom", &TensorWrap::CopyFrom),
        InstanceMethod("getHost", &TensorWrap::GetHost),
        InstanceMethod("getElementSize", &TensorWrap::GetElementSize),
    });
    
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    
    exports.Set("Tensor", func);
    return exports;
}

TensorWrap::TensorWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<TensorWrap>(info), tensor_(nullptr), owner_(false) {
}

TensorWrap::~TensorWrap() {
    if (owner_ && tensor_ != nullptr) {
        delete tensor_;
    }
}

Napi::Object TensorWrap::NewInstance(Napi::Env env, MNN::Tensor* tensor, bool own) {
    Napi::Object obj = constructor.New({});
    TensorWrap* wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
    wrap->tensor_ = tensor;
    wrap->owner_ = own;
    return obj;
}

Napi::Value TensorWrap::GetShape(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (tensor_ == nullptr) {
        THROW_ERROR(env, "Tensor is null");
        return env.Undefined();
    }
    
    std::vector<int> shape;
    for (int i = 0; i < tensor_->dimensions(); i++) {
        shape.push_back(tensor_->length(i));
    }
    
    return IntVectorToArray(env, shape);
}

Napi::Value TensorWrap::GetDataType(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (tensor_ == nullptr) {
        THROW_ERROR(env, "Tensor is null");
        return env.Undefined();
    }
    
    halide_type_t type = tensor_->getType();
    
    // Return data type as integer constant
    // 0: FLOAT, 1: INT32, 2: INT64, 3: UINT8
    if (type.code == halide_type_float && type.bits == 32) {
        return Napi::Number::New(env, 0);
    } else if (type.code == halide_type_int && type.bits == 32) {
        return Napi::Number::New(env, 1);
    } else if (type.code == halide_type_int && type.bits == 64) {
        return Napi::Number::New(env, 2);
    } else if (type.code == halide_type_uint && type.bits == 8) {
        return Napi::Number::New(env, 3);
    }
    
    return Napi::Number::New(env, 0); // default to FLOAT
}

Napi::Value TensorWrap::GetData(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (tensor_ == nullptr) {
        THROW_ERROR(env, "Tensor is null");
        return env.Undefined();
    }
    
    halide_type_t type = tensor_->getType();
    size_t elementSize = tensor_->elementSize();
    size_t totalElements = tensor_->elementSize();
    
    // Create ArrayBuffer
    Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(env, totalElements * (type.bits / 8));
    
    // Copy data from tensor to ArrayBuffer
    void* bufferData = arrayBuffer.Data();
    const void* tensorData = tensor_->host<void>();
    std::memcpy(bufferData, tensorData, totalElements * (type.bits / 8));
    
    // Create TypedArray from ArrayBuffer
    napi_typedarray_type arrayType = GetTypedArrayType(type);
    Napi::TypedArray typedArray;
    
    switch (arrayType) {
        case napi_float32_array:
            typedArray = Napi::Float32Array::New(env, totalElements, arrayBuffer, 0);
            break;
        case napi_float64_array:
            typedArray = Napi::Float64Array::New(env, totalElements, arrayBuffer, 0);
            break;
        case napi_int32_array:
            typedArray = Napi::Int32Array::New(env, totalElements, arrayBuffer, 0);
            break;
        case napi_uint8_array:
            typedArray = Napi::Uint8Array::New(env, totalElements, arrayBuffer, 0);
            break;
        default:
            typedArray = Napi::Float32Array::New(env, totalElements, arrayBuffer, 0);
    }
    
    return typedArray;
}

Napi::Value TensorWrap::CopyFrom(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    CHECK_ARG_COUNT(info, 1);
    
    if (tensor_ == nullptr) {
        THROW_ERROR(env, "Tensor is null");
        return env.Undefined();
    }
    
    if (!info[0].IsTypedArray()) {
        THROW_TYPE_ERROR(env, "Argument must be a TypedArray");
        return env.Undefined();
    }
    
    Napi::TypedArray typedArray = info[0].As<Napi::TypedArray>();
    size_t byteLength = typedArray.ByteLength();
    size_t tensorByteLength = tensor_->elementSize() * (tensor_->getType().bits / 8);
    
    if (byteLength < tensorByteLength) {
        THROW_ERROR(env, "TypedArray size is smaller than tensor size");
        return env.Undefined();
    }
    
    // Copy data from TypedArray to tensor
    void* tensorData = tensor_->host<void>();
    void* arrayData = typedArray.ArrayBuffer().Data();
    std::memcpy(tensorData, arrayData, tensorByteLength);
    
    return env.Undefined();
}

Napi::Value TensorWrap::GetHost(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (tensor_ == nullptr) {
        THROW_ERROR(env, "Tensor is null");
        return env.Undefined();
    }
    
    // Return pointer address as BigInt (for debugging)
    void* host = tensor_->host<void>();
    return Napi::BigInt::New(env, reinterpret_cast<uint64_t>(host));
}

Napi::Value TensorWrap::GetElementSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (tensor_ == nullptr) {
        THROW_ERROR(env, "Tensor is null");
        return env.Undefined();
    }
    
    return Napi::Number::New(env, tensor_->elementSize());
}

} // namespace mnn_node
