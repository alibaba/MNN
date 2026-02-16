//
//  Utility functions for type conversion and config parsing
//
#ifndef MNN_NODE_UTILS_H
#define MNN_NODE_UTILS_H

#include "common.h"

namespace mnn_node {

// Convert JavaScript object to ScheduleConfig
bool GetScheduleConfig(const Napi::Object& obj, MNN::ScheduleConfig& config);

// Convert JavaScript object to BackendConfig
bool GetBackendConfig(const Napi::Object& obj, MNN::BackendConfig& config);

// Convert MNN::halide_type_t to Napi::TypedArray type
Napi::TypedArray CreateTypedArrayFromTensor(Napi::Env env, const MNN::Tensor* tensor);

// Get TypedArray element type from halide_type_t
napi_typedarray_type GetTypedArrayType(const halide_type_t& type);

// Convert Napi::Array to std::vector<int>
std::vector<int> ArrayToIntVector(const Napi::Array& arr);

// Convert std::vector to Napi::Array
Napi::Array IntVectorToArray(Napi::Env env, const std::vector<int>& vec);

} // namespace mnn_node

#endif // MNN_NODE_UTILS_H
