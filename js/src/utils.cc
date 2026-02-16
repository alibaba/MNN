//
//  Utility functions implementation
//
#include "utils.h"
#include <MNN/MNNForwardType.h>

namespace mnn_node {

bool GetScheduleConfig(const Napi::Object& obj, MNN::ScheduleConfig& config) {
    // Set defaults
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    config.backupType = MNN_FORWARD_CPU;
    
    if (obj.Has("type")) {
        Napi::Value typeVal = obj.Get("type");
        if (typeVal.IsNumber()) {
            config.type = static_cast<MNNForwardType>(typeVal.As<Napi::Number>().Int32Value());
        }
    }
    
    if (obj.Has("numThread")) {
        Napi::Value numThreadVal = obj.Get("numThread");
        if (numThreadVal.IsNumber()) {
            config.numThread = numThreadVal.As<Napi::Number>().Int32Value();
        }
    }
    
    if (obj.Has("backupType")) {
        Napi::Value backupTypeVal = obj.Get("backupType");
        if (backupTypeVal.IsNumber()) {
            config.backupType = static_cast<MNNForwardType>(backupTypeVal.As<Napi::Number>().Int32Value());
        }
    }
    
    if (obj.Has("mode")) {
        Napi::Value modeVal = obj.Get("mode");
        if (modeVal.IsNumber()) {
            config.mode = modeVal.As<Napi::Number>().Int32Value();
        }
    }
    
    return true;
}

bool GetBackendConfig(const Napi::Object& obj, MNN::BackendConfig& config) {
    // Set defaults
    config.precision = MNN::BackendConfig::Precision_Normal;
    config.power = MNN::BackendConfig::Power_Normal;
    config.memory = MNN::BackendConfig::Memory_Normal;
    
    if (obj.Has("precision")) {
        Napi::Value precisionVal = obj.Get("precision");
        if (precisionVal.IsNumber()) {
            config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(
                precisionVal.As<Napi::Number>().Int32Value()
            );
        }
    }
    
    if (obj.Has("power")) {
        Napi::Value powerVal = obj.Get("power");
        if (powerVal.IsNumber()) {
            config.power = static_cast<MNN::BackendConfig::PowerMode>(
                powerVal.As<Napi::Number>().Int32Value()
            );
        }
    }
    
    if (obj.Has("memory")) {
        Napi::Value memoryVal = obj.Get("memory");
        if (memoryVal.IsNumber()) {
            config.memory = static_cast<MNN::BackendConfig::MemoryMode>(
                memoryVal.As<Napi::Number>().Int32Value()
            );
        }
    }
    
    return true;
}

napi_typedarray_type GetTypedArrayType(const halide_type_t& type) {
    if (type.code == halide_type_float) {
        if (type.bits == 32) {
            return napi_float32_array;
        } else if (type.bits == 64) {
            return napi_float64_array;
        }
    } else if (type.code == halide_type_int) {
        if (type.bits == 8) {
            return napi_int8_array;
        } else if (type.bits == 16) {
            return napi_int16_array;
        } else if (type.bits == 32) {
            return napi_int32_array;
        }
    } else if (type.code == halide_type_uint) {
        if (type.bits == 8) {
            return napi_uint8_array;
        } else if (type.bits == 16) {
            return napi_uint16_array;
        } else if (type.bits == 32) {
            return napi_uint32_array;
        }
    }
    return napi_float32_array; // default
}

std::vector<int> ArrayToIntVector(const Napi::Array& arr) {
    std::vector<int> vec;
    uint32_t length = arr.Length();
    vec.reserve(length);
    
    for (uint32_t i = 0; i < length; i++) {
        Napi::Value val = arr[i];
        if (val.IsNumber()) {
            vec.push_back(val.As<Napi::Number>().Int32Value());
        }
    }
    
    return vec;
}

Napi::Array IntVectorToArray(Napi::Env env, const std::vector<int>& vec) {
    Napi::Array arr = Napi::Array::New(env, vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        arr[i] = Napi::Number::New(env, vec[i]);
    }
    return arr;
}

} // namespace mnn_node
