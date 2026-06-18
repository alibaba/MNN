#ifndef SafetensorModelRegistry_hpp
#define SafetensorModelRegistry_hpp

#include <memory>
#include <string>

#include "config.hpp"

#include <rapidjson/document.h>

#include <MNN/MNNDefine.h>

namespace MNN {
struct NetT;

namespace SafeTensors {

class Converter;

using ModelBuilder = bool (*)(const Converter* converter, const rapidjson::Value* model, modelConfig& modelPath);

class MNN_PUBLIC SafetensorModelRegistry {
public:
    static SafetensorModelRegistry* get();

    void insert(const std::string& name, ModelBuilder builder);
    ModelBuilder find(const std::string& name) const;

private:
    SafetensorModelRegistry() = default;

    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

class MNN_PUBLIC SafetensorModelRegister {
public:
    SafetensorModelRegister(const char* name, ModelBuilder builder);
};

MNN_PUBLIC void optimizeAndWrite(modelConfig& modelPath, std::unique_ptr<MNN::NetT>& netT);

} // namespace SafeTensors
} // namespace MNN

#define MNN_SAFETENSOR_JOIN_INNER(x, y) x##y
#define MNN_SAFETENSOR_JOIN(x, y) MNN_SAFETENSOR_JOIN_INNER(x, y)

#define REGISTER_SAFETENSOR_MODEL_BUILDER(modelName, builderFunc) \
    static ::MNN::SafeTensors::SafetensorModelRegister \
        MNN_SAFETENSOR_JOIN(__mnn_safetensor_model_register_, __COUNTER__)(modelName, builderFunc)

#endif
