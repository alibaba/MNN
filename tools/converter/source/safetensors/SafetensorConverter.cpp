#include <fstream>
#include <limits>
#include <sstream>

#include <MNN/MNNDefine.h>
#include <MNN/expr/ExprCreator.hpp>

#include <rapidjson/document.h>

#include "SafetensorConverter.hpp"
#include "SafetensorModelRegistry.hpp"
#include "WorkflowJson.hpp"

#include "../common/CommonUtils.hpp"

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"
namespace MNN {
namespace SafeTensors {

static halide_type_t _convertSafeTensorDType(safetensors::dtype dtype) {
    switch (dtype) {
        case safetensors::kBOOL:
            // Safetensors stores BOOL as 1 byte. MNN's 1-bit bool type is not widely supported.
            return halide_type_of<uint8_t>();
        case safetensors::kUINT8:
            return halide_type_of<uint8_t>();
        case safetensors::kINT8:
            return halide_type_of<int8_t>();
        case safetensors::kINT16:
            return halide_type_of<int16_t>();
        case safetensors::kUINT16:
            return halide_type_of<uint16_t>();
        case safetensors::kINT32:
            return halide_type_of<int32_t>();
        case safetensors::kUINT32:
            return halide_type_of<uint32_t>();
        case safetensors::kINT64:
            return halide_type_of<int64_t>();
        case safetensors::kUINT64:
            return halide_type_of<uint64_t>();
        case safetensors::kFLOAT16:
            return halide_type_t(halide_type_float, 16);
        case safetensors::kBFLOAT16:
            return halide_type_t(halide_type_bfloat, 16);
        case safetensors::kFLOAT32:
            return halide_type_of<float>();
        case safetensors::kFLOAT64:
            return halide_type_of<double>();
        default:
            break;
    }
    return halide_type_of<float>();
}

struct Converter::Content {
    rapidjson::Document mWorkFlow;
    safetensors::safetensors_t mSt;
};

Converter::Converter(const std::string& jsonFile) {
    mMain = new Content;

    std::ifstream fileNames(jsonFile);
    std::ostringstream output;
    output << fileNames.rdbuf();
    auto outputStr = output.str();

    mMain->mWorkFlow.Parse(outputStr.c_str());
    if (mMain->mWorkFlow.HasParseError() || !mMain->mWorkFlow.IsObject()) {
        MNN_ERROR("Invalid json\n");
        mMain->mWorkFlow.SetObject();
        return;
    }
}

Converter::~ Converter() {
    delete mMain;
}

std::vector<std::string> Converter::listModels() const {
    std::vector<std::string> res;
    if (nullptr == mMain) {
        return res;
    }
    auto models = WorkflowJson::getArray(mMain->mWorkFlow, "models");
    if (nullptr == models) {
        return res;
    }
    for (auto& model : models->GetArray()) {
        if (!model.IsObject()) {
            continue;
        }
        auto name = WorkflowJson::getString(model, "name");
        if (name.empty()) {
            continue;
        }
        res.emplace_back(std::move(name));
    }
    return res;
}
void Converter::loadSafeTensors(const std::string& safeTensorFile) {
    std::string warn, err;
    auto ret = safetensors::mmap_from_file(safeTensorFile, &mMain->mSt, &warn, &err);
    if (warn.size()) {
        FUNC_PRINT_ALL(warn.c_str(), s);
    }
    if (!ret) {
        FUNC_PRINT_ALL(err.c_str(), s);
        return;
    }
}
bool Converter::convert(const std::string& name, modelConfig& modelPath) {
    auto builder = SafetensorModelRegistry::get()->find(name);
    if (nullptr == builder) {
        MNN_ERROR("SafetensorConverter: unsupported model %s\n", name.c_str());
        return false;
    }

    const rapidjson::Value* model = nullptr;
    if (mMain != nullptr && mMain->mWorkFlow.IsObject()) {
        auto models = WorkflowJson::getArray(mMain->mWorkFlow, "models");
        if (nullptr != models) {
            for (auto& item : models->GetArray()) {
                if (!item.IsObject()) {
                    continue;
                }
                auto modelName = WorkflowJson::getString(item, "name");
                MNN_PRINT("Checking model config for: %s (target: %s)\n", modelName.c_str(), name.c_str());
                if (!modelName.empty() && modelName == name) {
                    model = &item;
                    int weightQuantBits = WorkflowJson::getInt(item, "weightQuantBits", -1);
                    if (weightQuantBits >= 0) {
                        MNN_PRINT("Override weightQuantBits to %d for model %s\n", weightQuantBits, name.c_str());
                        modelPath.weightQuantBits = weightQuantBits;
                    }
                    break;
                }
            }
        }
    }

    return builder(this, model, modelPath);
}
bool Converter::hasTensor(const std::string& name) const {
    safetensors::tensor_t t;
    if (mMain->mSt.tensors.at(name, &t)) {
        return true;
    }
    return false;
}

MNN::Express::VARP Converter::loadTensor(const std::string& name, bool print) const {
    safetensors::tensor_t t;
    bool find = mMain->mSt.tensors.at(name, &t);
    if (!find) {
        if (print) {
            FUNC_PRINT_ALL(name.c_str(), s);
        }
        return nullptr;
    }

    const uint8_t* dataBufferAddr = nullptr;
    size_t dataBufferSize = 0;
    if (mMain->mSt.mmaped) {
        dataBufferAddr = mMain->mSt.databuffer_addr;
        dataBufferSize = mMain->mSt.databuffer_size;
    } else {
        dataBufferAddr = mMain->mSt.storage.data();
        dataBufferSize = mMain->mSt.storage.size();
    }
    if (nullptr == dataBufferAddr || dataBufferSize == 0) {
        MNN_ERROR("Safetensors databuffer is empty, please call loadSafeTensors first\n");
        return nullptr;
    }

    const size_t offsetBegin = t.data_offsets[0];
    const size_t offsetEnd = t.data_offsets[1];
    if (offsetBegin > offsetEnd || offsetEnd > dataBufferSize) {
        MNN_ERROR("Invalid tensor offsets for %s: [%zu, %zu), databuffer=%zu\n", name.c_str(), offsetBegin, offsetEnd, dataBufferSize);
        return nullptr;
    }

    const size_t nitems = safetensors::get_shape_size(t);
    const size_t itemBytes = safetensors::get_dtype_bytes(t.dtype);
    const size_t expectedBytes = nitems * itemBytes;
    const size_t actualBytes = offsetEnd - offsetBegin;
    if (expectedBytes != actualBytes) {
        MNN_ERROR("Invalid tensor %s: expected %zu bytes(%zu*%zu), got %zu\n", name.c_str(), expectedBytes, nitems, itemBytes, actualBytes);
        return nullptr;
    }

    MNN::Express::INTS shape;
    shape.reserve(t.shape.size());
    for (auto dim : t.shape) {
        if (dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
            MNN_ERROR("Tensor %s has too large shape dim: %zu\n", name.c_str(), dim);
            return nullptr;
        }
        shape.emplace_back(static_cast<int>(dim));
    }

    auto tensorStart = dataBufferAddr + offsetBegin;
    auto dtype = _convertSafeTensorDType(t.dtype);
    auto var = MNN::Express::_Const(tensorStart, std::move(shape), MNN::Express::NCHW, dtype);
    if (t.dtype == safetensors::kBFLOAT16) {
        var = MNN::Express::_Cast<float>(var);
        var.fix(MNN::Express::VARP::CONSTANT);
    }
    return var;
}

};
};
