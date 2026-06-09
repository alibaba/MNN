#include "RKNNBundle.hpp"

#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "CaffeOp_generated.h"
#include "CommonUtils.hpp"
#include "MNN/ErrorCode.hpp"
#include "MNN_generated.h"
#include "../optimizer/Program.hpp"
#include "core/MNNFileUtils.h"
#include "logkit.h"

namespace {
static const char* MNN_RKNN_TARGET_ENV = "MNN_RKNN_TARGET";
static const char* MNN_RKNN_PYTHON_ENV = "MNN_RKNN_PYTHON";
static const char* MNN_RKNN_SCRIPT_ENV = "MNN_RKNN_SCRIPT";
static const char* MNN_RKNN_OUTPUT_DIR_ENV = "MNN_RKNN_OUTPUT_DIR";

static std::string getEnvValue(const char* name) {
    auto value = std::getenv(name);
    if (nullptr == value) {
        return "";
    }
    return value;
}

static bool loadRequiredEnv(std::string& dst, const char* name) {
    dst = getEnvValue(name);
    if (dst.empty()) {
        MNN_ERROR("RKNN sidecar requires environment variable %s\n", name);
        return false;
    }
    return true;
}

static std::string shellEscape(const std::string& input) {
    std::string escaped = "'";
    for (char c : input) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(c);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

static std::string basenameWithoutExtension(const std::string& path) {
    auto slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    auto dot = name.find_last_of('.');
    if (dot == std::string::npos) {
        return name;
    }
    return name.substr(0, dot);
}
struct InputInfo {
    std::string name;
    std::vector<int> dims;
    MNN::DataType dtype = MNN::DataType_DT_FLOAT;
    MNN::MNN_DATA_FORMAT dformat = MNN::MNN_DATA_FORMAT_NC4HW4;
};

struct OutputInfo {
    std::string name;
    std::vector<int> dims;
    MNN::DataType dtype = MNN::DataType_DT_FLOAT;
    MNN::MNN_DATA_FORMAT dformat = MNN::MNN_DATA_FORMAT_NC4HW4;
};

static std::vector<InputInfo> collectInputInfos(const MNN::NetT& net) {
    std::vector<InputInfo> inputs;
    for (const auto& op : net.oplists) {
        if (nullptr == op || op->type != MNN::OpType_Input || op->outputIndexes.empty()) {
            continue;
        }
        auto input = op->main.AsInput();
        if (nullptr == input) {
            continue;
        }
        const auto outputIndex = op->outputIndexes[0];
        if (outputIndex < 0 || outputIndex >= net.tensorName.size()) {
            MNN_ERROR("RKNN wrapper: invalid input tensor index %d\n", outputIndex);
            return {};
        }
        InputInfo info;
        info.name = net.tensorName[outputIndex];
        info.dims.assign(input->dims.begin(), input->dims.end());
        info.dtype = input->dtype;
        info.dformat = input->dformat;
        inputs.emplace_back(std::move(info));
    }
    return inputs;
}

static std::vector<std::string> collectOutputNames(const MNN::NetT& net) {
    if (!net.outputName.empty()) {
        return net.outputName;
    }
    std::set<int> inputIndexes;
    std::set<int> outputIndexes;
    std::vector<std::string> outputNames;
    for (const auto& op : net.oplists) {
        if (nullptr == op) {
            continue;
        }
        for (auto inputIndex : op->inputIndexes) {
            inputIndexes.insert(inputIndex);
        }
        for (auto outputIndex : op->outputIndexes) {
            outputIndexes.insert(outputIndex);
        }
    }
    for (auto outputIndex : outputIndexes) {
        if (inputIndexes.find(outputIndex) != inputIndexes.end()) {
            continue;
        }
        if (outputIndex < 0 || outputIndex >= net.tensorName.size()) {
            continue;
        }
        outputNames.emplace_back(net.tensorName[outputIndex]);
    }
    return outputNames;
}

static MNN::DataType mapExprDataType(const halide_type_t& type) {
    if (type.code == halide_type_float) {
        if (type.bits == 16) {
            return MNN::DataType_DT_HALF;
        }
        if (type.bits == 64) {
            return MNN::DataType_DT_DOUBLE;
        }
        return MNN::DataType_DT_FLOAT;
    }
    if (type.code == halide_type_uint) {
        if (type.bits == 8) {
            return MNN::DataType_DT_UINT8;
        }
        if (type.bits == 16) {
            return MNN::DataType_DT_UINT16;
        }
        if (type.bits == 32) {
            return MNN::DataType_DT_INT32;
        }
        return MNN::DataType_DT_INT32;
    }
    if (type.code == halide_type_int) {
        if (type.bits == 8) {
            return MNN::DataType_DT_INT8;
        }
        if (type.bits == 16) {
            return MNN::DataType_DT_INT16;
        }
        if (type.bits == 64) {
            return MNN::DataType_DT_INT64;
        }
        return MNN::DataType_DT_INT32;
    }
    if (type.code == halide_type_handle) {
        return MNN::DataType_DT_STRING;
    }
    return MNN::DataType_DT_FLOAT;
}

static MNN::MNN_DATA_FORMAT mapExprFormat(MNN::Express::Dimensionformat format) {
    switch (format) {
        case MNN::Express::NHWC:
            return MNN::MNN_DATA_FORMAT_NHWC;
        case MNN::Express::NC4HW4:
            return MNN::MNN_DATA_FORMAT_NC4HW4;
        case MNN::Express::NCHW:
        default:
            return MNN::MNN_DATA_FORMAT_NCHW;
    }
}

static std::vector<OutputInfo> collectOutputInfos(const MNN::NetT& net) {
    auto outputNames = collectOutputNames(net);
    if (outputNames.empty()) {
        return {};
    }
    auto program = MNN::Express::Program::create(&net, true, true);
    if (nullptr == program) {
        MNN_ERROR("RKNN wrapper: failed to build Program for output shape inference\n");
        return {};
    }

    std::map<std::string, const MNN::Express::Variable::Info*> infoMap;
    for (const auto& output : program->outputs()) {
        if (output == nullptr) {
            continue;
        }
        auto info = output->getInfo();
        if (nullptr == info) {
            continue;
        }
        infoMap.insert(std::make_pair(output->name(), info));
    }

    std::vector<OutputInfo> outputs;
    outputs.reserve(outputNames.size());
    for (const auto& name : outputNames) {
        auto infoIter = infoMap.find(name);
        if (infoIter == infoMap.end() || nullptr == infoIter->second) {
            MNN_ERROR("RKNN wrapper: failed to infer output info for tensor %s\n", name.c_str());
            return {};
        }
        OutputInfo info;
        info.name = name;
        info.dims.assign(infoIter->second->dim.begin(), infoIter->second->dim.end());
        info.dtype = mapExprDataType(infoIter->second->type);
        info.dformat = mapExprFormat(infoIter->second->order);
        outputs.emplace_back(std::move(info));
    }
    return outputs;
}

static std::unique_ptr<MNN::AttributeT> makeStringAttr(const std::string& key, const std::string& value) {
    std::unique_ptr<MNN::AttributeT> attr(new MNN::AttributeT);
    attr->key = key;
    attr->s = value;
    attr->type = MNN::DataType_DT_STRING;
    return attr;
}

static std::unique_ptr<MNN::AttributeT> makeStringListAttr(const std::string& key, const std::vector<std::string>& values) {
    std::unique_ptr<MNN::AttributeT> attr(new MNN::AttributeT);
    attr->key = key;
    attr->list.reset(new MNN::ListValueT);
    attr->list->s = values;
    return attr;
}

static std::unique_ptr<MNN::AttributeT> makeBlobAttr(const std::string& key, const OutputInfo& info) {
    std::unique_ptr<MNN::AttributeT> attr(new MNN::AttributeT);
    attr->key = key;
    attr->tensor.reset(new MNN::BlobT);
    attr->tensor->dataType = info.dtype;
    attr->tensor->dims = info.dims;
    attr->tensor->dataFormat = info.dformat;
    return attr;
}

static int ensureTensorIndex(const std::string& name, std::map<std::string, int>* tensorMap,
                             std::vector<std::string>* tensorNames) {
    auto iter = tensorMap->find(name);
    if (iter != tensorMap->end()) {
        return iter->second;
    }
    const int index = static_cast<int>(tensorNames->size());
    tensorNames->emplace_back(name);
    tensorMap->insert(std::make_pair(name, index));
    return index;
}
}

namespace MNN {

bool PopulateRKNNConfigFromEnv(modelConfig& modelPath) {
    if (!loadRequiredEnv(modelPath.rknnTarget, MNN_RKNN_TARGET_ENV)) {
        return false;
    }
    if (!loadRequiredEnv(modelPath.rknnPython, MNN_RKNN_PYTHON_ENV)) {
        return false;
    }
    if (!loadRequiredEnv(modelPath.rknnScript, MNN_RKNN_SCRIPT_ENV)) {
        return false;
    }
    if (!loadRequiredEnv(modelPath.rknnOutputDir, MNN_RKNN_OUTPUT_DIR_ENV)) {
        return false;
    }
    if (!CommonKit::FileIsExist(modelPath.rknnScript)) {
        MNN_ERROR("RKNN script does not exist: %s\n", modelPath.rknnScript.c_str());
        return false;
    }
    return true;
}

bool GenerateRKNNBundle(const modelConfig& modelPath, RKNNBundlePaths* bundlePaths) {
    if (modelPath.model != modelConfig::ONNX) {
        MNN_ERROR("RKNN sidecar only supports ONNX source models\n");
        return false;
    }
    if (modelPath.modelFile.empty() || modelPath.MNNModel.empty()) {
        MNN_ERROR("RKNN sidecar requires both source ONNX path and output MNN path\n");
        return false;
    }
    if (!MNNDirExist(modelPath.rknnOutputDir.c_str()) && !MNNCreateDir(modelPath.rknnOutputDir.c_str())) {
        MNN_ERROR("Create RKNN output dir failed: %s\n", modelPath.rknnOutputDir.c_str());
        return false;
    }

    const auto baseName = basenameWithoutExtension(modelPath.MNNModel);
    const auto rknnPath = MNNFilePathConcat(modelPath.rknnOutputDir, baseName + "_" + modelPath.rknnTarget + ".rknn");
    const auto manifestPath = MNNFilePathConcat(modelPath.rknnOutputDir, baseName + ".rknn.bundle.json");

    std::ostringstream command;
    command << shellEscape(modelPath.rknnPython) << " "
            << shellEscape(modelPath.rknnScript)
            << " --onnx " << shellEscape(modelPath.modelFile)
            << " --output " << shellEscape(rknnPath)
            << " --target " << shellEscape(modelPath.rknnTarget);

    MNN_PRINT("Generate RKNN sidecar with command: %s\n", command.str().c_str());
    auto ret = std::system(command.str().c_str());
    if (ret != 0) {
        MNN_ERROR("RKNN sidecar generation failed, exit code: %d\n", ret);
        return false;
    }
    if (!MNNFileExist(rknnPath.c_str())) {
        MNN_ERROR("RKNN sidecar is not generated: %s\n", rknnPath.c_str());
        return false;
    }

    std::ofstream manifest(manifestPath.c_str(), std::ios::out | std::ios::trunc);
    if (!manifest.good()) {
        MNN_ERROR("Open RKNN manifest failed: %s\n", manifestPath.c_str());
        return false;
    }
    manifest << "{\n";
    manifest << "  \"onnx_model\": \"" << modelPath.modelFile << "\",\n";
    manifest << "  \"mnn_model\": \"" << modelPath.MNNModel << "\",\n";
    manifest << "  \"rknn_model\": \"" << rknnPath << "\",\n";
    manifest << "  \"target\": \"" << modelPath.rknnTarget << "\"";
    const auto weightPath = modelPath.MNNModel + ".weight";
    if (MNNFileExist(weightPath.c_str())) {
        manifest << ",\n  \"mnn_external_weight\": \"" << weightPath << "\"\n";
    } else {
        manifest << "\n";
    }
    manifest << "}\n";
    manifest.close();

    if (!manifest.good()) {
        MNN_ERROR("Write RKNN manifest failed: %s\n", manifestPath.c_str());
        return false;
    }

    MNN_PRINT("RKNN sidecar generated: %s\n", rknnPath.c_str());
    MNN_PRINT("RKNN manifest generated: %s\n", manifestPath.c_str());
    if (nullptr != bundlePaths) {
        bundlePaths->rknnPath = rknnPath;
        bundlePaths->manifestPath = manifestPath;
    }
    return true;
}

std::unique_ptr<NetT> BuildRKNNWrapperNet(const NetT& sourceNet, const modelConfig& modelPath,
                                          const RKNNBundlePaths& bundlePaths) {
    auto inputs = collectInputInfos(sourceNet);
    if (inputs.empty()) {
        MNN_ERROR("RKNN wrapper: failed to collect input tensors from source net\n");
        return nullptr;
    }
    auto outputs = collectOutputInfos(sourceNet);
    if (outputs.empty()) {
        MNN_ERROR("RKNN wrapper: failed to collect output tensors from source net\n");
        return nullptr;
    }

    std::unique_ptr<NetT> wrapper(new NetT);
    wrapper->bizCode = modelPath.bizCode;
    wrapper->sourceType = NetSource_ONNX;
    wrapper->usage = Usage_INFERENCE;
    wrapper->preferForwardType = ForwardType_CPU;

    std::map<std::string, int> tensorMap;
    std::vector<int> inputIndexes;
    std::vector<int> outputIndexes;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

    for (const auto& input : inputs) {
        const int tensorIndex = ensureTensorIndex(input.name, &tensorMap, &wrapper->tensorName);
        inputIndexes.emplace_back(tensorIndex);
        inputNames.emplace_back(input.name);

        std::unique_ptr<OpT> inputOp(new OpT);
        inputOp->name = input.name;
        inputOp->type = OpType_Input;
        inputOp->main.type = OpParameter_Input;
        inputOp->main.value = new InputT;
        inputOp->main.AsInput()->dims.assign(input.dims.begin(), input.dims.end());
        inputOp->main.AsInput()->dtype = input.dtype;
        inputOp->main.AsInput()->dformat = input.dformat;
        inputOp->outputIndexes = {tensorIndex};
        inputOp->defaultDimentionFormat = input.dformat;
        wrapper->oplists.emplace_back(std::move(inputOp));
    }

    for (const auto& output : outputs) {
        outputIndexes.emplace_back(ensureTensorIndex(output.name, &tensorMap, &wrapper->tensorName));
        outputNames.emplace_back(output.name);
    }

    std::unique_ptr<OpT> rknnOp(new OpT);
    rknnOp->name = "RKNNSubgraph";
    rknnOp->type = OpType_Plugin;
    rknnOp->main.type = OpParameter_Plugin;
    rknnOp->main.value = new PluginT;
    rknnOp->main.AsPlugin()->type = "RKNN";
    rknnOp->main.AsPlugin()->attr.emplace_back(makeStringAttr("model_path", bundlePaths.rknnPath));
    rknnOp->main.AsPlugin()->attr.emplace_back(makeStringAttr("bundle_manifest", bundlePaths.manifestPath));
    rknnOp->main.AsPlugin()->attr.emplace_back(makeStringAttr("target", modelPath.rknnTarget));
    rknnOp->main.AsPlugin()->attr.emplace_back(makeStringListAttr("inputs", inputNames));
    rknnOp->main.AsPlugin()->attr.emplace_back(makeStringListAttr("outputs", outputNames));
    for (int i = 0; i < outputs.size(); ++i) {
        rknnOp->main.AsPlugin()->attr.emplace_back(makeBlobAttr("o_" + std::to_string(i), outputs[i]));
    }
    rknnOp->inputIndexes = inputIndexes;
    rknnOp->outputIndexes = outputIndexes;
    wrapper->oplists.emplace_back(std::move(rknnOp));

    wrapper->outputName = outputNames;
    wrapper->tensorNumber = static_cast<int>(wrapper->tensorName.size());
    return wrapper;
}
} // namespace MNN
