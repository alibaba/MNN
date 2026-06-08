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
        if ('\'' == c) {
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

static std::unique_ptr<MNN::AttributeT> makeStringAttr(const std::string& key, const std::string& value) {
    std::unique_ptr<MNN::AttributeT> attr(new MNN::AttributeT);
    attr->key = key;
    attr->s = value;
    attr->type = MNN::DataType_DT_STRING;
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
    auto outputs = collectOutputNames(sourceNet);
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

    for (const auto& input : inputs) {
        const int tensorIndex = ensureTensorIndex(input.name, &tensorMap, &wrapper->tensorName);
        inputIndexes.emplace_back(tensorIndex);

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
        outputIndexes.emplace_back(ensureTensorIndex(output, &tensorMap, &wrapper->tensorName));
    }

    std::unique_ptr<OpT> rknnOp(new OpT);
    rknnOp->name = "RKNNSubgraph";
    rknnOp->type = OpType_Extra;
    rknnOp->main.type = OpParameter_Extra;
    rknnOp->main.value = new ExtraT;
    rknnOp->main.AsExtra()->type = "RKNN";
    rknnOp->main.AsExtra()->engine = "MNN";
    rknnOp->main.AsExtra()->attr.emplace_back(makeStringAttr("model_path", bundlePaths.rknnPath));
    rknnOp->main.AsExtra()->attr.emplace_back(makeStringAttr("bundle_manifest", bundlePaths.manifestPath));
    rknnOp->main.AsExtra()->attr.emplace_back(makeStringAttr("target", modelPath.rknnTarget));
    rknnOp->inputIndexes = inputIndexes;
    rknnOp->outputIndexes = outputIndexes;
    wrapper->oplists.emplace_back(std::move(rknnOp));

    wrapper->outputName = outputs;
    wrapper->tensorNumber = static_cast<int>(wrapper->tensorName.size());
    return wrapper;
}
}
