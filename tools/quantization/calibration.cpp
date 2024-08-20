//
//  calibration.cpp
//  MNN
//
//  Created by MNN on 2019/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "calibration.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <algorithm>
#include <MNN/ImageProcess.hpp>
#include "flatbuffers/util.h"
#include "logkit.h"
#include "quantizeWeight.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "Helper.hpp"
#include "core/TensorUtils.hpp"
#include "core/IDSTEncoder.hpp"

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include "train/source/nn/NN.hpp"
#include "train/source/datasets/ImageNoLabelDataset.hpp"
#include "train/source/datasets/ImageDataset.hpp"
#include "train/source/optimizer/SGD.hpp"
#include "train/source/transformer/Transformer.hpp"
#include "cpp/ConvertToFullQuant.hpp"
#include "core/ConvolutionCommon.hpp"
#include <MNN/expr/Expr.hpp>


#include <fstream>
#include <iostream>
#define DUMP_NUM_DATA(type)                          \
    auto data = tensor->host<type>();                \
    for (int z = 0; z < outside; ++z) {              \
        for (int x = 0; x < width; ++x) {            \
            outputOs << data[x + z * width] << "\t"; \
        }                                            \
        outputOs << "\n";                            \
    }

#define DUMP_CHAR_DATA(type)                                           \
    auto data = tensor->host<type>();                                  \
    for (int z = 0; z < outside; ++z) {                                \
        for (int x = 0; x < width; ++x) {                              \
            outputOs << static_cast<int>(data[x + z * width]) << "\t"; \
        }                                                              \
        outputOs << "\n";                                              \
    }

static void dumpTensor2File(const Tensor* tensor, const char* file) {
    std::ofstream outputOs(file);
    auto type = tensor->getType();

    int dimension = tensor->buffer().dimensions;
    int width     = 1;
    if (dimension > 1) {
        width = tensor->length(dimension - 1);
    }

    const int outside = tensor->elementSize() / width;

    const auto dataType  = type.code;
    const auto dataBytes = type.bytes();

    if (dataType == halide_type_float) {
        DUMP_NUM_DATA(float);
    }
    if (dataType == halide_type_int && dataBytes == 4) {
        DUMP_NUM_DATA(int32_t);
    }
    if (dataType == halide_type_uint && dataBytes == 1) {
        DUMP_CHAR_DATA(uint8_t);
    }
    if (dataType == halide_type_int && dataBytes == 1) {
#ifdef MNN_USE_SSE
        auto data = tensor->host<uint8_t>();
        for (int z = 0; z < outside; ++z) {
            for (int x = 0; x < width; ++x) {
                outputOs << (static_cast<int>(data[x + z * width]) - 128) << "\t";
            }
            outputOs << "\n";
        }
#else
        DUMP_CHAR_DATA(int8_t);
#endif
    }
}

using namespace MNN::CV;
using namespace MNN::Train;
using namespace MNN::Express;

static std::vector<VARP> getModuleInputs(std::string file, const Module::Info* netInfo, std::vector<std::string> inputNames, std::map<std::string, std::vector<int>> inputShape) {
#define LOAD_DATA(TYPE)\
std::ostringstream fileNameOs;\
fileNameOs << file << "/" << inputName << ".txt";\
auto fileName = fileNameOs.str();\
std::ifstream inputOs(fileName.c_str());\
if (inputOs.fail()) {\
MNN_ERROR("TESTERROR Can't open %s\n", fileName.c_str());\
continue;\
}\
for (int i=0; i<info->size; ++i) {\
double tempValue;\
inputOs >> tempValue;\
ptr[i] = tempValue;\
}\

#define LOAD_DATA_SHAPE() \
std::ostringstream file_name_os;\
file_name_os << file << "/input.json";\
auto file_name = file_name_os.str();\
std::ifstream input_os(file_name.c_str());\
if (input_os.fail()) {\
MNN_PRINT("input.json does not exit in %s, use default shape.\n", file.c_str());\
dyInputShape=inputShape;\
} else {\
rapidjson::Document document;\
std::ostringstream json_os;\
json_os << input_os.rdbuf();\
document.Parse(json_os.str().c_str());\
if (document.HasParseError()) {\
MNN_ERROR("Invalid json: %s\n", file_name.c_str());\
}\
auto cfgObj = document.GetObject();\
if (cfgObj.HasMember("inputs")) {\
auto inputsInfo = document["inputs"].GetArray();\
for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {\
auto obj = iter->GetObject();\
std::string name = obj["name"].GetString();\
if (obj.HasMember("shape")) {\
auto dims = obj["shape"].GetArray();\
std::vector<int> shapes;\
for (auto iter = dims.begin(); iter != dims.end(); iter++) {\
    shapes.emplace_back(iter->GetInt());\
}\
dyInputShape.insert(std::make_pair(name, shapes));\
}\
}\
}\
else {\
MNN_ERROR("input.json must have inputs infomation!\n");\
}\
}\

    std::vector<VARP> inputs;
    inputs.resize(netInfo->inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        inputs[i] = _Input(netInfo->inputs[i].dim, netInfo->inputs[i].order, netInfo->inputs[i].type);
    }
    // Load inputs
    std::map<std::string, std::vector<int>> dyInputShape;
    LOAD_DATA_SHAPE()
    for (int i=0; i<inputs.size(); ++i) {
        auto inputName = inputNames[i];
        // Resize
        auto shapeIter = dyInputShape.find(inputName);
        if (shapeIter != dyInputShape.end()) {
            auto s = shapeIter->second;
            inputs[i] = _Input(s, netInfo->defaultFormat, netInfo->inputs[i].type);
        }
        auto info = inputs[i]->getInfo();
        if (info->type == halide_type_of<float>()){
            auto ptr = inputs[i]->writeMap<float>();
            LOAD_DATA(float)
        } else {
            auto floatVar = _Input(info->dim, info->order, halide_type_of<float>());
            auto ptr = floatVar->writeMap<float>();
            LOAD_DATA(float)
            auto temp = _Cast(floatVar, info->type);
            inputs[i]->input(temp);
        }
        inputs[i] = _Convert(inputs[i], netInfo->inputs[i].order);
    }

#undef LOAD_DATA
#undef LOAD_DATA_SHAPE
    return inputs;
}

Calibration::Calibration(MNN::NetT* model, const uint8_t* modelBuffer, const int bufferSize, const std::string& configPath, std::string originalModelFile, std::string destModelFile) : _originalModel(model), _originalModelFile(originalModelFile), _destModelFile(destModelFile) {
    // when the format of input image is RGB/BGR, channels equal to 3, GRAY is 1
    _channels = 3;

    rapidjson::Document document;
    {
        std::ifstream fileNames(configPath.c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            mValid = false;
            return;
        }
    }
    auto picObj = document.GetObject();
    _imageProcessConfig.filterType = CV::BILINEAR;
    _imageProcessConfig.destFormat = BGR;
    {
        if (picObj.HasMember("format")) {
            auto format = picObj["format"].GetString();
            static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}, {"RGBA", RGBA}, {"BGRA", BGRA}};
            if (formatMap.find(format) != formatMap.end()) {
                _imageProcessConfig.destFormat = formatMap.find(format)->second;
            }
        }
    }

    switch (_imageProcessConfig.destFormat) {
        case GRAY:
            _channels = 1;
            break;
        case RGB:
        case BGR:
            _channels = 3;
            break;
        case RGBA:
        case BGRA:
            _channels = 4;
            break;
        default:
            break;
    }

    _imageProcessConfig.sourceFormat = RGBA;
    _calibrationFileNum = 0;
    
    if (picObj.HasMember("mean")) {
        auto mean = picObj["mean"].GetArray();
        int cur   = 0;
        for (auto iter = mean.begin(); iter != mean.end(); iter++) {
            _imageProcessConfig.mean[cur++] = iter->GetFloat();
        }
    }
    if (picObj.HasMember("normal")) {
        auto normal = picObj["normal"].GetArray();
        int cur     = 0;
        for (auto iter = normal.begin(); iter != normal.end(); iter++) {
            _imageProcessConfig.normal[cur++] = iter->GetFloat();
        }
    }
    if (picObj.HasMember("center_crop_h")) {
        _preprocessConfig.centerCropHeight = picObj["center_crop_h"].GetFloat();
    }
    if (picObj.HasMember("center_crop_w")) {
        _preprocessConfig.centerCropWidth = picObj["center_crop_w"].GetFloat();
    }
    if (picObj.HasMember("width")) {
        _width = picObj["width"].GetInt();
        _preprocessConfig.targetWidth = _width;
    }
    if (picObj.HasMember("height")) {
        _height = picObj["height"].GetInt();
        _preprocessConfig.targetHeight = _height;
    }
    if (picObj.HasMember("batch_size")) {
        _batch = picObj["batch_size"].GetInt();
    }
    if (picObj.HasMember("quant_bits")) {
        _quant_bits = picObj["quant_bits"].GetInt();
    }
    if (!picObj.HasMember("path")) {
        MNN_ERROR("calibration data path not set in .json config file\n");
        return;
    }
    _calibrationFilePath = picObj["path"].GetString();
    if (picObj.HasMember("used_image_num")) {
        _calibrationFileNum = picObj["used_image_num"].GetInt();
    }
    if (picObj.HasMember("used_sample_num")) {
        _calibrationFileNum = picObj["used_sample_num"].GetInt();
    }
    if (picObj.HasMember("feature_quantize_method")) {
        std::string method = picObj["feature_quantize_method"].GetString();
        if (Helper::featureQuantizeMethod.find(method) != Helper::featureQuantizeMethod.end()) {
            _featureQuantizeMethod = method;
        } else {
            MNN_ERROR("not supported feature quantization method: %s\n", method.c_str());
            return;
        }
    }
    if (picObj.HasMember("weight_quantize_method")) {
        std::string method = picObj["weight_quantize_method"].GetString();
        if (Helper::weightQuantizeMethod.find(method) != Helper::weightQuantizeMethod.end()) {
            _weightQuantizeMethod = method;
        } else {
            MNN_ERROR("not supported weight quantization method: %s\n", method.c_str());
            return;
        }
    }
    DLOG(INFO) << "Use feature quantization method: " << _featureQuantizeMethod;
    DLOG(INFO) << "Use weight quantization method: " << _weightQuantizeMethod;
    if (picObj.HasMember("feature_clamp_value")) {
        float value = (int)picObj["feature_clamp_value"].GetFloat();
        if (value < 0.0f || value > 127.0f) {
            MNN_ERROR("feature_clamp_value should be in (0, 127], got: %f\n", value);
            return;
        }
        _featureClampValue = value;
    }
    if (picObj.HasMember("weight_clamp_value")) {
        float value = (int)picObj["weight_clamp_value"].GetFloat();
        if (value < 0.0f || value > 127.0f) {
            MNN_ERROR("weight_clamp_value should be in (0, 127], got: %f\n", value);
            return;
        }
        _weightClampValue = value;
        if (_quant_bits < 8) {
            _weightClampValue = (float)(1 << (_quant_bits - 1)) - 1.0f;
        }
    }
    DLOG(INFO) << "feature_clamp_value: " << _featureClampValue;
    DLOG(INFO) << "weight_clamp_value: " << _weightClampValue;
    if (picObj.HasMember("winogradOpt") && picObj["winogradOpt"].GetBool() == true) {
        if (_featureQuantizeMethod == "EMA") {
            _winogradOpt = true;
        } else {
            DLOG(ERROR) << "winogradOpt only be available under EMA";
        }
    }
    if (picObj.HasMember("skip_quant_op_names")) {
        auto skip_quant_op_names = picObj["skip_quant_op_names"].GetArray();
        for (auto iter = skip_quant_op_names.begin(); iter != skip_quant_op_names.end(); iter++) {
            std::string skip_quant_op_name = iter->GetString();
            _skip_quant_ops.emplace_back(skip_quant_op_name);
            DLOG(INFO) << "skip quant op name: " << skip_quant_op_name;
        }
    }
    if (picObj.HasMember("debug")) {
        _debug = picObj["debug"].GetBool();
    }
    _inputType = Helper::InputType::IMAGE;
    if (picObj.HasMember("input_type")) {
        std::string type = picObj["input_type"].GetString();
        if (type == "sequence") {
            _inputType = Helper::InputType::SEQUENCE;
        }
    }
    
    _module.reset(Module::load({}, {}, originalModelFile.c_str()));
    auto moduleInfo = _module->getInfo();
    for (int i = 0; i < moduleInfo->inputNames.size(); ++i) {
        mInputNames.emplace_back(moduleInfo->inputNames[i]);
    }
    for (int i = 0; i < moduleInfo->outputNames.size(); ++i) {
        mOutputNames.emplace_back(moduleInfo->outputNames[i]);
    }
    bool checkOutput = false;
    if (picObj.HasMember("inputs")) {
        auto inputsInfo = document["inputs"].GetArray();
        for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
            auto obj = iter->GetObject();
            std::string name = obj["name"].GetString();
            MNN_PRINT("input: %s\n", name.c_str());
            if (obj.HasMember("value")) {
                float value = obj["value"].GetFloat();
                mInputInfo.insert(std::make_pair(name, value));
            }
            if (obj.HasMember("shape")) {
                auto dims = obj["shape"].GetArray();
                std::vector<int> shapes;
                for (auto iter = dims.begin(); iter != dims.end(); iter++) {
                    shapes.emplace_back(iter->GetInt());
                }
                mInputShape.insert(std::make_pair(name, shapes));
            }
        }
    }
    if (!picObj.HasMember("inputs")) { // User do not provide input names and shapes.
        if (mInputNames.size() > 1) {
            MNN_ERROR("Error: Input names and shapes should provided in json file!\n");
        }
        std::string name = mInputNames[0];
        auto shape = moduleInfo->inputs[0].dim;
        auto dimensionFormat = moduleInfo->inputs[0].order;
        if (dimensionFormat == NCHW || dimensionFormat == NC4HW4) {
            if (shape.size() > 2) {
                shape[2] = _preprocessConfig.targetHeight;
            }
            if (shape.size() > 3) {
                shape[3] = _preprocessConfig.targetWidth;
            }
        } else { // NHWC
            if (shape.size() > 1) {
                shape[1] = _preprocessConfig.targetHeight;
            }
            if (shape.size() > 2) {
                shape[2] = _preprocessConfig.targetWidth;
            }
        }
        mInputShape.insert(std::make_pair(name, shape));
    }
    
    std::shared_ptr<ImageProcess> process(ImageProcess::create(_imageProcessConfig), ImageProcess::destroy);
    _process = process;

    // read images file names
    if (_calibrationFilePath.back() != '/') {
        _calibrationFilePath = _calibrationFilePath + "/";
    }
    Helper::readClibrationFiles(_calibrationFiles, _calibrationFilePath.c_str(), &_calibrationFileNum);
    std::sort(_calibrationFiles.begin(), _calibrationFiles.end());

    for (auto& op : _originalModel->oplists) {
        if (op->type == MNN::OpType_BatchNorm) {
            _featureQuantizeMethod = "EMA";
            DLOG(INFO) << "this model has BatchNorm, use EMA quantize method instead";
            break;
        }
    }
    for (auto& subgraph : _originalModel->subgraphs) {
        for (auto& op : subgraph->nodes) {
            if (op->type == MNN::OpType_BatchNorm) {
                _featureQuantizeMethod = "EMA";
                DLOG(INFO) << "this model has BatchNorm, use EMA quantize method instead";
                break;
            }
        }
    }
    
    MNN::ScheduleConfig config;
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = 1;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    if (_featureQuantizeMethod == "KL" || _featureQuantizeMethod == "ADMM") {
        rtmgr->setMode(Interpreter::Session_Debug);
    }
    _module.reset(Module::load(mInputNames, mOutputNames, originalModelFile.c_str(), rtmgr)); // rtmgr.mode->debug
    if (_debug) {
        _moduleOrigin.reset(Module::load(mInputNames, mOutputNames, originalModelFile.c_str(), rtmgr)); // rtmgr.mode->debug
    }

    if (_featureQuantizeMethod == "KL" || _featureQuantizeMethod == "ADMM") {
       _initMNNSession(modelBuffer, bufferSize);
        auto netInfo = _module->getInfo();
        if (_inputType == Helper::SEQUENCE) {
            mInputs = getModuleInputs(_calibrationFiles[0], _module->getInfo(), mInputNames, mInputShape);
            for (int i = 0; i < mInputs.size(); ++i) {
                auto input = mInputs[i];
                auto name = mInputNames[i];
                std::string fileName = _calibrationFiles[0] + "/" + name + ".txt";
                auto inputTensor = (MNN::Tensor*)input->getTensor();
                Helper::preprocessInput(_process.get(), _preprocessConfig, fileName, inputTensor, _inputType);
            }
        } else {
            mInputs.resize(1);
            mInputs[0] = _Input(netInfo->inputs[0].dim, netInfo->inputs[0].order, netInfo->inputs[0].type);
            auto inputTensor = (MNN::Tensor*)mInputs[0]->getTensor();
            Helper::preprocessInput(_process.get(), _preprocessConfig, _calibrationFiles[0], inputTensor, _inputType);
            // auto dest = mInputs[0]->writeMap<float>();
            // Helper::preprocessInput((float*)dest, _process.get(), _preprocessConfig, _imageProcessConfig, _calibrationFiles[0], 4, _inputType); // 4 for C4 dimensionType.
//            Helper::preprocessInput(_process.get(), _preprocessConfig, _calibrationFiles[0], inputTensor, _inputType);
        }
        _initMaps();
    }
}

void Calibration::_initMNNSession(const uint8_t* modelBuffer, const int bufferSize) {
    _fake_quant_weights();
}

void Calibration::_initMaps() {
    _featureInfo.clear();
    _featureInfoOrigin.clear();
    _tensorMap.clear();
        // run mnn once, initialize featureMap, opInfo map
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        for (auto t : nTensors) {
            auto des = TensorUtils::getDescribe(t);
            if (des->index >= 0) {
                _tensorMap[des->index] = t;
            }
        }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (TensorUtils::getDescribe(t)->index < 0) {
                    continue;
                }
                if (_featureInfo.find(t) == _featureInfo.end() && MNN::TensorUtils::getDescribe(t)->memoryType != MNN::Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                    _featureInfo[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo after = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        for (auto t : nTensors) {
            auto des = TensorUtils::getDescribe(t);
            if (des->index >= 0) {
                _tensorMap[des->index] = t;
            }
        }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (TensorUtils::getDescribe(t)->index < 0) {
                    continue;
                }
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return true;
    };

    Express::Executor::getGlobalExecutor()->setCallBack(std::move(before), std::move(after));
    // auto netInfo = _module->getInfo();
    // for (int k = 0; k < mInputs.size(); ++k) {
    //     mInputs[k] = _Convert(mInputs[k], netInfo->inputs[k].order);
    // }
    auto outputs = _module->onForward(mInputs);
    for (auto& output_: outputs) {
        output_->readMap<float>();
    }
    _featureInfoOrigin = _featureInfo;

    if (_featureQuantizeMethod == "KL") {
        // set the tensor-statistic method of input tensor as THRESHOLD_MAX
        auto inputTensorStatistic = _featureInfo.find(mInputs[0]->getTensor());
        if (inputTensorStatistic != _featureInfo.end()) {
            inputTensorStatistic->second->setThresholdMethod(THRESHOLD_MAX);
        }
    }
}

void Calibration::_computeFeatureMapsRange() {
    // feed input data according to input images
    int count = 0;
    
    auto netInfo = _module->getInfo();
    for (const auto& file: _calibrationFiles) {
        std::vector<VARP> inputs;
        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }
        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedRangeFlags();
        }
        
        if (_inputType == Helper::SEQUENCE) {
            inputs = getModuleInputs(file, netInfo, mInputNames, mInputShape);
            for (int i = 0; i < inputs.size(); ++i) {
                auto input = inputs[i];
                auto name = mInputNames[i];
                std::string fileName = file + "/" + name + ".txt";
                auto inputTensor = (MNN::Tensor*)input->getTensor();
                Helper::preprocessInput(_process.get(), _preprocessConfig, fileName, inputTensor, _inputType);
            }
        } else {
            auto inputTensor = (MNN::Tensor*)mInputs[0]->getTensor();
            Helper::preprocessInput(_process.get(), _preprocessConfig, file, inputTensor, _inputType);
            // auto dest = mInputs[0]->writeMap<float>();
            // Helper::preprocessInput((float*)dest, _process.get(), _preprocessConfig, _imageProcessConfig, file, 4, _inputType); // 4 for C4 dimensionType.
        }

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                             const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (TensorUtils::getDescribe(t)->index < 0) {
                    continue;
                }
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (TensorUtils::getDescribe(t)->index < 0) {
                    continue;
                }
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };
        Express::Executor::getGlobalExecutor()->setCallBack(std::move(before), std::move(after));
        auto outputs = _module->onForward(mInputs);
        // outputs[0]->readMap<float>();
    }
}

void Calibration::_collectFeatureMapsDistribution() {
    for (auto& iter : _featureInfo) {
        iter.second->resetDistribution();
    }
    // feed input data according to input images
    int count = 0;
    for (const auto& file : _calibrationFiles) {
        count++;

        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedDistributionFlag();
        }
        auto netInfo = _module->getInfo();
        std::vector<VARP> inputs;
        // Load inputs
        if (_inputType == Helper::SEQUENCE) {
            inputs = getModuleInputs(file, netInfo, mInputNames, mInputShape);
            for (int i = 0; i < inputs.size(); ++i) {
                auto input = inputs[i];
                auto name = mInputNames[i];
                std::string fileName = file + "/" + name + ".txt";
                auto inputTensor = (MNN::Tensor*)input->getTensor();
                Helper::preprocessInput(_process.get(), _preprocessConfig, fileName, inputTensor, _inputType);
            }
        } else {
            auto inputTensor = (MNN::Tensor*)mInputs[0]->getTensor();
            Helper::preprocessInput(_process.get(), _preprocessConfig, file, inputTensor, _inputType);
            // auto dest = mInputs[0]->writeMap<float>();
            // Helper::preprocessInput((float*)dest, _process.get(), _preprocessConfig, _imageProcessConfig, file, 4, _inputType); // 4 for C4 dimensionType.
        }

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (TensorUtils::getDescribe(t)->index < 0) {
                    continue;
                }
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateDistribution();
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (TensorUtils::getDescribe(t)->index < 0) {
                    continue;
                }
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateDistribution();
                    }
                }
            }
            return true;
        };

        Express::Executor::getGlobalExecutor()->setCallBack(std::move(before), std::move(after));
        auto outputs = _module->onForward(mInputs);
        // outputs[0]->readMap<float>();

        MNN_PRINT("\rCollectFeatureDistribution: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
}

void Calibration::_computeFeatureScaleKL() {
    _computeFeatureMapsRange();
    _collectFeatureMapsDistribution();

    _scales.clear();
    for (auto& iter : _featureInfo) {
        AUTOTIME;
        _scales[iter.first] = iter.second->finishAndCompute();
    }
    //_featureInfo.clear();//No need now
}

void Calibration::_computeFeatureScaleADMM() {
    // feed input data according to input images
    int count = 0;
    auto netInfo = _module->getInfo();
    std::vector<VARP> inputs(mInputNames.size());
    std::vector<const MNN::Tensor*> inputTensors(mInputNames.size());
    for (int i = 0; i < inputs.size(); ++i) {
        auto shape = mInputShape[mInputNames[i]];
        shape[0] = _calibrationFileNum;
        inputs[i] = _Input(shape, netInfo->inputs[i].order, netInfo->inputs[i].type);
        inputTensors[i] = inputs[i]->getTensor();
    }
    auto dimType                        = MNN::Tensor::CAFFE_C4;
    if (netInfo->inputs[0].order == NHWC) {
        dimType = MNN::Tensor::TENSORFLOW;
    }
    for (const auto& file : _calibrationFiles) {
        if (_inputType == Helper::SEQUENCE) {
            for (int j = 0; j < inputs.size(); ++j) {
                auto name = mInputNames[j];
                std::string fileName = file + "/" + name + ".txt";
                auto inputPtr = inputTensors[j]->host<float>() + count * inputTensors[j]->stride(0);
                std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create(mInputShape[name], netInfo->inputs[j].type, inputPtr, dimType), MNN::Tensor::destroy);
                Helper::preprocessInput(_process.get(), _preprocessConfig, fileName, tensor.get(), _inputType);
            }
        } else {
            auto inputPtr = inputTensors[0]->host<float>() + count * inputTensors[0]->stride(0);
            auto name = mInputNames[0];
            std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create(mInputShape[name], netInfo->inputs[0].type, inputPtr, dimType), MNN::Tensor::destroy);
            Helper::preprocessInput(_process.get(), _preprocessConfig, file, tensor.get(), _inputType);
        }
        count++;
        MNN_PRINT("\rProcessCalibrationFiles: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
    _scales.clear();

    const int totalLayers = static_cast<int32_t>(_featureInfo.size());
    count                 = 0;

    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _scales[t] = _featureInfo[t]->computeScaleADMM();
                        count++;
                        MNN_PRINT("\rComputeADMM: %.2lf %%", (float)count * 100.0f / (float)totalLayers);
                        fflush(stdout);
                    }
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _scales[t] = _featureInfo[t]->computeScaleADMM();
                        count++;
                        MNN_PRINT("\rComputeADMM: %.2lf %%", (float)count * 100.0f / (float)totalLayers);
                        fflush(stdout);
                    }
                }
            }
        }
        return true;
    };
    Express::Executor::getGlobalExecutor()->setCallBack(std::move(before), std::move(after));
    _module->onForward(inputs);
    MNN_PRINT("\n");
}

void Calibration::_fake_quant_weights() {
    auto findAbsMax = [&] (const float* weights, const int size) {
        float absMax = 0;
        for (int i = 0; i < size; i++) {
            if (std::fabs(weights[i]) > absMax) {
                absMax = std::fabs(weights[i]);
            }
        }

        return absMax;
    };

    for (const auto& op : _originalModel->oplists) {
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise) {
            continue;
        }

        auto param = op->main.AsConvolution2D();
        const int kernelNum = param->common->outputCount;
        std::vector<float> weights = param->weight;
        const int weightSize = static_cast<int32_t>(weights.size());
        const int kernelSize = weightSize / kernelNum;

        for (int i = 0; i < kernelNum; i++) {
            const int offset = i * kernelSize;
            float absMax = findAbsMax(weights.data() + offset, kernelSize);
            float scale = absMax / _weightClampValue;
            if (absMax < 1e-6f) {
                scale = absMax;
            }

            for (int j = 0; j < kernelSize; j++) {
                float value = weights[offset + j];
                float quantValue = std::round(value / scale);
                float clampedValue = std::max(std::min(quantValue, _weightClampValue), -_weightClampValue);
                float dequantValue = scale * clampedValue;
                param->weight[offset + j] = dequantValue;
            }
        }
    }
    DLOG(INFO) << "fake quant weights done.";
}

void Calibration::_insertScale() {
    for (const auto iter :  _scales) {
        std::unique_ptr<MNN::TensorDescribeT> describe(new MNN::TensorDescribeT);
        auto des = TensorUtils::getDescribe(iter.first);
        if (des->index < 0) {
            continue;
        }
        describe->index = des->index;
        describe->quantInfo.reset(new MNN::TensorQuantInfoT);
        describe->quantInfo->scale = iter.second.first;
        describe->quantInfo->zero = iter.second.second;
        describe->quantInfo->type = MNN::DataType_DT_INT8;
        describe->quantInfo->min = -1 * _featureClampValue;
        describe->quantInfo->max = 1 * _featureClampValue;
        _originalModel->extraTensorDescribe.emplace_back(std::move(describe));
    }
    for (const auto& op : _originalModel->oplists) {
        const auto opType = op->type;

        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise && opType != MNN::OpType_Deconvolution) {
            continue;
        }
        if (op->inputIndexes.size() > 1) {
            continue;
        }
        auto inputTensor = _tensorMap[op->inputIndexes[0]];
        auto outputTensor = _tensorMap[op->outputIndexes[0]];
        if (inputTensor == nullptr || outputTensor == nullptr) {
            continue;
        }
        // below is Conv/DepthwiseConv weight quant
        const float inputScale  = _scales[inputTensor].first;
        const float outputScale = _scales[outputTensor].first;
        const int inputChannel = inputTensor->channel();
        const int outputChannel = outputTensor->channel();
        auto param                = op->main.AsConvolution2D();
        param->common->inputCount = inputChannel;
        const int channles        = param->common->outputCount;
        param->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        param->symmetricQuan->nbits = _quant_bits;
        const float* originWeight = param->weight.data();
        int originWeightSize   = static_cast<int32_t>(param->weight.size());
        auto conv2d = param;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
        std::unique_ptr<Tensor> externalWeightTensor, externalBiasTensor;
        if (nullptr != conv2d->quanParameter.get()) {
            flatbuffers::FlatBufferBuilder tempBuilder;
            tempBuilder.Finish(IDSTQuan::Pack(tempBuilder, conv2d->quanParameter.get()));
            tempBuilder.Finish(Convolution2D::Pack(tempBuilder, conv2d));
            auto conv2d = flatbuffers::GetRoot<Convolution2D>(tempBuilder.GetBufferPointer());
            bool forceFloat = true;
            quanCommon = ConvolutionCommon::load(conv2d, nullptr, true, true);
            // Back to float
            originWeight     = quanCommon->weightFloat.get();
            originWeightSize = quanCommon->weightFloat.size();
        }
        const int weightSize      = originWeightSize;
        std::vector<int8_t> quantizedWeight(weightSize);
        std::vector<float> quantizedWeightScale(outputChannel);
        if (_weightQuantizeMethod == "MAX_ABS"){
            SymmetricQuantizeWeight(originWeight, weightSize, quantizedWeight.data(), quantizedWeightScale.data(), outputChannel, _weightClampValue);
        } else if (_weightQuantizeMethod == "ADMM") {
            QuantizeWeightADMM(originWeight, weightSize, quantizedWeight.data(), quantizedWeightScale.data(), outputChannel, _weightClampValue);
        }
        param->quanParameter = IDSTEncoder::encode(originWeight, quantizedWeightScale, weightSize/channles, channles, false, quantizedWeight.data(), -_weightClampValue);
        param->quanParameter->scaleIn = inputScale;
        param->quanParameter->scaleOut = outputScale;
        if (param->common->relu6) {
            param->common->relu  = true;
            param->common->relu6 = false;
        }
        param->weight.clear();
    }
}

void Calibration::_computeQuantError() {
    std::map<std::string, std::vector<float>> overflowRatiosMap;
    std::map<std::string, std::vector<float>> tensorCosDistanceMap;
    std::map<std::string, std::vector<float>> fakeQuantedFeatures;
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            if (info->type() == "Raster") {
                return true;
            }
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio = _featureInfo[t]->fakeQuantFeature();
                        fakeQuantedFeatures[_featureInfo[t]->name()] = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[_featureInfo[t]->name()].emplace_back(dequantFeatureAndOverflowRatio.second);
                    }
                }
            }
            return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    auto dequantFeatureAndOverflowRatio = _featureInfo[t]->fakeQuantFeature();
                    fakeQuantedFeatures[_featureInfo[t]->name()] = dequantFeatureAndOverflowRatio.first;
                    overflowRatiosMap[_featureInfo[t]->name()].emplace_back(dequantFeatureAndOverflowRatio.second);
                }
            }
        }
        return true;
    };

    MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
        if (info->type() == "Raster") {
            return true;
        }
        for (auto t : nTensors) {
            if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                if (_featureInfoOrigin[t]->visited() == false) {
                    auto name = _featureInfoOrigin[t]->name();
                    float cosDis = _featureInfoOrigin[t]->computeDistance(fakeQuantedFeatures[name]);
                    tensorCosDistanceMap[name].emplace_back(cosDis);
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo afterOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                            const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                if (_featureInfoOrigin[t]->visited() == false) {
                    auto name = _featureInfoOrigin[t]->name();
                    float cosDis = _featureInfoOrigin[t]->computeDistance(fakeQuantedFeatures[name]);
                    tensorCosDistanceMap[name].emplace_back(cosDis);
                }
            }
        }
        return true;
    };
    int count = 0;

    for (const auto& file : _calibrationFiles) {
        count++;
        
        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }
        for (auto& iter : _featureInfoOrigin) {
            iter.second->setVisited(false);
        }
        std::vector<VARP> inputs(mInputNames.size());
        auto netInfo = _module->getInfo();
        if (_inputType == Helper::SEQUENCE) {
            inputs = getModuleInputs(file, netInfo, mInputNames, mInputShape);
            for (int i = 0; i < inputs.size(); ++i) {
                auto input = inputs[i];
                auto name = mInputNames[i];
                std::string fileName = file + "/" + name + ".txt";
                auto inputTensor = (MNN::Tensor*)input->getTensor();
                Helper::preprocessInput(_process.get(), _preprocessConfig, fileName, inputTensor, _inputType);
            }
        } else {
            inputs.resize(1);
            inputs[0] = _Input(netInfo->inputs[0].dim, netInfo->inputs[0].order, netInfo->inputs[0].type);
            inputs[0] = _Convert(inputs[0], netInfo->inputs[0].order);
            auto inputTensor = (MNN::Tensor*)inputs[0]->getTensor();
            Helper::preprocessInput(_process.get(), _preprocessConfig, file, inputTensor, _inputType);
        }
        Express::Executor::getGlobalExecutor()->setCallBack(std::move(before), std::move(after));
        _module->onForward(inputs);
        Express::Executor::getGlobalExecutor()->setCallBack(std::move(beforeOrigin), std::move(afterOrigin));
        _moduleOrigin->onForward(inputs);

        MNN_PRINT("\rcomputeDistance: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n\nDebug info:\n\n");

    for (auto& iter : tensorCosDistanceMap) {
        auto name = iter.first;
        float sumCos = 0.0f, sumOverflow = 0.0f;
        for (int i = 0; i < iter.second.size(); i++) {
            sumCos += iter.second[i];
            sumOverflow += overflowRatiosMap[name][i];
        }
        float avgCosDistance = sumCos / _calibrationFiles.size();
        float avgOverflowRatio = sumOverflow / _calibrationFiles.size();

        MNN_PRINT("%s:  cos distance: %f, overflow ratio: %f\n", name.c_str(), avgCosDistance, avgOverflowRatio);
    }
}

void Calibration::_quantizeModelEMA() {
    auto varMap = Variable::loadMap(_originalModelFile.c_str());
    if (varMap.empty()) {
        MNN_ERROR("Can not load model %s\n", _originalModelFile.c_str());
        return;
    }

    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto varInputs       = Variable::mapToSequence(inputOutputs.first);
    auto varOutputs      = Variable::mapToSequence(inputOutputs.second);
    auto originInfo = varInputs[0]->getInfo();
    auto originFormat = NC4HW4;
    auto originType = halide_type_of<float>();
    std::vector<int> originDims;
    if (nullptr != originInfo) {
        originFormat = originInfo->order;
        originDims = originInfo->dim;
        originType = originInfo->type;
    }
    _module.reset(NN::extract(varInputs, varOutputs, true), Module::destroy);
    NN::turnQuantize(_module.get(), _quant_bits, NN::PerTensor, NN::MovingAverage, _winogradOpt);

    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);

    std::shared_ptr<SGD> solver(new SGD(_module));
    solver->setLearningRate(1e-5);
    solver->setMomentum(0.9f);
    solver->setWeightDecay(0.00004f);

    DLOG(INFO) << "batch size: " << _batch;
    DLOG(INFO) << "quant bits: " << _quant_bits;
    if (_calibrationFileNum < _batch) {
        MNN_ERROR("_calibrationFileNum %d < batch size %d, set batch size as %d\n", _calibrationFileNum, _batch, _calibrationFileNum);
        _batch = _calibrationFileNum;
    }

    const int trainIterations = _calibrationFileNum / _batch;
    _module->clearCache();
    exe->gc(Executor::FULL);

    _module->setIsTraining(true);
    for (int it = 0; it < trainIterations; it++) {
        std::vector<VARP> inputs(varInputs.size()); // inputs[i].dim=[batch, c, h, w]
        int indicesStart = it * _batch;
        int indicesEnd   = indicesStart + _batch;
        // Init batch size inputs
        if (_inputType == Helper::IMAGE) {
            inputs[0] = _Input({_batch, originDims[1], originDims[2], originDims[3]}, originFormat, originType);
        } else {
            for (auto& input: inputs) {
                input = _Input({_batch, originDims[1], originDims[2], originDims[3]}, originFormat, originType);
            }
        }
        // Compose batch input
        for (int k = indicesStart; k < indicesEnd; ++k) {
            const auto file = _calibrationFiles[k];
            if (_inputType == Helper::SEQUENCE) {
                std::map<std::string, std::vector<int>> dyInputShape = mInputShape;
                std::string jsonFile = file + "/" + "input.json";
                std::ifstream f(jsonFile.c_str());
                if (f.is_open()) {
                    rapidjson::Document document;
                    std::ostringstream json_os;
                    json_os << f.rdbuf();
                    document.Parse(json_os.str().c_str());
                    if (document.HasParseError()) {
                        MNN_ERROR("Invalid json: %s\n", jsonFile.c_str());
                    }
                    auto cfgObj = document.GetObject();
                    if (cfgObj.HasMember("inputs")) {
                        auto inputsInfo = document["inputs"].GetArray();
                        for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
                            auto obj = iter->GetObject();
                            std::string name = obj["name"].GetString();
                            if (obj.HasMember("shape")) {
                                auto dims = obj["shape"].GetArray();
                                std::vector<int> shapes;
                                for (auto iter = dims.begin(); iter != dims.end(); iter++) {
                                    shapes.emplace_back(iter->GetInt());
                                }
                                dyInputShape[name] = shapes;
                                dyInputShape[name][0] = _batch;
                            }
                        }
                    }
                }
                
                for (int i = 0; i < inputs.size(); ++i) {
                    auto name = varInputs[i]->name();
                    auto input = _Input(dyInputShape[name], varInputs[i]->getInfo()->order, varInputs[i]->getInfo()->type);
                    std::string fileName = file + "/" + name + ".txt";
                    
                    auto inputTensor = (MNN::Tensor*)input->getTensor();
                    Helper::preprocessInput(_process.get(), _preprocessConfig, fileName, inputTensor, _inputType);
                    ::memcpy(input->writeMap<float>(), inputTensor->host<float>(), inputTensor->elementSize() * sizeof(float));
                    inputs[i] = input;
                }
            } else {
                auto singleInput = _Input(originDims, originFormat, originType);
                auto inputTensor = (MNN::Tensor*)singleInput->getTensor();
                Helper::preprocessInput(_process.get(), _preprocessConfig, file, inputTensor, _inputType);
                ::memcpy(inputs[0]->writeMap<float>() + k * inputTensor->elementSize(), inputTensor->host<float>(), inputTensor->elementSize() * sizeof(float));
                
            }
        }
        auto predicts = _module->onForward(inputs);
        for (auto &output: predicts) {
            auto ptr = output->readMap<float>();
        }
        MNN_PRINT("\rquantize with EMA: %.2lf %%", (it + 1) * 100.0f / trainIterations);
        fflush(stdout);
        solver->step(_Scalar<float>(0.0f));
    }
    MNN_PRINT("\n");

    _module->setIsTraining(false);
    exe->gc(Executor::PART);
    std::vector<VARP> inputsForward;
    inputsForward.resize(varInputs.size());
    for (int i = 0; i < inputsForward.size(); ++i) {
        auto name = varInputs[i]->name();
        auto input = _Input(mInputShape[name], varInputs[i]->getInfo()->order, varInputs[i]->getInfo()->type);
        input->setName(name);
        inputsForward[i] = input;
    }
    
    auto predicts = _module->onForward(inputsForward);
    
    Transformer::turnModelToInfer()->onExecute(predicts);
    for (int i = 0; i < predicts.size(); i++) {
        predicts[i]->setName(varOutputs[i]->name());
    }
    Variable::save(predicts, _destModelFile.c_str());
    ConvertToFullQuant::convert(_destModelFile);

    std::unique_ptr<MNN::NetT> netT;
    {
        std::ifstream input(_destModelFile, std::ifstream::in | std::ifstream::binary);
        std::ostringstream outputOs;
        outputOs << input.rdbuf();
        netT = MNN::UnPackNet(outputOs.str().c_str());
    }
    ComputeUnaryBuffer(netT.get());
    {
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        builderOutput.ForceDefaults(true);
        auto len = MNN::Net::Pack(builderOutput, netT.get());
        builderOutput.Finish(len);
        std::ofstream output(_destModelFile, std::ofstream::binary);
        output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
    }
}

void Calibration::runQuantizeModel() {
    if (_featureQuantizeMethod == "EMA") {
        _quantizeModelEMA();
        return;
    }

    if (_featureQuantizeMethod == "KL") {
        _computeFeatureScaleKL();
    } else if (_featureQuantizeMethod == "ADMM") {
        _computeFeatureScaleADMM();
    }
    if (_debug) {
        _computeQuantError();
    }
    _insertScale();
    ComputeUnaryBuffer(_originalModel);

    {
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        builderOutput.ForceDefaults(true);
        auto len = MNN::Net::Pack(builderOutput, _originalModel);
        builderOutput.Finish(len);
        std::ofstream output(_destModelFile, std::ofstream::binary);
        output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
    }
}

void Calibration::dumpTensorScales(const std::string& modelFile) {
    rapidjson::StringBuffer sb;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);

    writer.StartArray();

    for (auto iter = _originalModel->oplists.begin(); iter != _originalModel->oplists.end(); iter++) {
        auto op           = iter->get();
        const auto opType = op->type;
        const auto name   = op->name;

        if (opType == MNN::OpType_Raster) {
            continue;
        }

        writer.StartObject();

        writer.Key("name");
        writer.String(rapidjson::StringRef(name.c_str(), name.size()));

        auto& inputIndexes  = op->inputIndexes;
        const int inputSize = static_cast<int32_t>(inputIndexes.size());

        if (inputSize > 0) {
            writer.Key("inputs");
            writer.StartArray();
            for (int i = 0; i < inputSize; ++i) {
                const auto curInputIndex = inputIndexes[i];

                auto input        = _tensorMap[curInputIndex];
                auto inputOpScale = _scales[input];

                writer.StartObject();
                writer.Key("tensorIndex");
                writer.Int(curInputIndex);

                writer.Key("scales");
                writer.StartArray();
                writer.Double(inputOpScale.first);
                writer.EndArray();

                writer.Key("zeropoint");
                writer.StartArray();
                writer.Double(inputOpScale.second);
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }

        auto& outputIndexes  = op->outputIndexes;
        const int outputSize = static_cast<int32_t>(outputIndexes.size());

        if (outputSize > 0) {
            writer.Key("outputs");
            writer.StartArray();
            for (int i = 0; i < outputSize; ++i) {
                const auto curOutputIndex = outputIndexes[i];

                auto output        = _tensorMap[curOutputIndex];
                auto outputOpScale = _scales[output];

                writer.StartObject();
                writer.Key("tensorIndex");
                writer.Int(curOutputIndex);

                writer.Key("scales");
                writer.StartArray();
                writer.Double(outputOpScale.first);
                writer.EndArray();

                writer.Key("zeropoint");
                writer.StartArray();
                writer.Double(outputOpScale.second);
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }

        writer.EndObject();
    }
    writer.EndArray();

    std::string scaleFile = modelFile + ".json";
    std::ofstream os(scaleFile);
    if (os.is_open()) {
        os << sb.GetString() << std::endl;
        os.close();
    } else {
        std::cerr << "open scale file " << scaleFile << " fail. error code:" << os.failbit << std::endl;
    }
}

typedef VARP (*unaryProc)(VARP input);
static unaryProc selectUnaryProc(int type) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return MNN::Express::_Abs;
        case UnaryOpOperation_SQUARE:
            return MNN::Express::_Square;
        case UnaryOpOperation_NEG:
            return MNN::Express::_Negative;
        case UnaryOpOperation_RSQRT:
            return MNN::Express::_Rsqrt;
        case UnaryOpOperation_EXP:
            return MNN::Express::_Exp;
        case UnaryOpOperation_COS:
            return MNN::Express::_Cos;
        case UnaryOpOperation_SIN:
            return MNN::Express::_Sin;
        case UnaryOpOperation_SIGMOID:
            return MNN::Express::_Sigmoid;
        case UnaryOpOperation_TANH:
            return MNN::Express::_Tanh;
        case UnaryOpOperation_TAN:
            return MNN::Express::_Tan;
        case UnaryOpOperation_ATAN:
            return MNN::Express::_Atan;
        case UnaryOpOperation_SQRT:
            return MNN::Express::_Sqrt;
        case UnaryOpOperation_RECIPROCAL:
            return MNN::Express::_Reciprocal;
        case UnaryOpOperation_LOG1P:
            return MNN::Express::_Log1p;
        case UnaryOpOperation_LOG:
            return MNN::Express::_Log;
        case UnaryOpOperation_ACOSH:
            return MNN::Express::_Acosh;
        case UnaryOpOperation_SINH:
            return MNN::Express::_Sinh;
        case UnaryOpOperation_ASINH:
            return MNN::Express::_Asinh;
        case UnaryOpOperation_ATANH:
            return MNN::Express::_Atanh;
        case UnaryOpOperation_SIGN:
            return MNN::Express::_Sign;
        case UnaryOpOperation_COSH:
            return MNN::Express::_Cosh;
        case UnaryOpOperation_ERF:
            return MNN::Express::_Erf;
        case UnaryOpOperation_ERFC:
            return MNN::Express::_Erfc;
        case UnaryOpOperation_ERFINV:
            return MNN::Express::_Erfinv;
        case UnaryOpOperation_EXPM1:
            return MNN::Express::_Expm1;
        case UnaryOpOperation_ASIN:
            return MNN::Express::_Asin;
        case UnaryOpOperation_ACOS:
            return MNN::Express::_Acos;
        case UnaryOpOperation_HARDSWISH:
            return MNN::Express::_Hardswish;
        case UnaryOpOperation_GELU:
            return MNN::Express::_Gelu;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}
void Calibration::ComputeUnaryBuffer(MNN::NetT* net) {
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); ++iter) {
        auto op = iter->get();
        auto opType = op->type;
        std::map<int, TensorDescribeT*> describes;
        for (auto& des : _originalModel->extraTensorDescribe) {
            describes.insert(std::make_pair(des->index, des.get()));
        }
        if (opType == MNN::OpType_Sigmoid || opType == MNN::OpType_TanH) {
            op->type = OpType_UnaryOp;
            op->main.value = new UnaryOpT;
            op->main.type = OpParameter_UnaryOp;
            op->main.AsUnaryOp()->opType = UnaryOpOperation_SIGMOID;
            if (opType == MNN::OpType_TanH) {
                op->main.AsUnaryOp()->opType = UnaryOpOperation_TANH;
            }
            opType = op->type;
        }
        if (opType == MNN::OpType_UnaryOp) {
            auto type = op->main.AsUnaryOp()->opType;
            if (type == UnaryOpOperation_ABS || type == UnaryOpOperation_NEG || type == UnaryOpOperation_SIGN) {
                continue;
            }

            auto outputId = op->outputIndexes[0];
            if (describes.find(outputId) == describes.end()) {
                MNN_PRINT("Can't find input extraTensorDescribe for %s\n", op->name.c_str());
                continue;
            }
            auto unaryDes = describes.find(outputId)->second;
            float outScale = unaryDes->quantInfo->scale;
            float outZero  = unaryDes->quantInfo->zero;
            auto inputId = op->inputIndexes[0];
            if (describes.find(inputId) == describes.end()) {
                MNN_PRINT("Can't find input extraTensorDescribe for %s\n", op->name.c_str());
                continue;
            }
            op->main.AsUnaryOp()->tableInt8.resize(255);
            auto unaryParam = op->main.AsUnaryOp()->tableInt8.data();
            unaryDes = describes.find(inputId)->second;
            float inpScale = unaryDes->quantInfo->scale;
            float inpZero  = unaryDes->quantInfo->zero;

            // Read input data.
            std::vector<float> dataInput;
            float fx = 0.f;
            auto input = _Input({255}, NCHW, halide_type_of<float>());
            input->setName("input_tensor");
            auto ptr_in = input->template writeMap<float>();
            for (int i = -127; i <= 127; ++i) {
                fx = (i - inpZero) * inpScale;
                dataInput.push_back(fx);
                ptr_in[i + 127] = fx;
            }
            input->unMap();
            // Compute output data.
            VARP output;
            auto func = selectUnaryProc(type);
            if (nullptr == func) {
                MNN_ERROR("Don't support quantizing UnaryOP: %s to Int8\n", op->name.c_str());
            }
            output = func(input);
            auto gotOutput = output->template readMap<float>();
            // Write output data.
            int val;
            for (int i = 0; i < 255; ++i) {
                val = (int)roundf(gotOutput[i] / outScale) + outZero;
                if (val > 127) {
                    val = 127;
                }
                if (val < -127) {
                    val = -127;
                }
                unaryParam[i] = val;
                            }
        }
    }
}

int quant_main(int argc, const char* argv[]) {
    if (argc < 4) {
        DLOG(INFO) << "Usage: ./quantized.out src.mnn dst.mnn preTreatConfig.json\n";
        return 0;
    }
    const char* modelFile      = argv[1];
    const char* preTreatConfig = argv[3];
    const char* dstFile        = argv[2];
    DLOG(INFO) << ">>> modelFile: " << modelFile;
    DLOG(INFO) << ">>> preTreatConfig: " << preTreatConfig;
    DLOG(INFO) << ">>> dstFile: " << dstFile;
    std::unique_ptr<MNN::NetT> netT;
    {
        std::shared_ptr<MNN::Interpreter> interp(MNN::Interpreter::createFromFile(modelFile), MNN::Interpreter::destroy);
        if (nullptr == interp.get()) {
            return 0;
        }
        netT = MNN::UnPackNet(interp->getModelBuffer().first);
    }

    // temp build net for inference
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, netT.get());
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto ocontent = builder.GetBufferPointer();

    // model buffer for creating mnn Interpreter
    std::unique_ptr<uint8_t> modelForInference(new uint8_t[size]);
    memcpy(modelForInference.get(), ocontent, size);

    std::unique_ptr<uint8_t> modelOriginal(new uint8_t[size]);
    memcpy(modelOriginal.get(), ocontent, size);

    netT.reset();
    netT = MNN::UnPackNet(modelOriginal.get());

    // quantize model's weight
    DLOG(INFO) << "Calibrate the feature and quantize model...";
    std::shared_ptr<Calibration> calibration(
        new Calibration(netT.get(), modelForInference.get(), size, preTreatConfig, std::string(modelFile), std::string(dstFile)));
    if (!calibration->valid()) {
        return 0;
    }
    calibration->runQuantizeModel();
    calibration->dumpTensorScales(dstFile);
    DLOG(INFO) << "Quantize model done!";

    return 0;
}
