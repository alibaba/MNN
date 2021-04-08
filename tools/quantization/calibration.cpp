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
#include "cpp/IDSTEncoder.hpp"

using namespace MNN::CV;

Calibration::Calibration(MNN::NetT* model, const uint8_t* modelBuffer, const int bufferSize, const std::string& configPath)
    : _originaleModel(model) {
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
            return;
        }
    }
    auto picObj = document.GetObject();
    ImageProcess::Config config;
    config.filterType = BILINEAR;
    config.destFormat = BGR;
    {
        if (picObj.HasMember("format")) {
            auto format = picObj["format"].GetString();
            static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}, {"RGBA", RGBA}, {"BGRA", BGRA}};
            if (formatMap.find(format) != formatMap.end()) {
                config.destFormat = formatMap.find(format)->second;
            }
        }
    }

    switch (config.destFormat) {
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

    config.sourceFormat = RGBA;
    std::string calibrationFilePath;
    _calibrationFileNum = 0;
    {
        if (picObj.HasMember("mean")) {
            auto mean = picObj["mean"].GetArray();
            int cur   = 0;
            for (auto iter = mean.begin(); iter != mean.end(); iter++) {
                config.mean[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("normal")) {
            auto normal = picObj["normal"].GetArray();
            int cur     = 0;
            for (auto iter = normal.begin(); iter != normal.end(); iter++) {
                config.normal[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("width")) {
            _width = picObj["width"].GetInt();
        }
        if (picObj.HasMember("height")) {
            _height = picObj["height"].GetInt();
        }
        if (picObj.HasMember("path")) {
            calibrationFilePath = picObj["path"].GetString();
        }
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
        }
        DLOG(INFO) << "feature_clamp_value: " << _featureClampValue;
        DLOG(INFO) << "weight_clamp_value: " << _weightClampValue;
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
        _inputType = InputType::IMAGE;
        if (picObj.HasMember("input_type")) {
            std::string type = picObj["input_type"].GetString();
            if (type == "sequence") {
                _inputType = InputType::SEQUENCE;
            }
        }
    }
    std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
    _process = process;

    // read images file names
    Helper::readClibrationFiles(_calibrationFiles, calibrationFilePath.c_str(), &_calibrationFileNum);

    _initMNNSession(modelBuffer, bufferSize);
    _initMaps();
}

std::vector<int> Calibration::_getInputShape(std::string filename) {
    std::vector<int> inputShape;
    if (_inputType == InputType::IMAGE) {
        inputShape.resize(4);
        auto inputTensorDataFormat = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
        if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
            inputShape[0] = 1;
            inputShape[1] = _height;
            inputShape[2] = _width;
            inputShape[3] = _channels;
        } else {
            inputShape[0] = 1;
            inputShape[1] = _channels;
            inputShape[2] = _height;
            inputShape[3] = _width;
        }
    }
    if (_inputType == InputType::SEQUENCE) {
        if (!Helper::stringEndWith(filename, ".txt")) {
            MNN_ERROR("Error: only '.txt' files are supported for sequence input.\n");
        }

        std::ifstream f(filename);
        if (!f.is_open()) {
            MNN_ERROR("open file %s failed.\n", filename.c_str());
        }

        std::string line;
        _channels = 0;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            float v;
            int count = 0;
            while (ss >> v) {
                count++;
            }
            if (count > 0) {
                _channels++;
                _height = count;
            }
        }

        if (_channels == 0) {
            MNN_ERROR("Error: no data found in file %s.", filename.c_str());
        }

        inputShape.resize(3);
        auto inputTensorDataFormat = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
        if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
            inputShape[0] = 1;
            inputShape[1] = _height;
            inputShape[2] = _channels;
        } else {
            inputShape[0] = 1;
            inputShape[1] = _channels;
            inputShape[2] = _height;
        }
    }

    return inputShape;
}

void Calibration::_resizeIfNeeded(std::string filename, bool force) {
    std::vector<int> inputShape = _getInputShape(filename);
    
    if ((inputShape != _inputTensorDims && _featureQuantizeMethod == "KL") || force) {
        _inputTensorDims = inputShape;
        _interpreter->resizeTensor(_inputTensor, _inputTensorDims);
        _interpreter->resizeSession(_session);
        _interpreterOrigin->resizeTensor(_inputTensorOrigin, _inputTensorDims);
        _interpreterOrigin->resizeSession(_sessionOrigin);
    }
}

void Calibration::_initMNNSession(const uint8_t* modelBuffer, const int bufferSize) {
    _interpreterOrigin.reset(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    MNN::ScheduleConfig config;
    _sessionOrigin     = _interpreterOrigin->createSession(config);
    _inputTensorOrigin = _interpreterOrigin->getSessionInput(_sessionOrigin, NULL);

    _fake_quant_weights();

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, _originaleModel);
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto buffer = builder.GetBufferPointer();

    _interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    _session     = _interpreter->createSession(config);
    _inputTensor = _interpreter->getSessionInput(_session, NULL);

    if (_featureQuantizeMethod == "ADMM") {
        DCHECK((_calibrationFileNum * 4 * _height * _width) < (INT_MAX / 4)) << "Use Little Number of Images When Use ADMM";
        for (auto file : _calibrationFiles) {
            std::vector<int> sampleShape = _getInputShape(file);
            if (_inputTensorDims.empty()) {
                _inputTensorDims = sampleShape;
            }
            if (sampleShape != _inputTensorDims) {
                MNN_ERROR("samples must have the same shape when using ADMM method for sequence inputs.");
            }
        }
        _inputTensorDims[0] = _calibrationFileNum;
        _interpreter->resizeTensor(_inputTensor, _inputTensorDims);
        _interpreter->resizeSession(_session);
        _interpreterOrigin->resizeTensor(_inputTensorOrigin, _inputTensorDims);
        _interpreterOrigin->resizeSession(_sessionOrigin);
    }

    _resizeIfNeeded(_calibrationFiles[0]);
}

void Calibration::_initMaps() {
    _featureInfo.clear();
    _featureInfoOrigin.clear();
    _opInfo.clear();
    _tensorMap.clear();
    // run mnn once, initialize featureMap, opInfo map
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        _opInfo[opName].first = nTensors;
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
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
        _opInfo[opName].second = nTensors;
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return true;
    };
    _interpreter->runSessionWithCallBackInfo(_session, before, after);


    MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) == _featureInfoOrigin.end()) {
                    _featureInfoOrigin[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo afterOrigin = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) == _featureInfoOrigin.end()) {
                    _featureInfoOrigin[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return true;
    };
    _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

    for (auto& op : _originaleModel->oplists) {
        if (_opInfo.find(op->name) == _opInfo.end()) {
            continue;
        }
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            _tensorMap[op->inputIndexes[i]] = _opInfo[op->name].first[i];
            _tensorIdx[_opInfo[op->name].first[i]] = op->inputIndexes[i];
        }
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            _tensorMap[op->outputIndexes[i]] = _opInfo[op->name].second[i];
            _tensorIdx[_opInfo[op->name].second[i]] = op->outputIndexes[i];
        }
    }

    if (_featureQuantizeMethod == "KL") {
        // set the tensor-statistic method of input tensor as THRESHOLD_MAX
        auto inputTensorStatistic = _featureInfo.find(_inputTensor);
        if (inputTensorStatistic != _featureInfo.end()) {
            inputTensorStatistic->second->setThresholdMethod(THRESHOLD_MAX);
        }
    }
}

void Calibration::_computeFeatureMapsRange() {
    // feed input data according to input images
    int count = 0;
    for (const auto& file : _calibrationFiles) {
        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedRangeFlags();
        }
        count++;
        _resizeIfNeeded(file);
        Helper::preprocessInput(_process.get(), _width, _height, file, _inputTensor, _inputType);

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
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
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };

        _interpreter->runSessionWithCallBackInfo(_session, before, after);
        MNN_PRINT("\rComputeFeatureRange: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
}

void Calibration::_collectFeatureMapsDistribution() {
    for (auto& iter : _featureInfo) {
        iter.second->resetDistribution();
    }
    // feed input data according to input images
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
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
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->updateDistribution();
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

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedDistributionFlag();
        }
        _resizeIfNeeded(file);
        Helper::preprocessInput(_process.get(), _width, _height, file, _inputTensor, _inputType);
        _interpreter->runSessionWithCallBackInfo(_session, before, after);

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
    int count                           = 0;
    std::vector<int> oneImageTensorDims = _inputTensorDims;
    oneImageTensorDims[0]               = 1;
    auto inputTensorDataFormat          = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
    auto dimType                        = MNN::Tensor::CAFFE_C4;
    if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
        dimType = MNN::Tensor::TENSORFLOW;
    }

    for (const auto& file : _calibrationFiles) {
        auto curPtr = _inputTensor->host<float>() + count * _inputTensor->stride(0);
        std::shared_ptr<MNN::Tensor> tensorWarp(
            MNN::Tensor::create(oneImageTensorDims, _inputTensor->getType(), curPtr, dimType));
        Helper::preprocessInput(_process.get(), _width, _height, file, tensorWarp.get(), _inputType);

        count++;
        MNN_PRINT("\rProcessCalibrationFiles: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
    _scales.clear();

    const int totalLayers = _featureInfo.size();
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

    _interpreter->runSessionWithCallBackInfo(_session, before, after);
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

    for (const auto& op : _originaleModel->oplists) {
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
        const int weightSize = weights.size();
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
        describe->index = _tensorIdx[iter.first];
        describe->quantInfo.reset(new MNN::TensorQuantInfoT);
        describe->quantInfo->scale = iter.second;
        describe->quantInfo->type = MNN::DataType_DT_INT8;
        _originaleModel->extraTensorDescribe.emplace_back(std::move(describe));
    }
    for (const auto& op : _originaleModel->oplists) {
        const auto opType = op->type;

        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }
        
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise) {
            continue;
        }
        auto tensorsPair = _opInfo.find(op->name);
        if (tensorsPair == _opInfo.end()) {
            MNN_ERROR("Can't find tensors for %s\n", op->name.c_str());
        }
        // below is Conv/DepthwiseConv weight quant
        const float inputScale  = _scales[tensorsPair->second.first[0]];
        const float outputScale = _scales[tensorsPair->second.second[0]];
        const int inputChannel = tensorsPair->second.first[0]->channel();
        const int outputChannel = tensorsPair->second.second[0]->channel();
        auto param                = op->main.AsConvolution2D();
        param->common->inputCount = tensorsPair->second.first[0]->channel();
        const int channles        = param->common->outputCount;
        const int weightSize      = param->weight.size();
        param->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        param->symmetricQuan->nbits = 8;
        std::vector<int8_t> quantizedWeight(weightSize);
        std::vector<float> quantizedWeightScale(outputChannel);
        if (_weightQuantizeMethod == "MAX_ABS"){
            SymmetricQuantizeWeight(param->weight.data(), weightSize, quantizedWeight.data(), quantizedWeightScale.data(), outputChannel, _weightClampValue);
        } else if (_weightQuantizeMethod == "ADMM") {
            QuantizeWeightADMM(param->weight.data(), weightSize, quantizedWeight.data(), quantizedWeightScale.data(), outputChannel, _weightClampValue);
        }
        param->quanParameter = IDSTEncoder::encode(param->weight, quantizedWeightScale, weightSize/channles, channles, false, quantizedWeight.data(), -_weightClampValue);
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
    int count = 0;
    std::map<std::string, std::vector<float>> overflowRatiosMap;
    std::map<std::string, std::vector<float>> tensorCosDistanceMap;

    for (const auto& file : _calibrationFiles) {
        count++;
        _resizeIfNeeded(file, true);
        Helper::preprocessInput(_process.get(), _width, _height, file, _inputTensor, _inputType);

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

        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        _interpreter->runSessionWithCallBackInfo(_session, before, after);

        Helper::preprocessInput(_process.get(), _width, _height, file, _inputTensorOrigin, _inputType);

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

        for (auto& iter : _featureInfoOrigin) {
            iter.second->setVisited(false);
        }

        _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

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

void Calibration::runQuantizeModel() {
    if (_featureQuantizeMethod == "KL") {
        _computeFeatureScaleKL();
    } else if (_featureQuantizeMethod == "ADMM") {
        _computeFeatureScaleADMM();
    }
    if (_debug) {
        _computeQuantError();
    }
    _insertScale();
}

void Calibration::dumpTensorScales(const std::string& modelFile) {
    rapidjson::StringBuffer sb;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);

    writer.StartArray();

    for (auto iter = _originaleModel->oplists.begin(); iter != _originaleModel->oplists.end(); iter++) {
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
        const int inputSize = inputIndexes.size();

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
                writer.Double(inputOpScale);
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }
 
        auto& outputIndexes  = op->outputIndexes;
        const int outputSize = outputIndexes.size();

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
                writer.Double(outputOpScale);
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
