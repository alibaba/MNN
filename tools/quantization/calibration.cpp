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
#include <set>
#include "flatbuffers/util.h"
#include "ImageProcess.hpp"
#include "logkit.h"
#include "quantizeWeight.hpp"
#include "rapidjson/document.h"
//#define MNN_OPEN_TIME_TRACE
#include <dirent.h>
#include <sys/stat.h>
#include "AutoTime.hpp"
#include "Helper.hpp"
using namespace MNN::CV;

Calibration::Calibration(MNN::NetT* model, uint8_t* modelBuffer, const int bufferSize, const std::string& configPath)
    : _originaleModel(model) {
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
            static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}};
            if (formatMap.find(format) != formatMap.end()) {
                config.destFormat = formatMap.find(format)->second;
            }
        }
    }
    config.sourceFormat = RGBA;
    std::string imagePath;
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
            imagePath = picObj["path"].GetString();
        }
        if (picObj.HasMember("used_image_num")) {
            _imageNum = picObj["used_image_num"].GetInt();
        }
        if (picObj.HasMember("feature_quantize_method")) {
            std::string method = picObj["feature_quantize_method"].GetString();
            if (Helper::featureQuantizeMethod.find(method) != Helper::featureQuantizeMethod.end()){
                _featureQuantizeMethod = method;
            }
            else {
                MNN_ERROR("not supported feature quantization method: %s\n", method.c_str());
                return;
            }
        }
        if (picObj.HasMember("weight_quantize_method")) {
            std::string method = picObj["weight_quantize_method"].GetString();
            if (Helper::weightQuantizeMethod.find(method) != Helper::weightQuantizeMethod.end()){
                _weightQuantizeMethod = method;
            }
            else {
                MNN_ERROR("not supported weight quantization method: %s\n", method.c_str());
                return;
            }
        }
        DLOG(INFO) << "use feature quantization method: " << _featureQuantizeMethod;
        DLOG(INFO) << "use weight quantization method: " << _weightQuantizeMethod;
    }
    std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
    _process = process;

    // read images file names
    Helper::readImages(_imgaes, imagePath.c_str(), _imageNum);

    _initMNNSession(modelBuffer, bufferSize);
    _initMaps();
}

void Calibration::_initMNNSession(const uint8_t* modelBuffer, const int bufferSize) {
    _interpreter.reset(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    MNN::ScheduleConfig config;
    _session     = _interpreter->createSession(config);
    _inputTensor = _interpreter->getSessionInput(_session, NULL);

    if (_featureQuantizeMethod == "KL"){
        _interpreter->resizeTensor(_inputTensor, 1, _inputTensor->channel(), _height, _width);
        _interpreter->resizeSession(_session);
    }
    else if (_featureQuantizeMethod == "ADMM") {
        _interpreter->resizeTensor(_inputTensor, _imageNum, _inputTensor->channel(), _height, _width);
        _interpreter->resizeSession(_session);
    }
    _interpreter->releaseModel();
}

void Calibration::_initMaps() {
    _featureInfo.clear();
    _opInfo.clear();
    _tensorMap.clear();
    // run mnn once, initialize featureMap, opInfo map
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        _opInfo[info->name()].first = nTensors;
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] = std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, info->name() + "__input"));
                }
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo after = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        _opInfo[info->name()].second = nTensors;
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] = std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, info->name()));
                }
            }
        }
        return true;
    };
    _interpreter->runSessionWithCallBackInfo(_session, before, after);

    for (auto& op : _originaleModel->oplists) {
        if (_opInfo.find(op->name) == _opInfo.end()) {
            continue;
        }
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            _tensorMap[op->inputIndexes[i]] = _opInfo[op->name].first[i];
        }
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            _tensorMap[op->outputIndexes[i]] = _opInfo[op->name].second[i];
        }
    }

    if (_featureQuantizeMethod == "KL") {
        // set the tensor-statistic method of input tensor as THRESHOLD_MAX
        auto inputTensorStatistic = _featureInfo.find(_inputTensor);
        DCHECK(inputTensorStatistic != _featureInfo.end()) << "input tensor error!";
        inputTensorStatistic->second->setThresholdMethod(THRESHOLD_MAX);
    }
}

void Calibration::_computeFeatureMapsRange() {
    // feed input data according to input images
    int count = 0;
    for (const auto& img : _imgaes) {
        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedRangeFlags();
        }
        count++;
        Helper::preprocessInput(_process.get(), _width, _height, img, _inputTensor);

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    _featureInfo[t]->updateRange();
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    _featureInfo[t]->updateRange();
                }
            }
            return true;
        };

        _interpreter->runSessionWithCallBackInfo(_session, before, after);
    }
}

void Calibration::_collectFeatureMapsDistribution() {
    for (auto& iter : _featureInfo) {
        iter.second->resetDistribution();
    }
    // feed input data according to input images
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                _featureInfo[t]->updateDistribution();
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                _featureInfo[t]->updateDistribution();
            }
        }
        return true;
    };
    for (const auto& img : _imgaes) {
        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedDistributionFlag();
        }
        Helper::preprocessInput(_process.get(), _width, _height, img, _inputTensor);
        _interpreter->runSessionWithCallBackInfo(_session, before, after);
    }
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
    std::vector<int> shape = {_imageNum, _inputTensor->channel(), _height, _width};
    
    for (const auto& img : _imgaes) {
        auto ptr = _inputTensor->host<float>() + count * _inputTensor->stride(0);
        std::shared_ptr<MNN::Tensor> tensorWarp(MNN::Tensor::create(shape, _inputTensor->getType(), ptr, MNN::Tensor::CAFFE_C4));
        Helper::preprocessInput(_process.get(), _width, _height, img, tensorWarp.get());
        
        count++;
    }

    _scales.clear();

    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    DLOG(INFO) << "start processing input tensors of op: " << info->name();
                    DLOG(INFO) << "info->type(): " << info->type();
                    _scales[t] = _featureInfo[t]->computeScaleADMM();
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                            const MNN::OperatorInfo* info) {
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    DLOG(INFO) << "start processing output tensors of op: " << info->name();
                    DLOG(INFO) << "info->type(): " << info->type();
                    _scales[t] = _featureInfo[t]->computeScaleADMM();
                }
            }
        }
        return true;
    };

    _interpreter->runSessionWithCallBackInfo(_session, before, after);
}

void Calibration::_updateScale() {
    for (const auto& op : _originaleModel->oplists) {
        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise) {
            continue;
        }
        auto tensorsPair = _opInfo.find(op->name);
        if (tensorsPair == _opInfo.end()) {
            MNN_ERROR("Can't find tensors for %s\n", op->name.c_str());
        }

        const auto& inputScale  = _scales[tensorsPair->second.first[0]];
        const auto& outputScale = _scales[tensorsPair->second.second[0]];

        auto param                = op->main.AsConvolution2D();
        param->common->inputCount = tensorsPair->second.first[0]->channel();
        const int channles        = param->common->outputCount;
        const int weightSize      = param->weight.size();
        param->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        auto& quantizedParam = param->symmetricQuan;
        quantizedParam->scale.resize(channles);
        quantizedParam->weight.resize(weightSize);
        quantizedParam->bias.resize(channles);

        if (opType == MNN::OpType_Convolution) {
            QuantizeConvPerChannel(param->weight.data(), param->weight.size(), param->bias.data(),
                                   quantizedParam->weight.data(), quantizedParam->bias.data(),
                                   quantizedParam->scale.data(), inputScale, outputScale, _weightQuantizeMethod);
            op->type = MNN::OpType_ConvInt8;

        } else {
            QuantizeDepthwiseConv(param->weight.data(), param->weight.size(), param->bias.data(),
                                  quantizedParam->weight.data(), quantizedParam->bias.data(),
                                  quantizedParam->scale.data(), inputScale, outputScale, _weightQuantizeMethod);
            op->type = MNN::OpType_DepthwiseConvInt8;
        }
        if (param->common->relu6) {
            param->common->relu  = true;
            param->common->relu6 = false;
        }
        param->weight.clear();
        param->bias.clear();
    }
}

void Calibration::_insertDequantize() {
    // Search All Int Tensors
    std::set<int> int8Tensors;
    std::set<int> int8Outputs;
    for (auto& op : _originaleModel->oplists) {
        if (Helper::INT8SUPPORTED_OPS.count(op->type) > 0) {
            for (auto index : op->inputIndexes) {
                int8Tensors.insert(index);
            }
            for (auto index : op->outputIndexes) {
                int8Tensors.insert(index);
                int8Outputs.insert(index);
            }
        }
    }
    for (auto& op : _originaleModel->oplists) {
        for (auto index : op->inputIndexes) {
            auto iter = int8Outputs.find(index);
            if (iter != int8Outputs.end()) {
                int8Outputs.erase(iter);
            }
        }
    }

    // Insert Convert For Not Support Int8 Ops
    for (auto iter = _originaleModel->oplists.begin(); iter != _originaleModel->oplists.end();) {
        auto op           = iter->get();
        const auto opType = op->type;
        const auto name   = op->name;
        // check whether is output op
        // if Yes, insert dequantization op after this op
        if (Helper::INT8SUPPORTED_OPS.find(opType) != Helper::INT8SUPPORTED_OPS.end()) {
            // this is quantized op
            iter++;
            continue;
        }

        auto& inputIndexes  = op->inputIndexes;
        const int inputSize = inputIndexes.size();

        // insert dequantization op before this op
        for (int i = 0; i < inputSize; ++i) {
            const auto curInputIndex = inputIndexes[i];
            if (int8Tensors.find(curInputIndex) == int8Tensors.end()) {
                continue;
            }
            auto input        = _tensorMap[curInputIndex];
            auto inputOpScale = _scales[input];

            // construct new op
            auto dequantizationOp       = new MNN::OpT;
            dequantizationOp->main.type = MNN::OpParameter_QuantizedFloatParam;
            dequantizationOp->name      = "___Int8ToFloat___For_" + name;

            dequantizationOp->type           = MNN::OpType_Int8ToFloat;
            auto dequantizationParam         = new MNN::QuantizedFloatParamT;
            dequantizationOp->main.value     = dequantizationParam;
            dequantizationParam->tensorScale = inputOpScale;

            dequantizationOp->inputIndexes.push_back(curInputIndex);
            dequantizationOp->outputIndexes.push_back(_originaleModel->tensorName.size());
            _originaleModel->tensorName.push_back(dequantizationOp->name);

            // reset current op's input index at i
            inputIndexes[i] = dequantizationOp->outputIndexes[0];

            iter = _originaleModel->oplists.insert(iter, std::unique_ptr<MNN::OpT>(dequantizationOp));
            iter++;
        }

        iter++;
        // LOG(INFO) << "insert quantization op after this op if neccessary";
        // insert quantization op after this op if neccessary
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            const auto outputIndex = op->outputIndexes[i];
            if (int8Tensors.find(outputIndex) == int8Tensors.end()) {
                continue;
            }
            auto output   = _tensorMap[outputIndex];
            auto curScale = _scales[output];
            // construct one quantization op(FloatToInt8)
            auto quantizationOp        = new MNN::OpT;
            quantizationOp->main.type  = MNN::OpParameter_QuantizedFloatParam;
            quantizationOp->name       = name + "___FloatToInt8___";
            quantizationOp->type       = MNN::OpType_FloatToInt8;
            auto quantizationParam     = new MNN::QuantizedFloatParamT;
            quantizationOp->main.value = quantizationParam;

            const int channels = curScale.size();
            std::vector<float> quantizationScale(channels);
            for (int i = 0; i < channels; ++i) {
                const auto scale = curScale[i];
                if (scale == .0f) {
                    quantizationScale[i] = 0.0f;
                } else {
                    quantizationScale[i] = 1.0 / scale;
                }
            }
            quantizationParam->tensorScale = quantizationScale;

            quantizationOp->inputIndexes.push_back(_originaleModel->tensorName.size());
            quantizationOp->outputIndexes.push_back(outputIndex);
            _originaleModel->tensorName.push_back(_originaleModel->tensorName[op->outputIndexes[i]]);
            _originaleModel->tensorName[op->outputIndexes[i]] = quantizationOp->name;
            op->outputIndexes[i]                              = quantizationOp->inputIndexes[i];

            iter = _originaleModel->oplists.insert(iter, std::unique_ptr<MNN::OpT>(quantizationOp));
            iter++;
        }
    }

    // Insert Turn float Op for output
    for (auto index : int8Outputs) {
        // construct new op
        auto dequantizationOp       = new MNN::OpT;
        dequantizationOp->main.type = MNN::OpParameter_QuantizedFloatParam;
        dequantizationOp->name      = "___Int8ToFloat___For_" + flatbuffers::NumToString(index);

        dequantizationOp->type           = MNN::OpType_Int8ToFloat;
        auto dequantizationParam         = new MNN::QuantizedFloatParamT;
        dequantizationOp->main.value     = dequantizationParam;
        dequantizationParam->tensorScale = _scales[_tensorMap[index]];

        dequantizationOp->inputIndexes.push_back(index);
        dequantizationOp->outputIndexes.push_back(_originaleModel->tensorName.size());
        auto originTensorName              = _originaleModel->tensorName[index];
        _originaleModel->tensorName[index] = dequantizationOp->name;
        _originaleModel->tensorName.emplace_back(originTensorName);

        _originaleModel->oplists.insert(_originaleModel->oplists.end(), std::unique_ptr<MNN::OpT>(dequantizationOp));
    }
}
void Calibration::runQuantizeModel() {
    if (_featureQuantizeMethod == "KL") {
        _computeFeatureScaleKL();
    }
    else if (_featureQuantizeMethod == "ADMM") {
        _computeFeatureScaleADMM();
    }
    _updateScale();
    _insertDequantize();
}
