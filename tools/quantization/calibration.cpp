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
#define STB_IMAGE_IMPLEMENTATION
#include "ImageProcess.hpp"
#include "logkit.h"
#include "quantizeWeight.hpp"
#include "rapidjson/document.h"
#include "stb_image.h"
//#define MNN_OPEN_TIME_TRACE
#include <dirent.h>
#include <sys/stat.h>
#include "AutoTime.hpp"
using namespace MNN::CV;
static std::set<std::string> gNeedFeatureOp = {"Convolution", "ConvolutionDepthwise"};

inline bool fileExist(const std::string& file) {
    struct stat buffer;
    return stat(file.c_str(), &buffer) == 0;
}

static void readImages(std::vector<std::string>& images, const std::string& filePath) {
    DIR* root = opendir(filePath.c_str());
    if (root == NULL) {
        DLOG(FATAL) << "Open " << filePath << "Failed!";
    }
    struct dirent* ent = readdir(root);
    while (ent != NULL) {
        if (ent->d_name[0] != '.') {
            const std::string fileName = filePath + "/" + ent->d_name;
            if (fileExist(fileName)) {
                // std::cout << "==> " << fileName << std::endl;
                images.push_back(fileName);
            }
        }
        ent = readdir(root);
    }
}

static void preprocessInput(MNN::CV::ImageProcess* pretreat, int targetWidth, int targetHeight,
                            const std::string& inputImageFileName, MNN::Tensor* input) {
    int originalWidth, originalHeight, comp;
    auto bitmap32bits = stbi_load(inputImageFileName.c_str(), &originalWidth, &originalHeight, &comp, 4);

    DCHECK(bitmap32bits != nullptr) << "input image error!";
    MNN::CV::Matrix trans;
    trans.setScale((float)(originalWidth - 1) / (float)(targetWidth - 1),
                   (float)(originalHeight - 1) / (float)(targetHeight - 1));
    // trans.setTranslate(16.0f, 16.0f);
    pretreat->setMatrix(trans);
    pretreat->convert(bitmap32bits, originalWidth, originalHeight, 0, input);

    stbi_image_free(bitmap32bits);
}

const std::set<MNN::OpType> Calibration::_INT8SUPPORTED_OPS = {
    MNN::OpType_ConvInt8, MNN::OpType_DepthwiseConvInt8, MNN::OpType_PoolInt8,
    // MNN::OpType_Int8ToFloat,
    // MNN::OpType_FloatToInt8,
};

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
    }
    std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
    _process = process;

    // read images file names
    readImages(_imgaes, imagePath.c_str());

    _initMNNSession(modelBuffer, bufferSize);
    _initMaps();
}

void Calibration::_initMNNSession(const uint8_t* modelBuffer, const int bufferSize) {
    _interpreter.reset(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    MNN::ScheduleConfig config;
    _session     = _interpreter->createSession(config);
    _inputTensor = _interpreter->getSessionInput(_session, NULL);
    _interpreter->resizeTensor(_inputTensor, 1, _inputTensor->channel(), _height, _width);
    _interpreter->resizeSession(_session);
    _interpreter->releaseModel();
}

void Calibration::_initMaps() {
    _featureInfo.clear();
    _opInfo.clear();
    _tensorMap.clear();
    // run mnn once, initialize featureMap, opInfo map
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        _opInfo[info->name()].first = nTensors;
        if (gNeedFeatureOp.find(info->type()) != gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] = std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _binNums));
                }
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo after = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        _opInfo[info->name()].second = nTensors;
        if (gNeedFeatureOp.find(info->type()) != gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] = std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _binNums));
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

    // set the tensor-statistic method of input tensor is THRESHOLD_MAX
    auto inputTensorStatistic = _featureInfo.find(_inputTensor);
    DCHECK(inputTensorStatistic != _featureInfo.end()) << "input tensor error!";
    inputTensorStatistic->second->setThresholdMethod(THRESHOLD_MAX);
}

void Calibration::_computeFeatureMapsRange() {
    // feed input data according to input images
    int count = 0;
    for (const auto& img : _imgaes) {
        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedRangeFlags();
        }
        count++;
        preprocessInput(_process.get(), _width, _height, img, _inputTensor);

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
    DLOG(INFO) << "Total Samples: " << count;
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
        preprocessInput(_process.get(), _width, _height, img, _inputTensor);
        _interpreter->runSessionWithCallBackInfo(_session, before, after);
    }
}

void Calibration::_updateScale() {
    _scales.clear();
    for (auto& iter : _featureInfo) {
        AUTOTIME;
        _scales[iter.first] = iter.second->finishAndCompute();
    }
    //_featureInfo.clear();//No need now
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

        // quantizedParam->tensorScale = outputScale;

        if (opType == MNN::OpType_Convolution) {
            QuantizeConvPerChannel(param->weight.data(), param->weight.size(), param->bias.data(),
                                   quantizedParam->weight.data(), quantizedParam->bias.data(),
                                   quantizedParam->scale.data(), inputScale, outputScale);
            op->type = MNN::OpType_ConvInt8;

        } else {
            QuantizeDepthwiseConv(param->weight.data(), param->weight.size(), param->bias.data(),
                                  quantizedParam->weight.data(), quantizedParam->bias.data(),
                                  quantizedParam->scale.data(), inputScale, outputScale);
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
        if (_INT8SUPPORTED_OPS.count(op->type) > 0) {
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
        if (_INT8SUPPORTED_OPS.find(opType) != _INT8SUPPORTED_OPS.end()) {
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
    _computeFeatureMapsRange();
    _collectFeatureMapsDistribution();
    _updateScale();
    _insertDequantize();
}
