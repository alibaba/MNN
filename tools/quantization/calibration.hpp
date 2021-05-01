//
//  calibration.hpp
//  MNN
//
//  Created by MNN on 2019/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP

#include <map>

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include "TensorStatistic.hpp"
#include "MNN_generated.h"
#include "Helper.hpp"

// Calibration find the optimal threshold according to KL-divergence
// process: the below process is applied on the whole Conv|DepthwiseConv layers
// 1. run the model on the batch samples, update the max(abs(feature_maps)) when the op is Convolution|Depthwise
// 2. cut the max(abs(feature_maps)) into 2048 slices
// 3. run the model on the batch samples again, update the distribution of feature maps every Conv|DepthwiseConv layer
// 4. apply Calibration on every distribution to get the optimal thereshold
// 5. compute the (input_scale * weight_scale) / output_scale, update the scale of symmetricQuan in Convolution Paramter
class Calibration {
public:
    Calibration(MNN::NetT* model, const uint8_t* modelBuffer, const int bufferSize, const std::string& configPath, std::string originalModelFile, std::string dstModelFile);

    void runQuantizeModel();
    
    void dumpTensorScales(const std::string& modelFile);

private:
    Calibration();
    MNN::NetT* _originalModel;
    std::shared_ptr<MNN::CV::ImageProcess> _process;
    const int _binNums = 2048;
    int _calibrationFileNum      = 0;
    int _width;
    int _height;
    int _channels;
    int _batch = 32;
    int _quant_bits = 8;
    Helper::PreprocessConfig _preprocessConfig;
    Helper::InputType _inputType;
    std::string _calibrationFilePath;
    std::string _originalModelFile;
    std::string _destModelFile;
    MNN::CV::ImageProcess::Config _imageProcessConfig;
    std::vector<std::string> _calibrationFiles;

    // Tensor and Info
    std::map<const MNN::Tensor*, std::shared_ptr<TensorStatistic>> _featureInfo;
    std::map<const MNN::Tensor*, std::shared_ptr<TensorStatistic>> _featureInfoOrigin;
    std::map<int, const MNN::Tensor*> _tensorMap;
    std::map<const MNN::Tensor*, int> _tensorIdx;

    // Op's name, Inputs, Outputs
    std::map<std::string, std::pair<std::vector<MNN::Tensor*>, std::vector<MNN::Tensor*>>> _opInfo;

    // The scale results
    std::map<const MNN::Tensor*, float> _scales;

    std::shared_ptr<MNN::Interpreter> _interpreter;
    // keep mnn forward information
    MNN::Session* _session;
    MNN::Tensor* _inputTensor;
    std::vector<int> _inputTensorDims;

    std::shared_ptr<MNN::Interpreter> _interpreterOrigin;
    MNN::Session* _sessionOrigin;
    MNN::Tensor* _inputTensorOrigin;

    std::string _featureQuantizeMethod = "KL";
    std::string _weightQuantizeMethod  = "MAX_ABS";

    float _featureClampValue = 127.0f;
    float _weightClampValue = 127.0f;
    std::vector<std::string> _skip_quant_ops;
    bool _debug = false;

    std::vector<int> _getInputShape(std::string filename);
    void _resizeIfNeeded(std::string filename, bool force = false);
    void _initMNNSession(const uint8_t* modelBuffer, const int bufferSize);
    void _initMaps();

    // compute min/max value for every Tensor
    void _computeFeatureMapsRange();
    void _collectFeatureMapsDistribution();
    void _computeFeatureScaleKL();
    void _computeFeatureScaleADMM();
    void _quantizeModelEMA();
    void _computeFeatureScaleMoving();
    void _fake_quant_weights();
    void _computeQuantError();
    void _insertScale();
};

#endif // CALIBRATION_HPP
