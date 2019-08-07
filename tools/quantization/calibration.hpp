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

#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "TensorStatistic.hpp"
#include "converter/source/IR/MNN_generated.h"

// Calibration find the optimal threshold according to KL-divergence
// process: the below process is applied on the whole Conv|DepthwiseConv layers
// 1. run the model on the batch samples, update the max(abs(feature_maps)) when the op is Convolution|Depthwise
// 2. cut the max(abs(feature_maps)) into 2048 slices
// 3. run the model on the batch samples again, update the distribution of feature maps every Conv|DepthwiseConv layer
// 4. apply Calibration on every distribution to get the optimal thereshold
// 5. compute the (input_scale * weight_scale) / output_scale, update the scale of symmetricQuan in Convolution Paramter
class Calibration {
public:
    Calibration(MNN::NetT* model, uint8_t* modelBuffer, const int bufferSize, const std::string& configPath);

    void runQuantizeModel();

private:
    Calibration();
    MNN::NetT* _originaleModel;
    std::shared_ptr<MNN::CV::ImageProcess> _process;
    const int _binNums = 2048;
    int _imageNum = 0;
    int _width;
    int _height;
    std::vector<std::string> _imgaes;

    // Tensor and Info
    std::map<const MNN::Tensor*, std::shared_ptr<TensorStatistic>> _featureInfo;
    std::map<int, const MNN::Tensor*> _tensorMap;

    // Op's name, Inputs, Outputs
    std::map<std::string, std::pair<std::vector<MNN::Tensor*>, std::vector<MNN::Tensor*>>> _opInfo;

    // The scale results
    std::map<const MNN::Tensor*, std::vector<float>> _scales;

    std::shared_ptr<MNN::Interpreter> _interpreter;
    // keep mnn forward information
    MNN::Session* _session;
    MNN::Tensor* _inputTensor;

    std::string _featureQuantizeMethod = "KL";
    std::string _weightQuantizeMethod = "MAX_ABS";

    void _initMNNSession(const uint8_t* modelBuffer, const int bufferSize);
    void _initMaps();

    void _computeFeatureMapsRange();
    void _collectFeatureMapsDistribution();
    void _computeFeatureScaleKL();
    void _computeFeatureScaleADMM();
    void _updateScale();

    // insert the dequantization op before the not supported op(int8), and insert dequantization op
    // after the output op, so that get original float data conveniently
    void _insertDequantize();
};

#endif // CALIBRATION_HPP
