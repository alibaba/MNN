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
#include "core/TensorUtils.hpp"
#include <MNN/expr/Module.hpp>
#include "MNN_generated.h"
#include "Helper.hpp"
#include "logkit.h"


// Calibration find the optimal threshold according to KL-divergence
// process: the below process is applied on the whole Conv|DepthwiseConv layers
// 1. run the model on the batch samples, update the max(abs(feature_maps)) when the op is Convolution|Depthwise
// 2. cut the max(abs(feature_maps)) into 2048 slices
// 3. run the model on the batch samples again, update the distribution of feature maps every Conv|DepthwiseConv layer
// 4. apply Calibration on every distribution to get the optimal thereshold
// 5. compute the (input_scale * weight_scale) / output_scale, update the scale of symmetricQuan in Convolution Paramter
struct WeakPtrCompare {
    template <typename T>
    bool operator()(const std::weak_ptr<T>& a, const std::weak_ptr<T>& b) const {
        // owner_before 比较的是底层控制块的地址，提供了一个稳定的排序依据
        return a.owner_before(b);
    }
};

class Calibration {
public:
    Calibration(MNN::NetT* model, const uint8_t* modelBuffer, const int bufferSize, const std::string& configPath, std::string originalModelFile, std::string dstModelFile);

    void runQuantizeModel();

    void dumpTensorScales(const std::string& modelFile);
    void ComputeUnaryBuffer(MNN::NetT* net);
    bool valid() const {
        return mValid;
    }
private:
    Calibration();
    MNN::NetT* _originalModel;
    std::shared_ptr<MNN::CV::ImageProcess> _process;
    std::map<int, std::unique_ptr<MNN::TensorDescribeT>> _tensorDescribes;
    bool mValid = true;
    const int _binNums = 2048;
    int _calibrationFileNum      = 0;
    int _width = 1;
    int _height = 1;
    int _channels;
    int _batch = 32;
    int _quant_bits = 8;
    bool _winogradOpt = false;
    Helper::PreprocessConfig _preprocessConfig;
    Helper::InputType _inputType;
    std::string _calibrationFilePath;
    std::string _originalModelFile;
    std::string _destModelFile;
    MNN::CV::ImageProcess::Config _imageProcessConfig;
    std::vector<std::string> _calibrationFiles;
    std::vector<std::string> mCalibrationDatasetDir;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
    std::map<std::string, float> mInputInfo;
    std::map<std::string, std::vector<int>> mInputShape;
    std::vector<MNN::Express::VARP> mInputs;
    std::shared_ptr<MNN::Backend> mBackend;

    // Tensor and Info
    std::map<std::weak_ptr<MNN::Tensor::InsideDescribe::NativeInsideDescribe>, std::shared_ptr<TensorStatistic>, WeakPtrCompare> _featureInfo;
    std::map<std::weak_ptr<MNN::Tensor::InsideDescribe::NativeInsideDescribe>, std::shared_ptr<TensorStatistic>, WeakPtrCompare> _featureInfoOrigin;
    std::map<int, std::pair<std::weak_ptr<MNN::Tensor::InsideDescribe::NativeInsideDescribe>, const MNN::Tensor*>> _tensorMap;

    // The scale results
    std::map<std::weak_ptr<MNN::Tensor::InsideDescribe::NativeInsideDescribe>, std::pair<float, int8_t>, WeakPtrCompare> _scales;

    // {opName, {outputDes, inputDes}}
    std::map<std::string, std::pair<std::weak_ptr<MNN::Tensor::InsideDescribe::NativeInsideDescribe>, std::vector<std::weak_ptr<MNN::Tensor::InsideDescribe::NativeInsideDescribe>>>> _rasterTensors;
    std::map<std::string, std::pair<std::weak_ptr<MNN::Tensor::InsideDescribe::NativeInsideDescribe>, std::vector<std::weak_ptr<MNN::Tensor::InsideDescribe::NativeInsideDescribe>>>> _poolTensors;

    // keep mnn forward information
    std::vector<MNN::Tensor*> mInputTensors;
    std::vector<int> _inputTensorDims;

    std::shared_ptr<MNN::Express::Module> _module;
    std::shared_ptr<MNN::Express::Module> _moduleOrigin;

    std::string _featureQuantizeMethod = "KL";
    std::string _weightQuantizeMethod  = "MAX_ABS";

    float _featureClampValue = 127.0f;
    float _weightClampValue = 127.0f;
    std::vector<std::string> _skip_quant_ops;
    bool _debug = false;

    std::vector<int> _getInputShape(std::string filename);
    void _resizeIfNeeded(std::string filename, bool force = false);
    void _initMNNSession(const uint8_t* modelBuffer, const int bufferSize);

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
int quant_main(int argc, const char* argv[]);

#endif // CALIBRATION_HPP
