//
//  config.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <mutex>
#include <string>

class ProjectConfig {
public:
    static const std::string version;
    static const std::string buildTime;

    static ProjectConfig *obtainSingletonInstance();

private:
    ProjectConfig() = default;

private:
    static ProjectConfig *m_pConfig;
    static std::mutex m_mutex;
};

struct modelConfig {
public:
    modelConfig()
        : MNNModel(),
          prototxtFile(),
          modelFile(),
          bizCode("MNN"),
          model(modelConfig::MAX_SOURCE),
          benchmarkModel(false),
          saveHalfFloat(false){
    }
    enum MODEL_SOURCE { TENSORFLOW = 0, CAFFE, ONNX, MNN, TFLITE, TORCHSCRIPT, MAX_SOURCE };

    // MNN model path
    std::string MNNModel;
    // if model is tensorflow, this value is NULL;
    std::string prototxtFile;
    // tensorflow pb, or caffe model
    std::string modelFile;
    // bizCode
    std::string bizCode;
    // input config file
    std::string inputConfigFile;
    // model source
    MODEL_SOURCE model;
    bool benchmarkModel;
    bool saveHalfFloat;
    bool forTraining = false;
    int weightQuantBits = 0;// If weightQuantBits > 0, it means the bit
    bool weightQuantAsymmetric = false;
    // The path of the model compression file that stores the int8 calibration table
    // or sparse parameters.
    std::string compressionParamsFile = "";
    bool saveStaticModel = false;
    int optimizePrefer = 0;
    float targetVersion = 1.2;
};

#endif // CONFIG_HPP
