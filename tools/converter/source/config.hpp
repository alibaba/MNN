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

#include "CONFIGURECONVERT.h"

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
          benchmarkModel(false)
          {
    }
    enum MODEL_SOURCE { TENSORFLOW = 0, CAFFE, ONNX, MNN, TFLITE, MAX_SOURCE };

    // MNN model path
    std::string MNNModel;
    // if model is tensorflow, this value is NULL;
    std::string prototxtFile;
    // tensorflow pb, or caffe model
    std::string modelFile;
    // bizCode
    std::string bizCode;
    // model source
    MODEL_SOURCE model;
    bool benchmarkModel;
};

#endif // CONFIG_HPP
