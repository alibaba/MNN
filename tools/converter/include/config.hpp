//
//  config.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <string>
#include <MNN/MNNDefine.h>
#include <fstream>
class MNN_PUBLIC modelConfig {
public:
    modelConfig()
        : MNNModel(),
          prototxtFile(),
          modelFile(),
          bizCode("MNN"),
          model(modelConfig::MAX_SOURCE),
          saveHalfFloat(false){
    }
    enum MODEL_SOURCE { TENSORFLOW = 0, CAFFE, ONNX, MNN, TFLITE, TORCH, JSON, MAX_SOURCE };

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
    bool saveHalfFloat;
    bool forTraining = false;
    int weightQuantBits = 0;// If weightQuantBits > 0, it means the bit
    bool weightQuantAsymmetric = true;
    int weightQuantBlock = -1;
    // The path of the model compression file that stores the int8 calibration table
    // or sparse parameters.
    std::string compressionParamsFile = "";
    bool saveStaticModel = false;
    int optimizePrefer = 0;
    float targetVersion = (float)MNN_VERSION_MAJOR + (float)MNN_VERSION_MINOR * 0.1f;
    int defaultBatchSize = 0;
    int optimizeLevel = 1;
    bool keepInputFormat = true;
    bool alignDenormalizedValue = true;
    bool detectSparseSpeedUp = true;
    bool convertMatmulToConv = true;
    bool transformerFuse = false;
    std::string customOpLibs = "";
    std::string authCode = "";
    std::string testDir = "";
    std::string testConfig;
    float testThredhold = 0.01;
    bool mnn2json = false;
    bool dumpInfo = false;
    bool saveExternalData = false;
    bool inSubGraph = false;
    // using external data when convert
    int64_t externalTreshold = 1024 * 64;
    std::ofstream* externalFile = nullptr;
    int64_t externalOffset = 0;
};

#endif // CONFIG_HPP
