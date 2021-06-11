//
//  liteConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LITECONVERTER_HPP
#define LITECONVERTER_HPP

#include <fstream>
#include <iostream>

#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"

// MNN fbs header
#include "MNN_generated.h"
// tflite fbs header
#include "schema_generated.h"
#include <MNN/MNNDefine.h>

class TfliteModel {
public:
    TfliteModel() = delete;

    TfliteModel(const std::string fileName);
    ~TfliteModel();

    void readModel();

    inline std::unique_ptr<tflite::ModelT>& get();

private:
    const std::string _modelName;
    std::unique_ptr<tflite::ModelT> _tfliteModel;
};

/**
 * @brief convert tflite model to MNN model
 * @param inputModel tflite model name
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
MNN_PUBLIC int tflite2MNNNet(const std::string inputModel, const std::string bizCode,
                  std::unique_ptr<MNN::NetT>& MNNNetT);

#endif // LITECONVERTER_HPP
