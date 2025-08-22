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
#include <MNN/MNNDefine.h>
#include <memory>
#include "MNN_generated.h"

/**
 * @brief convert tflite model to MNN model
 * @param inputModel tflite model name
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
int tflite2MNNNet(const std::string inputModel, const std::string bizCode,
                  std::unique_ptr<MNN::NetT>& MNNNetT);
bool dumpTflite2Json(const char* inputModel, const char* outputJson);

#endif // LITECONVERTER_HPP
