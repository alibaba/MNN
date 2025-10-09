//
//  onnxConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ONNXCONVERTER_HPP
#define ONNXCONVERTER_HPP
#include <MNN/MNNDefine.h>
#include "MNN_generated.h"

/**
 * @brief convert ONNX model to MNN model
 * @param inputModel ONNX model name(xxx.onnx)
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
int onnx2MNNNet(const std::string inputModel, const std::string bizCode,
                std::unique_ptr<MNN::NetT>& netT, MNN::OpT* meta, std::vector<std::string>& inputNames);

#endif // ONNXCONVERTER_HPP
