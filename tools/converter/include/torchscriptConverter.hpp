//
//  torchscriptConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TORCHSCRIPTCONVERTER_HPP
#define TORCHSCRIPTCONVERTER_HPP

#include "MNN_generated.h"

/**
 * @brief convert TorchScript model to MNN model
 * @param inputModel TorchScript model name(xxx.pt)
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
int torchscript2MNNNet(const std::string inputModel, const std::string bizCode,
                       std::unique_ptr<MNN::NetT>& netT);

#endif // TORCHSCRIPTCONVERTER_HPP
