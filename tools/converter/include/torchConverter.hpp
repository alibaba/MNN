//
//  torchConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TORCHCONVERTER_HPP
#define TORCHCONVERTER_HPP

#include "MNN_generated.h"

/**
 * @brief convert Torch model to MNN model
 * @param inputModel Torch model name(xxx.pt)
 * @param bizCode(not used, always is MNN)
 * @param MNN net
 */
MNN_PUBLIC int torch2MNNNet(const std::string inputModel, const std::string bizCode,
                            std::unique_ptr<MNN::NetT>& netT, std::string customTorchOps = "");

#endif // TORCHCONVERTER_HPP
