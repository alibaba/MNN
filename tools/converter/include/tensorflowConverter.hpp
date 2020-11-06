//
//  tensorflowConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TENSORFLOWCONVERTER_HPP
#define TENSORFLOWCONVERTER_HPP

#include <string>

#include "options.hpp"
#include "MNN_generated.h"
/**
 * @brief convert tensorflow model to MNN model
 * @param inputModel tensorflow model name(xx.pb)
 * @param bizCode(not used, always is MNN)
 * @param options(converter common options)
 * @param MNN net
 */
int tensorflow2MNNNet(const std::string inputModel, const std::string bizCode,
                      const common::Options& options,
                      std::unique_ptr<MNN::NetT>& netT);

#endif // TENSORFLOWCONVERTER_HPP
