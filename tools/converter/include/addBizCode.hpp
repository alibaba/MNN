//
//  addBizCode.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ADDBIZCODE_HPP
#define ADDBIZCODE_HPP

#include "options.hpp"
#include "MNN_generated.h"
int addBizCode(const std::string modelFile, const std::string bizCode,
               const common::Options& options, std::unique_ptr<MNN::NetT>& netT);

#endif // ADDBIZCODE_HPP
