//
//  writeFb.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WRITEFB_HPP
#define WRITEFB_HPP

#include <MNN/MNNDefine.h>
#include "MNN_generated.h"
#include "config.hpp"
#include <map>

/**
 *@brief save MNN net to file
 *@param MNNModelFile save mnn model path
 *@param benchmarkModel benchmarkModel is true, then delete the weight of Convolution etc.
 *@param saveHalfFloat when saveHalfFloat is true, save weight in half float data type
 */
MNN_PUBLIC int writeFb(std::unique_ptr<MNN::NetT>& netT, const std::string& MNNModelFile, const modelConfig& config);

#endif // WRITEFB_HPP
