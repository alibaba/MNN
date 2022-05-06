//
//  CommonUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2021/08/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef COMMMON_UTILS_HPP
#define COMMMON_UTILS_HPP

#include <MNN/MNNDefine.h>
#include "MNN_generated.h"
#include "config.hpp"
#include "MNN_compression.pb.h"
#include <map>

void converToStaticModel(const MNN::Net* net, std::map<std::string,std::vector<int>>& inputConfig, std::string mnnFile);
void removeParams(std::unique_ptr<MNN::NetT>& netT);
void castParamsToHalf(std::unique_ptr<MNN::NetT>& netT);
void AlignDenormalizedValue(std::unique_ptr<MNN::NetT>& netT);
void addSparseInfo(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);
void fullQuantAndCoding(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);
void weightQuantAndCoding(std::unique_ptr<MNN::NetT>& netT, const modelConfig& config);
void addUUID(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);

#endif // COMMMON_UTILS_HPP
