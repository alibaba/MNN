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
#include "../source/core/FileLoader.hpp"

struct PostTreatContext {
    MNN::Compression::Pipeline proto;
    std::map<std::pair<std::string, std::string>, const MNN::Compression::LayerQuantizeParams*> quantInfo;
    MNN::Compression::QuantizeParams* quantMutableInfo = nullptr;
    std::string subgraph;
    bool read = false;
    bool write = false;
};

void converToStaticModel(const MNN::Net* net, std::map<std::string,std::vector<int>>& inputConfig, std::string mnnFile);
void RemoveAndStoreParam(std::unique_ptr<MNN::OpT>& op, std::ofstream* fs, int64_t& offset);
void loadExternalParam(std::unique_ptr<MNN::OpT>& op, MNN::FileLoader* fl);
void CastParamsToHalf(std::unique_ptr<MNN::OpT>& op);
void AlignDenormalizedValue(std::unique_ptr<MNN::OpT>& op);
void AddSparseInfo(std::unique_ptr<MNN::OpT>& op, MNN::Compression::Pipeline proto);
void fullQuantAndCoding(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);
void WeightQuantAndCoding(std::unique_ptr<MNN::OpT>& op, const modelConfig& config, const PostTreatContext* context);

void addUUID(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);
void channelPruneConvert(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);

class CommonKit {
public:
    static void loadCompress(modelConfig& modelPath);
    static bool FileIsExist(std::string path);
    static bool json2protobuf(const char* jsonFile, const char* protoFile=nullptr, MNN::Compression::Pipeline* pipeline=nullptr);
    static bool protobuf2json(const char* jsonFile, const MNN::Compression::Pipeline* pipeline);
};

#endif // COMMMON_UTILS_HPP
