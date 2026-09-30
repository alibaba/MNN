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
#include <MNN/expr/Module.hpp>
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
    std::map<std::string, std::shared_ptr<MNN::Express::Module>> cacheModules;
    MNNForwardType accelerateType = MNN_FORWARD_CPU;
    MNN::BackendConfig bnConfig;
    int mode = 1;
    void startOptimize();
    void endOptimize();
};

int converToStaticModel(const MNN::Net* net, std::map<std::string, std::vector<int>>& inputConfig, std::string mnnFile,
                        const std::string& externalFile);
void RemoveAndStoreParam(std::unique_ptr<MNN::OpT>& op, std::ofstream* fs, int64_t& offset);
bool loadExternalParam(std::unique_ptr<MNN::OpT>& op, MNN::FileLoader* fl);
bool validateIndexedLinearExternal(const MNN::OpT* op, const MNN::FileLoader* fl);
void CastParamsToHalf(std::unique_ptr<MNN::OpT>& op);
void AlignDenormalizedValue(std::unique_ptr<MNN::OpT>& op);
void AddSparseInfo(std::unique_ptr<MNN::OpT>& op, MNN::Compression::Pipeline proto);
void fullQuantAndCoding(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);
void WeightQuantAndCoding(std::unique_ptr<MNN::OpT>& op, const modelConfig& config, const PostTreatContext* context);

void addUUID(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);
void channelPruneConvert(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto);

// Expose indexed weights to existing convolution weight utilities without
// creating child graph nodes or changing their serialized quantization format.
template <typename Visitor>
void forEachIndexedLinearWeight(std::unique_ptr<MNN::OpT>& op, Visitor visitor) {
    auto param = op->main.AsIndexedLinearParam();
    for (size_t i = 0; i < param->linears.size(); ++i) {
        auto& linear = param->linears[i];
        if (!linear) {
            continue;
        }
        std::unique_ptr<MNN::OpT> member(new MNN::OpT);
        member->type = MNN::OpType_Convolution;
        member->name = op->name + "/linear." + std::to_string(i);
        member->externalPath = op->externalPath;
        member->inputIndexes = {0};
        member->main.type = MNN::OpParameter_Convolution2D;
        member->main.value = linear.release();
        visitor(member);
        linear.reset(member->main.AsConvolution2D());
        member->main.value = nullptr;
        member->main.type = MNN::OpParameter_NONE;
    }
}

class CommonKit {
public:
    static void loadCompress(modelConfig& modelPath);
    static bool FileIsExist(std::string path);
    static bool json2protobuf(const char* jsonFile, const char* protoFile=nullptr, MNN::Compression::Pipeline* pipeline=nullptr);
    static bool protobuf2json(const char* jsonFile, const MNN::Compression::Pipeline* pipeline);
};

#endif // COMMMON_UTILS_HPP
