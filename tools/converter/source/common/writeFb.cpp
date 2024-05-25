//
//  writeFb.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <iostream>
#include <algorithm>
#include <set>
#include <string>
#include <sstream>

#include "MNN_generated.h"
#include "logkit.h"
#include "writeFb.hpp"
#include "CommonUtils.hpp"
#include "cpp/ConfigFile.hpp"
#include <MNN/MNNDefine.h>
#include "cli.hpp"
#include "commonKit.hpp"
#include "MNN_compression.pb.h"

using namespace MNN;
using namespace std;

static void _postTreatOp(std::unique_ptr<OpT>& op, FileLoader* fl, MNN::Compression::Pipeline proto, const modelConfig& config, std::ofstream& weightPath, int64_t& offset, bool needExternalWeight) {
    loadExternalParam(op, fl);
    if (config.alignDenormalizedValue) {
        AlignDenormalizedValue(op);
    }
    if (config.saveHalfFloat) {
        CastParamsToHalf(op);
    }
    if (config.detectSparseSpeedUp) {
        AddSparseInfo(op, proto);
    }
    WeightQuantAndCoding(op, config);
    if (needExternalWeight) {
        RemoveAndStoreParam(op, &weightPath, offset);
    }
}
static float _computeOpExternalSizeInMB(const MNN::OpT* op) {
    switch (op->main.type) {
        case MNN::OpParameter_Convolution2D:
        {
            auto conv2D = op->main.AsConvolution2D();
            if (conv2D->external.empty()) {
                return 0.0f;
            }
            return ((float)conv2D->external[1] + (float)conv2D->external[2]) / 1024.0f / 1024.0f;
        }
        case MNN::OpParameter_Blob:
        {
            auto blob = op->main.AsBlob();
            if (blob->external.empty()) {
                return 0.0f;
            }
            return blob->external[1] / 1024.0f / 1024.0f;
        }
            
        default:
            break;
    }
    return 0.0f;
}
static bool _largeModel(const MNN::NetT* netT) {
    float summer = 0.0f;
    for (auto& op : netT->oplists) {
        summer+= _computeOpExternalSizeInMB(op.get());
        if (summer > 2000.0f) {
            MNN_PRINT("Model larger than 2GB\n");
            return true;
        }
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            summer+= _computeOpExternalSizeInMB(op.get());
            if (summer > 2000.0f) {
                MNN_PRINT("Model larger than 2GB\n");
                return true;
            }
        }
    }
    return false;
}
int writeFb(std::unique_ptr<MNN::NetT>& netT, const std::string& MNNModelFile, const modelConfig& config) {
    std::string compressFileName = config.compressionParamsFile;
    Compression::Pipeline proto;
    if (compressFileName != "") {
        string jsonSuffix = "json";
        string suffix = compressFileName.substr(compressFileName.find_last_of('.') + 1);
        if (jsonSuffix.compare(suffix) != 0) { // protobuf.bin file
            std::fstream input(compressFileName.c_str(), std::ios::in | std::ios::binary);
            if (!proto.ParseFromIstream(&input)) {
                MNN_ERROR("Failed to parse compression pipeline proto.\n");
            }
        } else {
            CommonKit::json2protobuf(compressFileName.c_str(), nullptr, &proto);
        }
    }

    addUUID(netT, proto);

    // add version info to model
    netT->extraInfo.reset(new ExtraInfoT);
    netT->extraInfo->version = MNN_VERSION;
    if (!config.authCode.empty()) {
        // add auth code to model
        netT->extraInfo->name = config.authCode;
    }

    if (config.compressionParamsFile != "") {
        channelPruneConvert(netT, proto);
    }
    if (config.compressionParamsFile != "") {
        fullQuantAndCoding(netT, proto);
    }
    // Check If need external weight
    bool needExternalWeight = config.saveExternalData;
    if (!needExternalWeight) {
        needExternalWeight = _largeModel(netT.get());
    }
    std::ofstream externalWeightOs;
    if (needExternalWeight) {
        auto weightName = MNNModelFile + ".weight";
        MNN_PRINT("Save Weight to %s\n", weightName.c_str());
        externalWeightOs.open(weightName.c_str());
        if (externalWeightOs.fail()) {
            MNN_PRINT("Write %s failed\n", weightName.c_str());
        }
    }
    {
        int64_t offset = 0;
        FileLoader fl(".__convert_external_data.bin");
        for (auto& op : netT->oplists) {
            _postTreatOp(op, &fl, proto, config, externalWeightOs, offset, needExternalWeight);
        }
        for (auto& subgraph : netT->subgraphs) {
            for (auto& op : subgraph->nodes) {
                _postTreatOp(op, &fl, proto, config, externalWeightOs, offset, needExternalWeight);
            }
        }
    }
    {
        std::ofstream erase(".__convert_external_data.bin");
        erase << "0";
    }
    std::set<std::string> notSupportOps;
    auto CheckIfNotSupported = [&] (const std::unique_ptr<MNN::OpT>& op) {
        if (op->type == MNN::OpType_Extra) {
            if (op->main.AsExtra()->engine != "MNN") {
                notSupportOps.insert(op->main.AsExtra()->engine + "::" + op->main.AsExtra()->type);
            }
        }
    };
    for (auto& op : netT->oplists) {
        CheckIfNotSupported(op);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            CheckIfNotSupported(op);
        }
    }

    std::ostringstream notSupportInfo;
    if (!notSupportOps.empty()) {
        for (auto name : notSupportOps) {
            notSupportInfo << name << " | ";
        }
        auto opNames = notSupportInfo.str();
        LOG(FATAL) << "These Op Not Support: " << opNames.substr(0, opNames.size() - 2);
        return 1;
    }

    // dump input and output tensor name
    {
        std::set<int> inputIdx, outputIdx, realInput, realOutput;
        for (const auto& op : netT->oplists) {
            for (auto i : op->inputIndexes) {
                inputIdx.insert(i);
            }
            for (auto o : op->outputIndexes) {
                outputIdx.insert(o);
                if (op->type == OpType_Input) {
                    realInput.insert(o);
                }
            }
        }
        std::set_difference(outputIdx.begin(), outputIdx.end(), inputIdx.begin(), inputIdx.end(), std::inserter(realOutput, realOutput.begin()));
        std::cout << "inputTensors : [ ";
        for (int i : realInput) {
            std::cout << netT->tensorName[i] << ", ";
        }
        std::cout << "]\noutputTensors: [ ";
        if (netT->outputName.size() > 0) {
            for (auto& o : netT->outputName) {
                std::cout << o << ", ";
            }
        } else {
            for (int i : realOutput) {
                std::cout << netT->tensorName[i] << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);
    int sizeOutput    = builderOutput.GetSize();
    auto bufferOutput = builderOutput.GetBufferPointer();

    if (config.saveStaticModel && netT->usage != MNN::Usage_INFERENCE_STATIC) {
        std::map<std::string, std::vector<int>> inputConfig;
        // get config to set input size
        if (config.inputConfigFile.size() > 0) {
            ConfigFile conf(config.inputConfigFile);
            auto numOfInputs = conf.Read<int>("input_size");
            auto inputNames  = splitNames(numOfInputs, conf.Read<std::string>("input_names"));
            auto inputDims   = splitDims(numOfInputs, conf.Read<std::string>("input_dims"));
            for (int i = 0; i < numOfInputs; i++) {
                inputConfig.insert(std::make_pair(inputNames[i], inputDims[i]));
            }
        }
        const Net* net = flatbuffers::GetRoot<MNN::Net>(bufferOutput);
        converToStaticModel(net, inputConfig, MNNModelFile);
    } else {
        std::ofstream output(MNNModelFile, std::ofstream::binary);
        output.write((const char*)bufferOutput, sizeOutput);
    }
    if (!netT->subgraphs.empty()) {
        MNN_PRINT("The model has subgraphs, please use MNN::Express::Module to run it\n");
    }

#ifdef MNN_DUMP_SUBGRAPH
    for (int i = 0; i < netT->subgraphs.size(); ++i) {
        std::unique_ptr<MNN::NetT> subnet(new MNN::NetT);
        auto& subgraph = netT->subgraphs[i];
        subnet->oplists = std::move(subgraph->nodes);
        subnet->tensorName = subgraph->tensors;
        subnet->sourceType = netT->sourceType;
        subnet->bizCode = netT->bizCode;

        flatbuffers::FlatBufferBuilder builder(1024);
        builder.ForceDefaults(true);
        auto len = MNN::Net::Pack(builder, subnet.get());
        builder.Finish(len);
        int output_size = builder.GetSize();
        auto* output_ptr = builder.GetBufferPointer();

        std::string filename =
            MNNModelFile + "_subgraph_" + std::to_string(i) + ".mnn";
        std::ofstream output(filename.c_str(), std::ofstream::binary);
        output.write((const char*)output_ptr, output_size);
    }
#endif
    return 0;
}
