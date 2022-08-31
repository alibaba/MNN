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
#include "MNN_compression.pb.h"

using namespace MNN;
using namespace std;

int writeFb(std::unique_ptr<MNN::NetT>& netT, const std::string& MNNModelFile, const modelConfig& config) {
    std::string compressFileName = config.compressionParamsFile;
    Compression::Pipeline proto;
    if (compressFileName != "") {
        std::fstream input(compressFileName.c_str(), std::ios::in | std::ios::binary);
        if (!proto.ParseFromIstream(&input)) {
            MNN_ERROR("Failed to parse compression pipeline proto.\n");
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

    if (config.benchmarkModel) {
        removeParams(netT);
    }

    if (config.saveHalfFloat) {
        castParamsToHalf(netT);
    }
    if (config.alignDenormalizedValue) {
        AlignDenormalizedValue(netT);
    }
    if (config.detectSparseSpeedUp) {
        addSparseInfo(netT, proto);
    }
    if (config.compressionParamsFile != "") {
        fullQuantAndCoding(netT, proto);
    }

    weightQuantAndCoding(netT, config);


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
        MNN_PRINT("The model has subgraphs, please use MNN::Module to run it\n");
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
