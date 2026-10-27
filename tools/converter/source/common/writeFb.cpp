//
//  writeFb.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <set>
#include <string>
#include <sstream>
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "MNN_generated.h"
#include "core/MNNFileUtils.h"
#include "logkit.h"
#include "writeFb.hpp"
#include "CommonUtils.hpp"
#include "cpp/ConfigFile.hpp"
#include <MNN/MNNDefine.h>
#include "cli.hpp"
#include "MNN_compression.pb.h"

using namespace MNN;
using namespace std;

// Stage the complete output before replacing either half of a model/weight
// pair. Backups permit rollback on ordinary I/O failures; multiple renames are
// not a crash-atomic filesystem transaction.
class OutputFiles {
public:
    struct File {
        std::string target, staged, backup;
        bool backedUp = false;
        bool installed = false;
    };
    ~OutputFiles() {
        for (const auto& file : mFiles) {
            MNNRemoveFile(file.staged.c_str());
        }
    }
    std::string stage(const std::string& target) {
        const auto staged = target + ".tmp";
        const auto backup = target + ".bak";
        if (target.empty() || MNNDirExist(target.c_str()) || MNNFileExist(staged.c_str()) ||
            MNNFileExist(backup.c_str())) {
            MNN_ERROR("Cannot stage output: %s\n", target.c_str());
            return {};
        }
        for (const auto& file : mFiles) {
            if (target == file.target || target == file.staged || target == file.backup || staged == file.target ||
                backup == file.target) {
                MNN_ERROR("Conflicting output paths: %s\n", target.c_str());
                return {};
            }
        }
        File file;
        file.target = target;
        file.staged = staged;
        file.backup = backup;
        mFiles.emplace_back(std::move(file));
        return staged;
    }
    bool commit() {
        for (auto& file : mFiles) {
            if (MNNFileExist(file.target.c_str())) {
                if (std::rename(file.target.c_str(), file.backup.c_str()) != 0) {
                    rollback();
                    return false;
                }
                file.backedUp = true;
            }
        }
        for (auto& file : mFiles) {
            if (std::rename(file.staged.c_str(), file.target.c_str()) != 0) {
                rollback();
                return false;
            }
            file.installed = true;
        }
        for (const auto& file : mFiles) {
            if (file.backedUp)
                MNNRemoveFile(file.backup.c_str());
        }
        return true;
    }

private:
    void rollback() {
        for (auto iter = mFiles.rbegin(); iter != mFiles.rend(); ++iter) {
            if (iter->installed)
                MNNRemoveFile(iter->target.c_str());
            if (iter->backedUp && std::rename(iter->backup.c_str(), iter->target.c_str()) != 0) {
                // Keep the original bytes recoverable if even restoration fails.
                MNN_ERROR("Cannot restore %s; original saved at %s\n", iter->target.c_str(), iter->backup.c_str());
            }
        }
    }
    std::vector<File> mFiles;
};

static bool _postTreatOp(std::unique_ptr<OpT>& op, FileLoader* fl, const PostTreatContext& context,
                         const modelConfig& config, std::ofstream& weightPath, int64_t& offset,
                         bool needExternalWeight, bool indexedExpert = false) {
    if (op->main.type == OpParameter_IndexedLinearParam) {
        if (config.model == modelConfig::MNN) {
            bool hasExternal = false;
            bool hasFloatWeight = false;
            auto param = op->main.AsIndexedLinearParam();
            for (const auto& expert : param->linears) {
                if (expert) {
                    hasExternal |= !expert->external.empty();
                    hasFloatWeight |= !expert->quanParameter;
                }
            }
            if (hasExternal && op->externalPath.empty()) {
                op->externalPath = config.modelFile + ".weight";
            }
            // Already quantized banks need no weight post-treatment. Preserve
            // their external storage during the exporter's in-place graph pass.
            if (hasExternal && !hasFloatWeight && !needExternalWeight && config.modelFile == config.MNNModel) {
                if (!validateIndexedLinearExternal(op.get(), fl)) {
                    return false;
                }
                if (op->externalPath == config.modelFile + ".weight") {
                    op->externalPath.clear();
                }
                return true;
            }
        }
        if (!loadExternalParam(op, fl)) {
            return false;
        }
        bool success = true;
        forEachIndexedLinearWeight(op, [&](std::unique_ptr<OpT>& member) {
            success = success && _postTreatOp(member, fl, context, config, weightPath, offset, false, true);
        });
        if (!success) {
            return false;
        }
        if (needExternalWeight) {
            RemoveAndStoreParam(op, &weightPath, offset);
        }
        return !needExternalWeight || weightPath.good();
    }
    if (!loadExternalParam(op, fl)) {
        return false;
    }
    if (config.alignDenormalizedValue) {
        AlignDenormalizedValue(op);
    }
    if (config.saveHalfFloat) {
        CastParamsToHalf(op);
    }
    // IndexedLinear uses dense expert kernels and permits bias-free projections.
    if (config.detectSparseSpeedUp && !indexedExpert) {
        AddSparseInfo(op, context.proto);
    }
    WeightQuantAndCoding(op, config, &context);
    if (needExternalWeight) {
        RemoveAndStoreParam(op, &weightPath, offset);
    }
    return !needExternalWeight || weightPath.good();
}
static float _computeOpExternalSizeInMB(const MNN::OpT* op) {
    switch (op->main.type) {
        case MNN::OpParameter_IndexedLinearParam:
        {
            double size = 0.0;
            auto param = op->main.AsIndexedLinearParam();
            for (const auto& expert : param->linears) {
                if (!expert) {
                    continue;
                }
                for (size_t i = 1; i < expert->external.size(); ++i) {
                    size += expert->external[i];
                }
            }
            return size / 1024.0 / 1024.0;
        }
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
static int postTreat(std::unique_ptr<MNN::NetT>& netT, const modelConfig& config, OutputFiles& files,
                     std::string& stagedWeight) {
    AUTOTIME;
    std::string compressFileName = config.compressionParamsFile;
    auto& proto = config.compressInfo->proto;
    auto MNNModelFile = config.MNNModel;

    addUUID(netT, config.compressInfo->proto);

    bool useOriginQuant = config.compressInfo->proto.algo_size() > 0 && (!config.compressInfo->write);
    if (config.compressInfo->proto.has_for_guide() && config.compressInfo->proto.for_guide()) {
        useOriginQuant = false;
    }
    if (useOriginQuant) {
        channelPruneConvert(netT, proto);
    }
    if (useOriginQuant) {
        fullQuantAndCoding(netT, proto);
    }
    // Check If need external weight
    bool needExternalWeight = config.saveExternalData;
    if ((!needExternalWeight) && config.model != modelConfig::MNN) {
        needExternalWeight = _largeModel(netT.get());
    }
    std::ofstream externalWeightOs;
    if (needExternalWeight) {
        auto weightName = config.MNNModel + ".weight";
        MNN_PRINT("Save Weight to %s\n", weightName.c_str());
        // Never truncate a file that an input op may still reference.
        stagedWeight = files.stage(weightName);
        if (stagedWeight.empty())
            return 1;
        externalWeightOs.open(stagedWeight.c_str(), ios::binary);
        if (!externalWeightOs.is_open()) {
            MNN_ERROR("Failed to open external weights: %s\n", stagedWeight.c_str());
            return 1;
        }
    }
    {
        bool findQuant = false;
        auto& context = *config.compressInfo;
        for (int i=0; i<proto.algo_size(); ++i) {
            auto& algo = proto.algo(i);
            if (algo.type() != MNN::Compression::CompressionAlgo_CompressionType_QUANTIZE) {
                continue;
            }
            if (!algo.has_quant_params()) {
                continue;
            }
            findQuant = true;
            for (int j=0; j<algo.quant_params().layer_size(); ++j) {
                const auto& quant = algo.quant_params().layer(j);
                std::string opName;
                if (!quant.has_op_name()) {
                    continue;
                }
                opName = quant.op_name();
                std::string graphName;
                if (quant.has_subgraph_name()) {
                    graphName = quant.subgraph_name();
                }
                context.quantInfo.insert(std::make_pair(std::make_pair(graphName, opName), &quant));
            }
            break;
        }
        if ((!findQuant) && context.write) {
            // Add Quant param for write
            proto.set_version(MNN_VERSION);
            proto.set_for_guide(true);
            auto algo = proto.add_algo();
            algo->set_type( MNN::Compression::CompressionAlgo_CompressionType_QUANTIZE);
            context.quantMutableInfo = algo->mutable_quant_params();
        }
        int64_t offset = 0;
        FileLoader fl(".__convert_external_data.bin");
        for (auto& op : netT->oplists) {
            if (!_postTreatOp(op, &fl, context, config, externalWeightOs, offset, needExternalWeight)) {
                return 1;
            }
        }
        for (auto& subgraph : netT->subgraphs) {
            context.subgraph = subgraph->name;
            for (auto& op : subgraph->nodes) {
                if (!_postTreatOp(op, &fl, context, config, externalWeightOs, offset, needExternalWeight)) {
                    return 1;
                }
            }
        }
    }
    if (needExternalWeight) {
        // close() also checks buffered writes that may fail after write() succeeds.
        externalWeightOs.close();
        if (externalWeightOs.fail()) {
            MNN_ERROR("Failed to write external weights: %s.weight\n", config.MNNModel.c_str());
            return 1;
        }
    }
    {
        MNNRemoveFile(".__convert_external_data.bin");
    }
    if (config.compressInfo->write) {
        const auto stagedCompression = files.stage(compressFileName);
        if (stagedCompression.empty() || !CommonKit::protobuf2json(stagedCompression.c_str(), &proto))
            return 1;
    }
    return 0;
}
int writeFb(std::unique_ptr<MNN::NetT>& netT, const modelConfig& config, std::unique_ptr<MNN::OpT>&& metaOp) {
    OutputFiles files;
    std::string stagedWeight;
    auto error = postTreat(netT, config, files, stagedWeight);
    if (error != 0) {
        return error;
    }
    // Merge Meta to metaOp
    auto oplist = std::move(netT->oplists);
    for (auto& op : oplist) {
        if (op->type == OpType_Extra) {
            auto dstExtra = metaOp->main.AsExtra();
            auto extra = op->main.AsExtra();
            if (extra->type == "Meta" && extra->engine == "MNN") {
                for (auto& attr : extra->attr) {
                    dstExtra->attr.emplace_back(std::move(attr));
                }
                // Remove meta op
                continue;
            }
        }
        netT->oplists.emplace_back(std::move(op));
    }
    std::set<std::string> notSupportOps;
    
    // Detect unsupport op
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
    if (!notSupportOps.empty() && !config.allowCustomOp) {
        for (auto name : notSupportOps) {
            notSupportInfo << name << " | ";
        }
        auto opNames = notSupportInfo.str();
        LOG(FATAL) << "These Op Not Support: " << opNames.substr(0, opNames.size() - 2);
        return 1;
    }

    // dump input and output tensor name
    {
        std::set<int> inputIdx, outputIdx, realOutput;
        std::vector<int> realInput;
        for (const auto& op : netT->oplists) {
            for (auto i : op->inputIndexes) {
                inputIdx.insert(i);
            }
            for (auto o : op->outputIndexes) {
                outputIdx.insert(o);
                if (op->type == OpType_Input) {
                    realInput.emplace_back(o);
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
    // add version info to model
    netT->extraInfo.reset(new ExtraInfoT);
    netT->extraInfo->version = MNN_VERSION;
    if (!config.authCode.empty()) {
        // add auth code to model
        netT->extraInfo->name = config.authCode;
    }
    if (metaOp->main.AsExtra()->attr.size() > 0) {
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Extra::Pack(builder, metaOp->main.AsExtra()));
        netT->extraInfo->buffer.resize(builder.GetSize());
        ::memcpy(netT->extraInfo->buffer.data(), builder.GetBufferPointer(), builder.GetSize());
    }
    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);
    int sizeOutput    = builderOutput.GetSize();
    auto bufferOutput = builderOutput.GetBufferPointer();

    const auto stagedModel = files.stage(config.MNNModel);
    if (stagedModel.empty())
        return 1;
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
        if (converToStaticModel(net, inputConfig, stagedModel,
                                stagedWeight.empty() ? config.MNNModel + ".weight" : stagedWeight) != 0) {
            return 1;
        }
    } else {
        std::ofstream output(stagedModel, std::ofstream::binary);
        if (!output.is_open()) {
            MNN_ERROR("Failed to open output model: %s\n", config.MNNModel.c_str());
            return 1;
        }
        output.write((const char*)bufferOutput, sizeOutput);
        // close() also checks buffered writes that may fail after write() succeeds.
        output.close();
        if (output.fail()) {
            MNN_ERROR("Failed to write output model: %s\n", config.MNNModel.c_str());
            return 1;
        }
    }
    if (!files.commit()) {
        MNN_ERROR("Cannot commit output model: %s\n", config.MNNModel.c_str());
        return 1;
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
