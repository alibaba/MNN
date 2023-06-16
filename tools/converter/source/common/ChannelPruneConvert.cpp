//
//  ChannelPruneConvert.cpp
//  MNNConverter
//
//  Created by MNN on 2023/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include <vector>
#include <map>
#include <set>
#include <algorithm>

using namespace MNN;
using namespace std;

// TODO: add more unsafe ops
static std::vector<MNN::OpType> unSafeOpTypes = {
    OpType_BroadcastTo, OpType_BatchToSpaceND, OpType_Concat, OpType_LSTM, OpType_LSTMBlockCell, OpType_Reshape, OpType_Resize,
    OpType_RNN, OpType_RNNSequenceGRU, OpType_ScatterNd, OpType_Slice, OpType_SliceTf, OpType_SpaceToBatchND, OpType_Raster,
};

struct TensorMaskInfo {
    std::vector<int> mask; // per-channel 1 or 0
    std::string oriConvName;
};

std::vector<MNN::OpT*> findUserOps(int outputIndex, std::unique_ptr<MNN::NetT>& netT, SubGraphProtoT* subgraph) {
    std::vector<MNN::OpT*> userOps;
    if (subgraph) {
        for (auto& subOp : subgraph->nodes) {
            for (int inputIndex : subOp->inputIndexes) {
                if (inputIndex == outputIndex) {
                    userOps.push_back(subOp.get());
                }
            }
        }
    } else {
        for (auto& netOp : netT->oplists) {
            for (int inputIndex : netOp->inputIndexes) {
                if (inputIndex == outputIndex) {
                    userOps.push_back(netOp.get());
                }
            }
        }
    }

    return userOps;
}

// do the actual channel prune on weights and bias
void channelPrune(std::unique_ptr<MNN::OpT>& op, std::unique_ptr<MNN::NetT>& netT, SubGraphProtoT* subgraph, std::map<std::string, TensorMaskInfo>& tensorMaskInfo) {
    auto opType = op->type;
    if (opType != OpType_Convolution && opType != OpType_ConvolutionDepthwise && opType != OpType_Deconvolution && opType != OpType_DeconvolutionDepthwise && opType != OpType_BatchNorm) {
        return;
    }
    if (op->inputIndexes.size() != 1) {
        return;
    }

    int inputIndex = op->inputIndexes[0];
    int outputIndex = op->outputIndexes[0];
    std::string inputTensorName = subgraph ? subgraph->tensors[inputIndex] : netT->tensorName[inputIndex];
    std::string outputTensorName = subgraph ? subgraph->tensors[outputIndex] : netT->tensorName[outputIndex];

    std::vector<int> inputMask = tensorMaskInfo[inputTensorName].mask;
    int inputMaskSum = 0;
    for (int i = 0; i < inputMask.size(); i++) {
        inputMaskSum += inputMask[i];
    }

    if (opType == OpType_BatchNorm) {
        if (!(inputMaskSum < inputMask.size())) {
            return;
        }
        
        auto bnParams = op->main.AsBatchNorm();
        auto slopFloat = bnParams->slopeData;
        auto biasFloat = bnParams->biasData;
        auto meanFloat = bnParams->meanData;
        auto varianceFloat = bnParams->varData;

        bnParams->slopeData.clear();
        bnParams->biasData.clear();
        bnParams->meanData.clear();
        bnParams->varData.clear();

        for (int i = 0; i < varianceFloat.size(); i++) {
            if (inputMask[i] == 1) {
                bnParams->slopeData.push_back(slopFloat[i]);
                bnParams->biasData.push_back(biasFloat[i]);
                bnParams->meanData.push_back(meanFloat[i]);
                bnParams->varData.push_back(varianceFloat[i]);
            }
        }
        bnParams->channels = inputMaskSum;

        return;
    }

    auto convParams  = op->main.AsConvolution2D();
    auto weightFloat = convParams->weight;
    auto biasFloat   = convParams->bias;
    auto& common     = convParams->common;

    int ko = common->outputCount;
    int ki = common->inputCount / common->group;
    int kh = common->kernelY;
    int kw = common->kernelX;

    std::vector<int> opMask;
    for (auto info : tensorMaskInfo) {
        if (op->name == info.second.oriConvName) {
            opMask = info.second.mask;
            break;
        }
    }

    int opMaskSum = 0;
    for (int i = 0; i < opMask.size(); i++) {
        opMaskSum += opMask[i];
    }

    if (opMaskSum < opMask.size()) {
        convParams->weight.clear();
        convParams->bias.clear();

        for (int i = 0; i < ko; i++) {
            int offset = i * ki * kh * kw;
            if (opMask[i] == 1) {
                for (int j = 0; j < ki * kh * kw; j++) {
                    convParams->weight.emplace_back(weightFloat[offset + j]);
                }
                convParams->bias.emplace_back(biasFloat[i]);
            }
        }
        common->outputCount = opMaskSum;
    }

    if (inputMaskSum < inputMask.size()) {
        auto weightFloat = convParams->weight;
        convParams->weight.clear();

        int ko = common->outputCount;
        int ki = common->inputCount / common->group;
        int kh = common->kernelY;
        int kw = common->kernelX;
        
        for (int i = 0; i < ko; i++) {
            for (int j = 0; j < ki; j++) {
                int offset = i * ki * kh * kw + j * kh * kw;
                if (inputMask[j] == 1) {
                    for (int k = 0; k < kh * kw; k++) {
                        convParams->weight.emplace_back(weightFloat[offset + k]);
                    }
                }
            }
        }

        common->inputCount = inputMaskSum;

        // we will not do prune for depthwise, its channel pruning only depends on its input tensor's pruning
        if (opType == OpType_ConvolutionDepthwise || opType == OpType_DeconvolutionDepthwise) {
            common->outputCount = inputMaskSum;
        }
    }
}

// propagate and analyze prune mask info in model
void analyzePruneInfo(std::unique_ptr<MNN::OpT>& op, std::unique_ptr<MNN::NetT>& netT, SubGraphProtoT* subgraph, std::map<std::string, TensorMaskInfo>& tensorMaskInfo, std::set<std::string>& notSafeConvNames) {
    auto opType = op->type;
    auto inputIndices = op->inputIndexes;
    if (inputIndices.size() == 0) {
        return;
    }
    auto outputIndices = op->outputIndexes;
    std::vector<std::string> inputTensorNames;
    for (int i = 0; i < inputIndices.size(); i++) {
        inputTensorNames.push_back(subgraph ? subgraph->tensors[inputIndices[i]] : netT->tensorName[inputIndices[i]]);
    }
    std::vector<std::string> outputTensorNames;
    for (int i = 0; i < outputIndices.size(); i++) {
        outputTensorNames.push_back(subgraph ? subgraph->tensors[outputIndices[i]] : netT->tensorName[outputIndices[i]]);
    }

    if (opType == OpType_Convolution || opType == OpType_Deconvolution) {
        if (inputIndices.size() == 1) {
            auto convParams  = op->main.AsConvolution2D();
            auto weightFloat = convParams->weight;
            auto biasFloat   = convParams->bias;
            auto& common     = convParams->common;

            const int ko = common->outputCount;
            const int ki = common->inputCount / common->group;
            const int kh = common->kernelY;
            const int kw = common->kernelX;

            MNN::Express::VARP weightVar      = MNN::Express::_Const(weightFloat.data(), {ko, ki, kh, kw}, MNN::Express::NCHW);

            MNN::Express::VARP weightMask = MNN::Express::_Greater(MNN::Express::_ReduceSum(MNN::Express::_Abs(weightVar), {1, 2, 3}), MNN::Express::_Scalar<float>(1e-6));
            MNN::Express::VARP maskSum = MNN::Express::_ReduceSum(weightMask);
            auto maskInfo = weightMask->getInfo();
            auto maskPtr = weightMask->readMap<int>();

            if (maskSum->readMap<int>()[0] == maskInfo->size) {
                return;
            }
            
            // conv has pruned, propagate its mask down
            tensorMaskInfo[outputTensorNames[0]].oriConvName = op->name;
            for (int i = 0; i < maskInfo->size; i++) {
                tensorMaskInfo[outputTensorNames[0]].mask.push_back(maskPtr[i]);
            }
        }

        return;
    }

    std::vector<MNN::OpType>::iterator iter;
    iter = std::find(unSafeOpTypes.begin(), unSafeOpTypes.end(), opType);
    // not safe op and num_outputs > 1 op are not safe
    if ((iter != unSafeOpTypes.end()) || (outputTensorNames.size() > 1)) {
        for (auto name : inputTensorNames) {
            if (!tensorMaskInfo[name].oriConvName.empty()) {
                // so that input tensor mask's oriConv op is not safe
                notSafeConvNames.insert(tensorMaskInfo[name].oriConvName);
            }
        }
        return;
    }

    // when a mask is propagated to the output, its oriConv ops are not safe
    std::vector<MNN::OpT*> userOps = findUserOps(outputIndices[0], netT, subgraph);
    if (userOps.size() == 0) {
        for (auto name : inputTensorNames) {
            if (!tensorMaskInfo[name].oriConvName.empty()) {
                notSafeConvNames.insert(tensorMaskInfo[name].oriConvName);
            }
        }
        return;
    }

    // if the op has more than one input (including const input)
    // we need its input tensor's masks are all from one oriConv op
    if (inputIndices.size() > 1) {
        std::string oriConvName;
        std::string oriTensorName;
        for (auto name : inputTensorNames) {
            if (!tensorMaskInfo[name].oriConvName.empty()) {
                oriConvName = tensorMaskInfo[name].oriConvName;
                oriTensorName = name;
            }
        }
        if (oriConvName.empty()) {
            return;
        }

        // oriConvName is not empty
        bool unsafe = false;
        for (auto name : inputTensorNames) {
            auto tOriName = tensorMaskInfo[name].oriConvName;
            if ((tOriName != oriConvName) && (!tOriName.empty())) {
                unsafe = true;
            }
        }

        // if unsafe, all its input tensor mask's oriConvs are not safe
        if (unsafe) {
            for (auto name : inputTensorNames) {
                auto tOriName = tensorMaskInfo[name].oriConvName;
                if (!tOriName.empty()) {
                    notSafeConvNames.insert(tOriName);
                }
            }
            return;
        }

        // if safe, propagate mask down
        tensorMaskInfo[outputTensorNames[0]].oriConvName = oriConvName;
        tensorMaskInfo[outputTensorNames[0]].mask = tensorMaskInfo[oriTensorName].mask;
        return;
    }

    // for 1 input and 1 output safe op, propagate mask down
    tensorMaskInfo[outputTensorNames[0]].oriConvName = tensorMaskInfo[inputTensorNames[0]].oriConvName;
    tensorMaskInfo[outputTensorNames[0]].mask = tensorMaskInfo[inputTensorNames[0]].mask;
}

void channelPruneConvert(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto) {
    bool filterPruned = false;
    for (const auto& algo : proto.algo()) {
        if (algo.type() == Compression::CompressionAlgo::PRUNE) {
            auto prune_type = algo.prune_params().type();
            auto prune_algo_type = MNN::SparseAlgo(prune_type);
            if (prune_type == Compression::PruneParams_PruneType_FILTER) {
                filterPruned = true;
                break;
            }
        }
    }
    
    if (!filterPruned) {
        return;
    }
 
    std::map<std::string, TensorMaskInfo> netMaskInfo;
    for (auto tensorName : netT->tensorName) {
        netMaskInfo[tensorName] = TensorMaskInfo();
    }

    std::set<std::string> notSafeConvNames;
    for (auto& op : netT->oplists) {
        analyzePruneInfo(op, netT, nullptr, netMaskInfo, notSafeConvNames);
    }

    std::set<std::string>::iterator iter;
    if (!notSafeConvNames.empty()) {
        for (auto& info : netMaskInfo) {
            iter = std::find(notSafeConvNames.begin(), notSafeConvNames.end(), info.second.oriConvName);
            if (iter != notSafeConvNames.end()) {
                for (int i = 0; i < info.second.mask.size(); i++) {
                    if (info.second.mask[i] == 0) {
                        info.second.mask[i] = 1;
                    }
                }
            }
        }
    }

    for (auto& op : netT->oplists) {
        channelPrune(op, netT, nullptr, netMaskInfo);
    }


    for (auto& subgraph : netT->subgraphs) {
        std::map<std::string, TensorMaskInfo> subgraphMaskInfo;
        for (auto tensorName : subgraph->tensors) {
            subgraphMaskInfo[tensorName] = TensorMaskInfo();
        }

        std::set<std::string> notSafeConvNames;
        for (auto& op : subgraph->nodes) {
            analyzePruneInfo(op, netT, subgraph.get(), subgraphMaskInfo, notSafeConvNames);
        }

        std::set<std::string>::iterator iter;
        if (!notSafeConvNames.empty()) {
            for (auto& info : subgraphMaskInfo) {
                iter = std::find(notSafeConvNames.begin(), notSafeConvNames.end(), info.second.oriConvName);
                if (iter != notSafeConvNames.end()) {
                    for (int i = 0; i < info.second.mask.size(); i++) {
                        if (info.second.mask[i] == 0) {
                            info.second.mask[i] = 1;
                        }
                    }
                }
            }
        }

        for (auto& op : subgraph->nodes) {
            channelPrune(op, netT, subgraph.get(), subgraphMaskInfo);
        }
    }
}
