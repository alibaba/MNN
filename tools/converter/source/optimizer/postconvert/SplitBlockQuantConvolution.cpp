//
//  SplitBlockQuantConvolution.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
#include "config.hpp"
#include "../Global.hpp"
#include "core/FileLoader.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/IDSTEncoder.hpp"
#include "../../common/CommonUtils.hpp"
using namespace MNN;

class SplitBlockQuantConvolution : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& mNet = net;
        auto config = Global<modelConfig>::Get();
        FileLoader originWeight((config->modelFile + ".weight").c_str());
        std::ofstream dstWeight((config->MNNModel + ".weight").c_str());
        int64_t currentDstOffset = 0;

        for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
            auto op = iter->get();
            if (nullptr == op) {
                iter++;
                continue;
            }
            bool split = false;
            do {
                if (op->main.type != OpParameter_Convolution2D || op->type != OpType_Convolution) {
                    break;
                }
                auto conv2D = op->main.AsConvolution2D();
                if (conv2D->quanParameter == nullptr || conv2D->quanParameter->type != 1) {
                    break;
                }
                if (!conv2D->external.empty()) {
                    op->externalPath = config->modelFile + ".weight";
                }
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(Op::Pack(builder, op));
                auto rawOp = flatbuffers::GetRoot<Op>(builder.GetBufferPointer());
                auto quanInfo = ConvolutionCommon::load(rawOp, nullptr, false, true);
                op->externalPath.clear();
                originWeight.offset(conv2D->external[0] + conv2D->external[1] + conv2D->external[2]);
                std::vector<float> bias(conv2D->common->outputCount);
                originWeight.read((char*)bias.data(), conv2D->external[3]);

                // Async is 2
                int divideNumber = quanInfo->asymmetric ? 2 : 1;
                auto alphaCount = quanInfo->alpha.size() / divideNumber;
                auto oc = conv2D->common->outputCount;
                auto groupCount = alphaCount / oc;
                if (groupCount <= 1) {
                    break;
                }
                auto blockSize = conv2D->common->inputCount * conv2D->common->kernelX * conv2D->common->kernelY / groupCount;
                // For 4bit, revert to 8bit
                if (quanInfo->canUseInt4) {
                    auto idxBufSize = quanInfo->weight.size();
                    auto blob = (int8_t*)MNNMemoryAllocAlign(idxBufSize * 2, 32);
                    auto idxBuf = (unsigned char*)quanInfo->weight.get();
                    for (int i = 0; i < idxBufSize; i++) {
                        int val = idxBuf[i];
                        int x1 = val / 16;
                        int x2 = val % 16;
                        blob[2 * i] = x1 - 8;
                        blob[2 * i + 1] = x2 - 8;
                    }
                    quanInfo->weight.set(blob, idxBufSize * 2);
                    quanInfo->canUseInt4 = false;
                }
                if (false) {
                    std::vector<float> subalpha(quanInfo->alpha.size());
                    ::memcpy(subalpha.data(), quanInfo->alpha.get(), quanInfo->alpha.size() * sizeof(float));
                    conv2D->quanParameter = IDSTEncoder::encode(nullptr, subalpha, blockSize * groupCount, oc, quanInfo->asymmetric, quanInfo->weight.get(), conv2D->quanParameter->aMin, quanInfo->originBits);
                    conv2D->external.clear();
                    conv2D->bias = bias;
                    RemoveAndStoreParam(*iter, &dstWeight, currentDstOffset);
                    split = true;
                    iter++;
                    break;
                }

                // Split Convolution
                std::vector<std::unique_ptr<OpT>> subConvolutions(groupCount);
                auto originOutputName = net->tensorName[op->outputIndexes[0]];
                auto originOutputIndex = op->outputIndexes[0];
                auto originInputIndex = op->inputIndexes[0];
                auto originOpName = op->name;
                for (int i=0; i<groupCount; ++i) {
                    subConvolutions[i].reset(rawOp->UnPack());
                    auto subOp = subConvolutions[i].get();
                    subOp->externalPath.clear();
                    subOp->main.AsConvolution2D()->external.clear();
                    subOp->name = op->name + "_" + std::to_string(i);
                    subOp->outputIndexes[0] = (int)net->tensorName.size();
                    net->tensorName.emplace_back(originOutputName + "_" + std::to_string(i));
                    subOp->main.AsConvolution2D()->common->inputCount = conv2D->common->inputCount / groupCount;
                    std::vector<int8_t> subdata(blockSize * oc);
                    std::vector<float> subalpha(oc * divideNumber);
                    for (int y=0; y<oc; ++y) {
                        auto src = quanInfo->weight.get() + y * blockSize * groupCount + i * blockSize;
                        auto dst = subdata.data() + blockSize * y;
                        ::memcpy(dst, src, blockSize);
                        ::memcpy(subalpha.data() + y * divideNumber, quanInfo->alpha.get() + y * divideNumber * groupCount + i * divideNumber, divideNumber * sizeof(float));
                    }
                    subOp->main.AsConvolution2D()->quanParameter = IDSTEncoder::encode(nullptr, subalpha, blockSize, oc, quanInfo->asymmetric, subdata.data(), conv2D->quanParameter->aMin, quanInfo->originBits);
                    if (0 == i) {
                        subOp->main.AsConvolution2D()->bias = bias;
                    } else {
                        subOp->main.AsConvolution2D()->bias = std::vector<float>(oc, 0);
                    }
                    RemoveAndStoreParam(subConvolutions[i], &dstWeight, currentDstOffset);
                }
                {
                    // Add slice
                    std::unique_ptr<OpT> slice(new OpT);
                    slice->type = OpType_Slice;
                    slice->name = op->name + "_inputslice";
                    slice->main.type = OpParameter_Slice;
                    slice->main.value = new SliceT;
                    slice->main.AsSlice()->axis = 1;
                    slice->main.AsSlice()->sourceType = NetSource_TORCH;
                    slice->inputIndexes = {originInputIndex};
                    for (int i=0; i<groupCount; ++i) {
                        subConvolutions[i]->inputIndexes[0] = (int)net->tensorName.size();
                        net->tensorName.emplace_back(net->tensorName[subConvolutions[i]->outputIndexes[0]] + "_input");
                        slice->outputIndexes.emplace_back(subConvolutions[i]->inputIndexes[0]);
                    }
                    iter = net->oplists.insert(iter, std::move(slice));
                    iter++;
                }
                *iter = std::move(subConvolutions[0]);
                auto lastIndex = iter->get()->outputIndexes[0];
                for (int i=1; i<groupCount; ++i) {
                    std::unique_ptr<OpT> add(new OpT);
                    add->type = OpType_BinaryOp;
                    add->main.type = OpParameter_BinaryOp;
                    add->main.value = new BinaryOpT;
                    add->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
                    add->inputIndexes = {lastIndex, subConvolutions[i]->outputIndexes[0]};
                    if (i == groupCount - 1) {
                        add->outputIndexes = {originOutputIndex};
                        add->name = originOpName;
                    } else {
                        add->name = net->tensorName[subConvolutions[i]->outputIndexes[0]] + "_add";
                        add->outputIndexes = {(int)net->tensorName.size()};
                        net->tensorName.emplace_back(net->tensorName[subConvolutions[i]->outputIndexes[0]] + "_add");
                        lastIndex = add->outputIndexes[0];
                    }
                    iter = net->oplists.insert(iter + 1, std::move(subConvolutions[i]));
                    iter = net->oplists.insert(iter + 1, std::move(add));
                }
                iter++;
                split = true;
            } while (false);
            if (split) {
                continue;
            }
            // Copy External
            auto paramType = op->main.type;
            std::vector<int64_t>* external = nullptr;
            switch (paramType) {
                case MNN::OpParameter_Convolution2D:
                    external = &op->main.AsConvolution2D()->external;
                    break;
                case MNN::OpParameter_Scale:
                    external = &op->main.AsScale()->external;
                    break;
                case MNN::OpParameter_LayerNorm:
                    external = &op->main.AsLayerNorm()->external;
                    break;
                case MNN::OpParameter_Blob:
                    external = &op->main.AsBlob()->external;
                    break;
                default:
                    break;
            }
            if (nullptr == external || external->empty()) {
                iter++;
                continue;
            }
            size_t sizeSum = 0;
            for (int j=1; j<external->size(); ++j) {
                sizeSum += external->data()[j];
            }
            originWeight.offset(external->data()[0]);
            std::vector<char> data(sizeSum);
            originWeight.read(data.data(), sizeSum);
            dstWeight.write(data.data(), sizeSum);
            external->data()[0] = currentDstOffset;
            currentDstOffset += sizeSum;
            iter++;
        }
        return true;
    }
};
static PostConverterRegister<SplitBlockQuantConvolution> __l("SplitBlockQuantConvolution");
