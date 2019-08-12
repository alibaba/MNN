//
//  PostTreatUtils.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PostTreatUtils.hpp"
#include <iostream>
#include <set>
using namespace MNN;
static bool _OpNeedConvertContent(OpType type, int index) {
    switch (type) {
        case OpType_Shape:
        case OpType_PriorBox:
        case OpType_Const:
            return false;
        case OpType_Interp:
        case OpType_Crop:
        case OpType_Reshape:
        case OpType_Resize:
        case OpType_Padding:
            if (1 == index) {
                return false;
            }
            break;
        default:
            break;
    }
    return true;
}

PostTreatUtils::PostTreatUtils(std::unique_ptr<MNN::NetT>& net) : mNet(std::move(net)) {
}

const std::set<MNN::OpType> PostTreatUtils::NC4HW4_OPs = {
    MNN::OpType_Convolution,
    MNN::OpType_ConvolutionDepthwise,
    MNN::OpType_Pooling,
    MNN::OpType_ROIPooling,
    MNN::OpType_Resize,
    MNN::OpType_LSTM,
    MNN::OpType_SpatialProduct,
    MNN::OpType_Deconvolution,
    MNN::OpType_DeconvolutionDepthwise,
    MNN::OpType_Proposal,
    MNN::OpType_PriorBox,
    MNN::OpType_DetectionOutput,
    MNN::OpType_Eltwise,
    MNN::OpType_LRN,
    MNN::OpType_Interp,
    MNN::OpType_Crop,
    MNN::OpType_Scale,
    MNN::OpType_TfQuantizedConv2D,
    MNN::OpType_QuantizedDepthwiseConv2D,
    MNN::OpType_BatchToSpaceND,
    MNN::OpType_SpaceToBatchND,
    MNN::OpType_BatchNorm,
    MNN::OpType_Moments,
    MNN::OpType_QuantizedAvgPool,
    MNN::OpType_QuantizedAdd,
    MNN::OpType_PReLU,
};

const std::set<MNN::OpType> PostTreatUtils::COMPABILITY_OPs = {
    MNN::OpType_ReLU,    MNN::OpType_ReLU6,         MNN::OpType_Concat,  MNN::OpType_Slice,   MNN::OpType_Permute,
    MNN::OpType_Selu,    MNN::OpType_ConvertTensor, MNN::OpType_Sigmoid, MNN::OpType_Cast,
    MNN::OpType_Reshape, MNN::OpType_TanH,          MNN::OpType_ArgMax, MNN::OpType_Padding};

const std::vector<MNN::OpType> PostTreatUtils::DELETE_Ops = {
    MNN::OpType_Seq2Out,
    MNN::OpType_Dropout,
};

void PostTreatUtils::treatIm2Seq() {
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto& op = *iter;
        if (op->type != MNN::OpType_Im2Seq) {
            iter++;
            continue;
        }
        auto inputId    = op->inputIndexes[0];
        auto outputId   = op->outputIndexes[0];
        auto outputname = mNet->tensorName[outputId];

        // New Reshape
        MNN::OpT* reshapeT = new MNN::OpT;
        reshapeT->name     = "____reshape____" + op->name;
        auto reshapeP      = new MNN::ReshapeT;
        reshapeP->dims.push_back(0);  // b
        reshapeP->dims.push_back(-1); // c
        reshapeP->dims.push_back(1);  // h
        reshapeP->dims.push_back(0);  // w
        reshapeT->main.type  = MNN::OpParameter_Reshape;
        reshapeT->type       = MNN::OpType_Reshape;
        reshapeT->main.value = reshapeP;

        // Net Tensor
        mNet->tensorName.push_back(reshapeT->name);
        int tempId = mNet->tensorName.size() - 1;

        reshapeT->inputIndexes.push_back(inputId);
        reshapeT->outputIndexes.push_back(tempId);

        op->inputIndexes[0] = tempId;
        op->type            = MNN::OpType_Permute;

        auto convP     = new MNN::PermuteT;
        op->main.type  = MNN::OpParameter_Permute;
        op->main.value = convP;
        convP->dims.push_back(0);
        convP->dims.push_back(3);
        convP->dims.push_back(2);
        convP->dims.push_back(1);

        iter = mNet->oplists.insert(iter, std::unique_ptr<MNN::OpT>(reshapeT));
    }
}

void PostTreatUtils::deleteUnusefulOp() {
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto& op          = *iter;
        bool shouldDelete = false;
        for (int i = 0; i < PostTreatUtils::DELETE_Ops.size(); ++i) {
            if (op->type == PostTreatUtils::DELETE_Ops[i]) {
                shouldDelete = true;
                break;
            }
        }

        if (!shouldDelete) {
            iter++;
            continue;
        }

        // Find the next op
        auto originInput  = op->inputIndexes[0];
        auto originOutput = op->outputIndexes[0];
        iter              = mNet->oplists.erase(iter);
        for (auto subIter = mNet->oplists.begin(); subIter != mNet->oplists.end(); subIter++) {
            auto& subOp = *subIter;
            for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
                if (subOp->inputIndexes[v] == originOutput) {
                    subOp->inputIndexes[v] = originInput;
                }
            }
        }
    }
}

bool PostTreatUtils::_merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) {
    if (inplaceOp->type == MNN::OpType_ReLU && inplaceOp->main.AsRelu()->slope == 0.0f) {
        convolutionOp->main.AsConvolution2D()->common->relu = true;
        return true;
    }
    if (inplaceOp->type == MNN::OpType_ReLU6) {
        convolutionOp->main.AsConvolution2D()->common->relu6 = true;
        return true;
    }

    const auto& convCommon = convolutionOp->main.AsConvolution2D()->common;
    if (convCommon->relu || convCommon->relu6) {
        return false;
    }

    if (inplaceOp->type == MNN::OpType_BatchNorm || inplaceOp->type == MNN::OpType_Scale) {
        std::vector<float> alpha;
        std::vector<float> bias;
        if (inplaceOp->type == MNN::OpType_BatchNorm) {
            auto l = inplaceOp->main.AsBatchNorm();
            alpha.resize(l->channels);
            bias.resize(l->channels);
            const float* slopePtr    = l->slopeData.data();
            const float* meanDataPtr = l->meanData.data();
            const float* varDataPtr  = l->varData.data();
            const float* biasDataPtr = l->biasData.data();

            for (int i = 0; i < l->channels; i++) {
                float sqrt_var = sqrt(varDataPtr[i]);
                bias[i]        = biasDataPtr[i] - slopePtr[i] * meanDataPtr[i] / sqrt_var;
                alpha[i]       = slopePtr[i] / sqrt_var;
            }
        }
        if (inplaceOp->type == MNN::OpType_Scale) {
            bias  = inplaceOp->main.AsScale()->biasData;
            alpha = inplaceOp->main.AsScale()->scaleData;
        }

        auto conv2D     = convolutionOp->main.AsConvolution2D();
        int outputCount = conv2D->common->outputCount;
        for (int i = 0; i < outputCount; ++i) {
            conv2D->bias[i] = conv2D->bias[i] * alpha[i] + bias[i];
        }

        if (nullptr != conv2D->quanParameter.get()) {
            for (int i = 0; i < outputCount; ++i) {
                conv2D->quanParameter->alpha[i] *= alpha[i];
            }
        } else {
            int weightPartSize = conv2D->weight.size() / outputCount;
            if (convolutionOp->type == OpType_Deconvolution) {
                int inputCount =
                    conv2D->weight.size() / outputCount / conv2D->common->kernelX / conv2D->common->kernelY;
                for (int i = 0; i < inputCount; ++i) {
                    auto dstPos = i * outputCount * conv2D->common->kernelY * conv2D->common->kernelX;
                    for (int j = 0; j < outputCount; ++j) {
                        auto dstPosJ = dstPos + j * conv2D->common->kernelY * conv2D->common->kernelX;
                        float a      = alpha[j];
                        for (int k = 0; k < conv2D->common->kernelY * conv2D->common->kernelX; ++k) {
                            conv2D->weight[dstPosJ + k] *= a;
                        }
                    }
                }
            } else {
                for (int i = 0; i < outputCount; ++i) {
                    float a = alpha[i];
                    for (int j = 0; j < weightPartSize; ++j) {
                        conv2D->weight[i * weightPartSize + j] *= a;
                    }
                }
            }
        }
        return true;
    }

    return false;
}

bool PostTreatUtils::_isSingleInputOutput(const MNN::OpT* op) {
    if (op->inputIndexes.size() != 1 || op->outputIndexes.size() != 1) {
        return false;
    }
    return true;
}

void PostTreatUtils::merge2Convolution() {
    // Merge Layer
    std::vector<MNN::OpT*> readyToDelete;
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
        MNN::OpT& currentOp = *(iter->get());
        if (currentOp.type != MNN::OpType_Convolution && currentOp.type != MNN::OpType_Deconvolution &&
            currentOp.type != MNN::OpType_ConvolutionDepthwise) {
            continue;
        }
        DCHECK(currentOp.outputIndexes.size() == 1) << "Conv output ERROR!";

        // merge Batchnorm/Relu/Relu6 to Convolution
        std::vector<MNN::OpT*> nextOp = this->_findOpByInputIndex(currentOp.outputIndexes[0]);
        while (1) {
            if (nextOp.size() != 1) {
                break;
            }
            const int nextOutputIndex = nextOp[0]->outputIndexes[0];
            bool succ                 = _merge2Convolution(nextOp[0], &currentOp);
            if (_isSingleInputOutput(nextOp[0]) && succ) {
                // LOG(INFO) << "Merge " << nextOp[0]->name.c_str()<< " into convolution: " << currentOp.name.c_str();
                currentOp.outputIndexes[0] = nextOp[0]->outputIndexes[0];
                readyToDelete.push_back(nextOp[0]);
                nextOp = this->_findOpByInputIndex(nextOutputIndex);
            } else {
                break;
            }
        }
    }
    for (auto op : readyToDelete) {
        _removeOpInNet(op);
    }
}

void PostTreatUtils::addTensorType() {
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
        auto& op = *iter;
        if (op->type == MNN::OpType_StridedSlice) {
            auto parameter = op->main.AsStridedSliceParam();
            auto dataType  = parameter->T;

            {
                int index                = op->inputIndexes[0];
                auto describe            = std::unique_ptr<MNN::TensorDescribeT>(new MNN::TensorDescribeT);
                describe->index          = index;
                describe->blob           = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
                describe->blob->dataType = dataType;
                mNet->extraTensorDescribe.push_back(std::move(describe));
            }
            {
                int index                = op->outputIndexes[0];
                auto describe            = std::unique_ptr<MNN::TensorDescribeT>(new MNN::TensorDescribeT);
                describe->index          = index;
                describe->blob           = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
                describe->blob->dataType = dataType;
                mNet->extraTensorDescribe.push_back(std::move(describe));
            }
        }
        if (op->type == MNN::OpType_Const) {
            auto constP = op->main.AsBlob();
            {
                int index                = op->outputIndexes[0];
                auto describe            = std::unique_ptr<MNN::TensorDescribeT>(new MNN::TensorDescribeT);
                describe->index          = index;
                describe->blob           = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
                describe->blob->dataType = constP->dataType;
                mNet->extraTensorDescribe.push_back(std::move(describe));
            }
        }
    }
}
void PostTreatUtils::removeInplaceOp() {
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
        auto& op = *iter;
        if (!_isSingleInputOutput(op.get())) {
            continue;
        }
        if (op->inputIndexes[0] != op->outputIndexes[0]) {
            continue;
        }
        auto originIndex = op->inputIndexes[0];
        mNet->tensorName.push_back(op->name);
        int newIndex         = mNet->tensorName.size() - 1;
        op->outputIndexes[0] = newIndex;
        for (auto subIter = iter + 1; subIter != mNet->oplists.end(); subIter++) {
            auto& subOp = *subIter;
            for (int i = 0; i < subOp->inputIndexes.size(); ++i) {
                if (subOp->inputIndexes[i] == originIndex) {
                    subOp->inputIndexes[i] = newIndex;
                }
            }
            for (int i = 0; i < subOp->outputIndexes.size(); ++i) {
                if (subOp->outputIndexes[i] == originIndex) {
                    subOp->outputIndexes[i] = newIndex;
                }
            }
        }
        mNet->tensorNumber = mNet->tensorName.size();
    }
}
void PostTreatUtils::turnOnnxPadToTensorflow() {
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto op = iter->get();
        if (OpType_Padding == op->type && op->main.type == OpParameter_Blob) {
            std::unique_ptr<OpT> paddingConst(new OpT);
            paddingConst->type          = OpType_Const;
            paddingConst->main.type     = OpParameter_Blob;
            paddingConst->main.value    = new BlobT(*op->main.AsBlob());
            paddingConst->name          = op->name + "padding";
            paddingConst->outputIndexes = {(int)mNet->tensorName.size()};
            mNet->tensorName.emplace_back(paddingConst->name);
            op->inputIndexes = {op->inputIndexes[0], paddingConst->outputIndexes[0]};
            op->main.Reset();
            iter = mNet->oplists.insert(iter, std::move(paddingConst));
            iter++;
            iter++;
            continue;
        }
        iter++;
    }
}
void PostTreatUtils::reIndexTensor() {
    std::map<int, int> usefulTensorIndexMap;
    std::vector<std::string> usefulTensorName;

    std::vector<bool> tensorValid(mNet->tensorName.size(), false);
    for (auto& op : mNet->oplists) {
        for (auto index : op->inputIndexes) {
            tensorValid[index] = true;
        }
        for (auto index : op->outputIndexes) {
            tensorValid[index] = true;
        }
    }

    for (int i = 0; i < tensorValid.size(); ++i) {
        if (tensorValid[i]) {
            usefulTensorIndexMap.insert(std::make_pair(i, usefulTensorName.size()));
            usefulTensorName.push_back(mNet->tensorName[i]);
        }
    }

    // Re index
    for (auto& op : mNet->oplists) {
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            auto iter = usefulTensorIndexMap.find(op->inputIndexes[i]);
            DCHECK(iter != usefulTensorIndexMap.end()) << "ERROR";
            op->inputIndexes[i] = iter->second;
        }
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            auto iter = usefulTensorIndexMap.find(op->outputIndexes[i]);
            DCHECK(iter != usefulTensorIndexMap.end()) << "ERROR";
            op->outputIndexes[i] = iter->second;
        }
    }

    mNet->tensorName = usefulTensorName;
    for (auto iter = mNet->extraTensorDescribe.begin(); iter != mNet->extraTensorDescribe.end();) {
        auto index = (*iter)->index;
        if (usefulTensorIndexMap.find(index) == usefulTensorIndexMap.end()) {
            iter = mNet->extraTensorDescribe.erase(iter);
            continue;
        }
        (*iter)->index = usefulTensorIndexMap.find(index)->second;
        iter++;
    }
}

void PostTreatUtils::addConverterForTensorFlowModel() {
    if(mNet->sourceType == MNN::NetSource_CAFFE){
        return;
    }

    auto originTensorType = MNN::MNN_DATA_FORMAT_NHWC;
    if (mNet->sourceType == MNN::NetSource_ONNX) {
        originTensorType = MNN::MNN_DATA_FORMAT_NCHW;
    }

    // set the layout of every tensor
    // Don't support inplace
    std::map<int, MNN::MNN_DATA_FORMAT> tensorType;
    std::map<std::string, MNN::MNN_DATA_FORMAT> opType;
    for (auto& iter : mNet->oplists) {
        // set output tensor layout of this op according to context
        auto type = originTensorType;
        if (iter->type == MNN::OpType_ConvertTensor) {
            type = iter->main.AsTensorConvertInfo()->dest;
        } else if (PostTreatUtils::NC4HW4_OPs.find(iter->type) != PostTreatUtils::NC4HW4_OPs.end()) {
            type = MNN::MNN_DATA_FORMAT_NC4HW4;
        } else if (PostTreatUtils::COMPABILITY_OPs.find(iter->type) != PostTreatUtils::COMPABILITY_OPs.end()) {
            int nc4hw4TypeNumber = 0;  // NC4HW4 number
            int originTypeNumber = 0;
            for (int i = 0; i < iter->inputIndexes.size(); ++i) {
                auto index = iter->inputIndexes[i];
                if (_OpNeedConvertContent(iter->type, i)) {
                    if (tensorType[index] == MNN::MNN_DATA_FORMAT_NC4HW4) {
                        nc4hw4TypeNumber++;
                    } else if (tensorType[index] == originTensorType) {
                        originTypeNumber++;
                    }
                }
            }
            if (nc4hw4TypeNumber > originTypeNumber) {
                type = MNN::MNN_DATA_FORMAT_NC4HW4;
            }
            if (iter->type == MNN::OpType_Reshape) {
                if (iter->main.AsReshape()->dims.size() != 4) {
                    type = originTensorType;
                }
            }
        }

        for (auto index : iter->outputIndexes) {
            tensorType[index] = type;
        }
        opType.insert(std::make_pair(iter->name, type));
    }
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto op = iter->get();
        // Insert Pretreat Op if needed
        if (opType.find(op->name)->second == MNN::MNN_DATA_FORMAT_NHWC) {
            iter++;
            continue;
        }
        if (op->type == OpType_Padding) {
            // Add Gather op for padding, turn nhwc -> nchw
            std::unique_ptr<OpT> gatherIndex(new OpT);
            gatherIndex->outputIndexes = {(int)mNet->tensorName.size()};
            gatherIndex->type          = OpType_Const;
            gatherIndex->name          = op->name + "_Gather_Index";
            mNet->tensorName.emplace_back(gatherIndex->name);
            gatherIndex->main.type                 = OpParameter_Blob;
            gatherIndex->main.value                = new BlobT;
            gatherIndex->main.AsBlob()->dataType   = DataType_DT_INT32;
            gatherIndex->main.AsBlob()->dataFormat = originTensorType;
            gatherIndex->main.AsBlob()->int32s     = {0, 3, 1, 2};
            gatherIndex->main.AsBlob()->dims       = {4};
            opType.insert(std::make_pair(gatherIndex->name, originTensorType));

            std::unique_ptr<OpT> gather(new OpT);
            gather->outputIndexes = {(int)mNet->tensorName.size()};
            gather->inputIndexes  = {op->inputIndexes[1], gatherIndex->outputIndexes[0]};

            gather->type = OpType_GatherV2;
            gather->name = op->name + "_Gather";
            mNet->tensorName.emplace_back(gather->name);
            opType.insert(std::make_pair(gather->name, originTensorType));

            op->inputIndexes[1]                       = gather->outputIndexes[0];
            tensorType[gather->outputIndexes[0]]      = originTensorType;
            tensorType[gatherIndex->outputIndexes[0]] = originTensorType;

            iter = mNet->oplists.insert(iter, std::move(gather));
            iter = mNet->oplists.insert(iter, std::move(gatherIndex));
            iter++;
            iter++;
            iter++;
        } else {
            iter++;
        }
    }

    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto& op         = *iter;
        auto currentType = opType.find(op->name)->second;
        std::vector<MNN::OpT*> transformOps;
        auto currentName         = op->name;
        const bool useAutoFormat = NC4HW4_OPs.find(op->type) != NC4HW4_OPs.end();

        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            auto inputIndex = op->inputIndexes[i];

            MNN::OpT* inputOp = this->_findOpByOutputIndex(inputIndex);
            if (inputOp && inputOp->type == MNN::OpType_Input && useAutoFormat) {
                auto inputOpParam      = inputOp->main.AsInput();
                inputOpParam->dformat  = MNN::MNN_DATA_FORMAT_NC4HW4;
                tensorType[inputIndex] = MNN::MNN_DATA_FORMAT_NC4HW4;
                opType[inputOp->name]  = MNN::MNN_DATA_FORMAT_NC4HW4;
                continue;
            }

            auto type = tensorType[inputIndex];
            if (type == currentType) {
                continue;
            }

            if (!_OpNeedConvertContent(op->type, i)) {
                continue;
            }

            // Insert Transform op
            MNN::OpT* transformOp = new MNN::OpT;
            transformOps.push_back(transformOp);
            MNN::TensorConvertInfoT* tc = new MNN::TensorConvertInfoT;
            tc->source                  = type;
            tc->dest                    = currentType;
            transformOp->main.type      = MNN::OpParameter_TensorConvertInfo;
            transformOp->main.value     = tc;
            transformOp->name           = mNet->tensorName[inputIndex] + "___tr4" + op->name;
            // printf("Insert convert for %s, %s 's input %d\n", net->tensorName[inputIndex].c_str(), op->name.c_str(),
            // i);
            transformOp->inputIndexes.push_back(inputIndex);
            transformOp->outputIndexes.push_back(mNet->tensorName.size());

            mNet->tensorName.push_back(transformOp->name);
            op->inputIndexes[i] = transformOp->outputIndexes[0];
            transformOp->type   = MNN::OpType_ConvertTensor;
        }
        for (int i = transformOps.size() - 1; i >= 0; i--) {
            iter = mNet->oplists.insert(iter, std::unique_ptr<MNN::OpT>(transformOps[i]));
        }
        for (; (*iter)->name != currentName; iter++) {
        }
        iter++;
    }

    if (mNet->sourceType == MNN::NetSource_ONNX) {
        return;
    }

    // Reset axis map
    const int axisMap[4] = {0, 2, 3, 1};

    for (auto& op : mNet->oplists) {
        if (opType.find(op->name) == opType.end()) {
            continue;
        }
        if (opType.find(op->name)->second == MNN::MNN_DATA_FORMAT_NHWC) {
            continue;
        }
        if (MNN::OpType_Input == op->type) {
            auto input = op->main.AsInput();
            if (4 == input->dims.size()) {
                int h          = input->dims[1];
                int c          = input->dims[3];
                int w          = input->dims[2];
                input->dims[1] = c;
                input->dims[2] = h;
                input->dims[3] = w;
            }
        }
        if (MNN::OpType_Concat == op->type) {
            auto axis       = op->main.AsAxis();
            auto concatAxis = axis->axis;
            if (concatAxis < 0) {
                concatAxis = 4 + concatAxis;
            }
            DCHECK(concatAxis >= 0 && concatAxis <= 3) << "Concat axis ERROR!";
            axis->axis = axisMap[concatAxis];
        }
        if (MNN::OpType_Permute == op->type) {
            auto permuteT = op->main.AsPermute();
            for (int i = 0; i < permuteT->dims.size(); ++i) {
                DCHECK(permuteT->dims[i] >= 0 && permuteT->dims[i] <= 3) << "Dim Error ==> " << op->name;
                permuteT->dims[i] = axisMap[permuteT->dims[i]];
            }
        }
        if (MNN::OpType_Slice == op->type) {
            auto slice      = op->main.AsSlice();
            auto concatAxis = slice->axis;
            if (concatAxis < 0) {
                concatAxis = 4 + concatAxis;
            }
            DCHECK(concatAxis >= 0 && concatAxis <= 3) << "Slice axis ERROR!";
            slice->axis = axisMap[concatAxis];
        }
        if (MNN::OpType_Reshape == op->type) {
            auto reshape   = op->main.AsReshape();
            auto originDim = reshape->dims;
            for (int i = 0; i < reshape->dims.size(); ++i) {
                CHECK(i >= 0 && i <= 3) << "Error";
                reshape->dims[axisMap[i]] = originDim[i];
            }
        }
    }

    std::set<int> tensorTypeSet;
    for (auto& iter : mNet->extraTensorDescribe) {
        auto index             = iter->index;
        iter->blob->dataFormat = tensorType[index];
        tensorTypeSet.insert(index);
    }
    for (auto iter : tensorType) {
        if (tensorTypeSet.find(iter.first) != tensorTypeSet.end()) {
            continue;
        }
        auto describe              = new MNN::TensorDescribeT;
        describe->index            = iter.first;
        describe->blob             = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
        describe->blob->dataFormat = iter.second;
        describe->blob->dataType   = MNN::DataType_DT_FLOAT;

        mNet->extraTensorDescribe.push_back(std::unique_ptr<MNN::TensorDescribeT>(describe));
    }
}

MNN::OpT* ensureOpInNet(std::unique_ptr<MNN::NetT>& net, MNN::OpT* op) {
    for (auto& _op : net->oplists) {
        if (_op.get() == op) {
            return op;
        }
    }
    return nullptr;
}

MNN::OpT* PostTreatUtils::_findOpByOutputIndex(int outputIndex) {
    for (auto& op : mNet->oplists) {
        if (inVector(op->outputIndexes, outputIndex)) {
            return op.get();
        }
    }
    return nullptr;
}

std::vector<MNN::OpT*> PostTreatUtils::_findOpByInputIndex(int inputIndex) {
    std::vector<MNN::OpT*> ops;
    for (auto& op : mNet->oplists) {
        if (inVector(op->inputIndexes, inputIndex)) {
            ops.push_back(op.get());
        }
    }

    // check whether the next op is in_place op
    const int opsSize = ops.size();
    if (opsSize > 1) {
        auto realNextOp = ops[0];
        if (inVector(realNextOp->outputIndexes, inputIndex)) {
            ops.clear();
            ops.push_back(realNextOp);
        }
    }

    return ops;
}

int PostTreatUtils::_getOpDecestorCount(MNN::OpT* op) {
    int decestorCount = 0;
    for (auto& otherOp : mNet->oplists) {
        if (otherOp.get() != op) {
            for (auto inputIndex : otherOp->inputIndexes) {
                if (inVector(op->outputIndexes, inputIndex)) {
                    decestorCount++;
                    break; // one decestor just count one.
                }
            }
        }
    }

    return decestorCount;
}

void PostTreatUtils::_removeOpInNet(MNN::OpT* op) {
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
        if (iter->get() == op) {
            // LOG(INFO) << "remove op: " << op->name;
            mNet->oplists.erase(iter);
            break;
        }
    }
}

void PostTreatUtils::_removeOnlyOneDecestorOps(MNN::OpT* op) {
    std::vector<MNN::OpT*> opsToBeChecked;
    opsToBeChecked.push_back(op);
    while (!opsToBeChecked.empty()) {
        bool hasRemoved = false;
        std::vector<MNN::OpT*> addedToBeChecked;
        for (auto iter = opsToBeChecked.begin(); iter != opsToBeChecked.end();) {
            MNN::OpT* op = *iter;
            if (!ensureOpInNet(mNet, op)) {
                hasRemoved = true;
                iter       = opsToBeChecked.erase(iter);
                continue;
            }
            if (this->_getOpDecestorCount(op) == 0) {
                for (int inputIndex : op->inputIndexes) {
                    addedToBeChecked.push_back(this->_findOpByOutputIndex(inputIndex));
                }
                hasRemoved = true;
                this->_removeOpInNet(op);
                iter = opsToBeChecked.erase(iter);
                continue;
            }
            iter++;
        }
        if (!hasRemoved)
            break;
        opsToBeChecked.insert(opsToBeChecked.end(), addedToBeChecked.begin(), addedToBeChecked.end());
    }
}

void PostTreatUtils::removeDeconvolutionShapeInput() {
    std::set<MNN::OpT*> shapeOps;
    for (auto& op : mNet->oplists) {
        if (op->type == MNN::OpType_Deconvolution || op->type == MNN::OpType_DeconvolutionDepthwise) {
            if (op->inputIndexes.size() == 1) {
                continue;
            }
            int firstInputIndex = op->inputIndexes[0];
            op->inputIndexes.erase(op->inputIndexes.begin());
            MNN::OpT* shapeOp = this->_findOpByOutputIndex(firstInputIndex);
            if (shapeOp) {
                shapeOps.insert(shapeOp);
            }
        }
    }
    for (auto& op : shapeOps) {
        this->_removeOnlyOneDecestorOps(op);
    }
}

void PostTreatUtils::turnInnerProduct2Convolution() {
    std::vector<MNN::OpT*> readyToDelete;
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto& op = *iter;
        if (op->type != MNN::OpType_InnerProduct) {
            iter++;
            continue;
        }
        // ONNX Gemm will be mapped to InnerProduct, check whether is Flatten before Gemm
        // then delete Flatten(mapped to Reshape, and this Reshape will reshape tensor to be
        // two dimensions, such as [M,K], which is the input of Gemm)
        auto inputId       = op->inputIndexes[0];
        auto beforeGemm    = _findOpByOutputIndex(inputId);
        auto refBeforeGemm = _findOpByInputIndex(beforeGemm->outputIndexes[0]);
        if (beforeGemm->type == MNN::OpType_Reshape && _isSingleInputOutput(beforeGemm) && refBeforeGemm.size() == 1) {
            // change the input index
            const int beforeGemmInputId = beforeGemm->inputIndexes[0];

            op->inputIndexes[0] = beforeGemmInputId;
            inputId             = beforeGemmInputId;
            readyToDelete.push_back(beforeGemm);
        }

        auto paramInner = op->main.AsInnerProduct();
        const auto axis = paramInner->axis;

        std::vector<MNN::OpT*> newOpPrevious;
        std::vector<MNN::OpT*> newOpPost;
        // New Reshape
        MNN::OpT* reshapeT = new MNN::OpT;
        newOpPrevious.push_back(reshapeT);
        reshapeT->name = "____reshape____" + op->name;
        auto reshapeP  = new MNN::ReshapeT;
        reshapeP->dims.resize(4);
        for (int i = 0; i < axis; ++i) {
            reshapeP->dims[i] = 0;
        }
        reshapeP->dims[axis] = -1;
        for (int i = axis + 1; i < 4; ++i) {
            reshapeP->dims[i] = 1;
        }
        if (mNet->sourceType == MNN::NetSource_TENSORFLOW) {
            reshapeP->dims[3] = -1;
            reshapeP->dims[1] = 1;
            reshapeP->dims[2] = 1;
        }

        reshapeT->main.type  = MNN::OpParameter_Reshape;
        reshapeT->type       = MNN::OpType_Reshape;
        reshapeT->main.value = reshapeP;

        // Net Tensor
        mNet->tensorName.push_back(reshapeT->name);
        int tempId = mNet->tensorName.size() - 1;

        reshapeT->inputIndexes.push_back(inputId);
        reshapeT->outputIndexes.push_back(tempId);
        auto opName      = op->name;
        bool needPermute = 1 != axis && mNet->sourceType == MNN::NetSource_CAFFE;

        if (needPermute) {
            // Add Permute
            auto permuteBefore       = new MNN::OpT;
            permuteBefore->type      = MNN::OpType_Permute;
            permuteBefore->main.type = MNN::OpParameter_Permute;
            auto permuteT            = new MNN::PermuteT;
            permuteBefore->name      = "___permute1__" + reshapeT->name;
            permuteT->dims.resize(4);
            for (int i = 0; i < 4; ++i) {
                permuteT->dims[i] = i;
            }
            permuteT->dims[1]         = axis;
            permuteT->dims[axis]      = 3;
            permuteT->dims[3]         = 1;
            permuteBefore->main.value = permuteT;
            permuteBefore->inputIndexes.push_back(tempId);
            mNet->tensorName.push_back(permuteBefore->name);
            tempId = mNet->tensorName.size() - 1;
            permuteBefore->outputIndexes.push_back(tempId);

            newOpPrevious.push_back(permuteBefore);
        }

        op->inputIndexes[0] = tempId;
        op->type            = MNN::OpType_Convolution;

        auto convP                 = new MNN::Convolution2DT;
        auto originInner           = op->main.AsInnerProduct();
        convP->common              = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
        convP->common->kernelX     = 1;
        convP->common->kernelY     = 1;
        convP->common->dilateX     = 1;
        convP->common->dilateY     = 1;
        convP->common->strideX     = 1;
        convP->common->strideY     = 1;
        convP->common->group       = 1;
        convP->common->outputCount = originInner->outputCount;
        convP->common->padX        = 0;
        convP->common->padY        = 0;
        convP->common->padMode     = MNN::PadMode_CAFFE;
        convP->bias                = originInner->bias;
        convP->weight              = originInner->weight;
        convP->quanParameter       = std::move(originInner->quanParameter);
        if (convP->quanParameter.get() != nullptr) {
            convP->quanParameter->has_scaleInt = false;
        }
        op->main.Reset();
        op->main.type  = MNN::OpParameter_Convolution2D;
        op->main.value = convP;

        if (needPermute) {
            // Add Permute After
            auto permuteBefore       = new MNN::OpT;
            permuteBefore->type      = MNN::OpType_Permute;
            permuteBefore->main.type = MNN::OpParameter_Permute;
            auto permuteT            = new MNN::PermuteT;
            permuteBefore->name      = "___permute2__" + reshapeT->name;
            permuteT->dims.resize(4);
            permuteT->dims[0]         = 0;
            permuteT->dims[1]         = 3;
            permuteT->dims[2]         = 2;
            permuteT->dims[3]         = 2;
            permuteT->dims[axis]      = 1;
            permuteBefore->main.value = permuteT;
            mNet->tensorName.push_back(permuteBefore->name);
            tempId = mNet->tensorName.size() - 1;
            permuteBefore->inputIndexes.push_back(tempId);
            permuteBefore->outputIndexes.push_back(op->outputIndexes[0]);
            op->outputIndexes[0] = tempId;

            newOpPost.push_back(permuteBefore);
        }

        for (int i = 0; i < newOpPrevious.size(); ++i) {
            iter = mNet->oplists.insert(iter, std::unique_ptr<MNN::OpT>(newOpPrevious[newOpPrevious.size() - i - 1]));
        }

        for (;; iter++) {
            auto& op = *iter;
            if (op->name == opName) {
                break;
            }
        }

        for (int i = 0; i < newOpPost.size(); ++i) {
            iter = mNet->oplists.insert(iter + 1, std::unique_ptr<MNN::OpT>(newOpPost[i]));
        }
    }

    for (auto op : readyToDelete) {
        _removeOpInNet(op);
    }
}

void PostTreatUtils::turnGroupConvolution() {
    // Pick DepthWise one
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
        auto& op           = *iter;
        const auto op_type = op->type;
        auto conv2D        = op->main.AsConvolution2D();
        auto& common       = conv2D->common;
        if (op_type == MNN::OpType_Convolution || op_type == MNN::OpType_Deconvolution) {
            bool turnConv2DW = false;
            // check whether idst quantization model
            if (nullptr != conv2D->quanParameter.get()) {
                auto& quanParam          = conv2D->quanParameter;
                auto quanWeightBuffer    = quanParam->buffer.data();
                const int weightShapeDim = static_cast<int>(quanWeightBuffer[0]);
                if (weightShapeDim == 4) {
                    const auto weightShapePtr = reinterpret_cast<unsigned short*>(quanWeightBuffer + 1);
                    int ci                    = weightShapePtr[1];
                    if (ci == 1 && common->group != 1 && mNet->sourceType == MNN::NetSource_CAFFE) {
                        ci = weightShapePtr[0];
                    }
                    turnConv2DW = common->outputCount == common->group && ci == common->outputCount;
                }
            } else {
                const int srcCount =
                    conv2D->weight.size() * common->group / common->outputCount / common->kernelX / common->kernelY;
                turnConv2DW = common->outputCount == common->group && srcCount == common->outputCount;
            }

            if (turnConv2DW) {
                switch (op_type) {
                    case MNN::OpType_Convolution:
                        op->type = MNN::OpType_ConvolutionDepthwise;
                        break;
                    case MNN::OpType_Deconvolution:
                        op->type = MNN::OpType_DeconvolutionDepthwise;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    // Delete Convolution With Grouop
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto& op = *iter;
        if (op->type != MNN::OpType_Convolution && op->type != MNN::OpType_Deconvolution) {
            iter++;
            continue;
        }
        auto conv2D  = op->main.AsConvolution2D();
        auto& common = conv2D->common;
        if (common->group == 1) {
            iter++;
            continue;
        }

        int srcCount = conv2D->weight.size() * common->group / common->outputCount / common->kernelX / common->kernelY;

        DCHECK(srcCount % common->group == 0 && common->outputCount % common->group == 0)
            << "split group convolution ERROR! ==> " << op->name;

        std::vector<int> newConvolutionInputIndex;
        std::vector<int> newConvolutionOutputIndex;

        for (int i = 0; i < common->group; ++i) {
            std::ostringstream newTensorNameOs;
            newTensorNameOs << op->name << "___input___" << i;
            newConvolutionInputIndex.push_back(mNet->tensorName.size());
            mNet->tensorName.push_back(newTensorNameOs.str());
        }
        for (int i = 0; i < common->group; ++i) {
            std::ostringstream newTensorNameOs;
            newTensorNameOs << op->name << "___output___" << i;
            newConvolutionOutputIndex.push_back(mNet->tensorName.size());
            mNet->tensorName.push_back(newTensorNameOs.str());
        }

        std::vector<MNN::OpT*> newOp;
        // Create slice op
        {
            MNN::OpT* sliceOp      = new MNN::OpT;
            sliceOp->type          = MNN::OpType_Slice;
            sliceOp->name          = op->name + "_____slice";
            sliceOp->inputIndexes  = op->inputIndexes;
            sliceOp->outputIndexes = newConvolutionInputIndex;
            auto sliceT            = new MNN::SliceT;
            sliceOp->main.type     = MNN::OpParameter_Slice;
            sliceOp->main.value    = sliceT;
            sliceT->axis           = 1;
            for (int i = 0; i < common->group - 1; ++i) {
                sliceT->slicePoints.push_back(srcCount / (common->group) * (i + 1));
            }
            newOp.push_back(sliceOp);
        }

        int partWeightSize = conv2D->weight.size() / common->group;
        int partBiasSize   = conv2D->bias.size() / common->group;

        // Create Sub Convolution
        for (int i = 0; i < common->group; ++i) {
            std::ostringstream opNameOs;
            auto newConvOp = new MNN::OpT;
            opNameOs << op->name << "__group__" << i;
            newConvOp->type      = op->type;
            newConvOp->name      = opNameOs.str();
            newConvOp->main.type = MNN::OpParameter_Convolution2D;
            newConvOp->inputIndexes.push_back(newConvolutionInputIndex[i]);
            newConvOp->outputIndexes.push_back(newConvolutionOutputIndex[i]);

            auto newConvolutionT    = new MNN::Convolution2DT;
            newConvOp->main.value   = newConvolutionT;
            newConvolutionT->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
            newConvolutionT->common->kernelX     = common->kernelX;
            newConvolutionT->common->kernelY     = common->kernelY;
            newConvolutionT->common->dilateY     = common->dilateY;
            newConvolutionT->common->dilateX     = common->dilateX;
            newConvolutionT->common->strideX     = common->strideX;
            newConvolutionT->common->strideY     = common->strideY;
            newConvolutionT->common->group       = 1;
            newConvolutionT->common->padMode     = common->padMode;
            newConvolutionT->common->outputCount = common->outputCount / common->group;
            newConvolutionT->common->padX        = common->padX;
            newConvolutionT->common->padY        = common->padY;
            newConvolutionT->common->relu        = common->relu;

            int startWeight = partWeightSize * i;
            int startBias   = partBiasSize * i;
            for (int v = 0; v < partWeightSize; ++v) {
                newConvolutionT->weight.push_back(conv2D->weight[startWeight + v]);
            }
            for (int v = 0; v < partBiasSize; ++v) {
                newConvolutionT->bias.push_back(conv2D->bias[startBias + v]);
            }
            newOp.push_back(newConvOp);
        }

        // Set this op be Concat Op
        {
            op->type         = MNN::OpType_Concat;
            op->inputIndexes = newConvolutionOutputIndex;
            op->main.Reset();
            op->main.type = MNN::OpParameter_Axis;

            auto axisT     = new MNN::AxisT;
            axisT->axis    = 1;
            op->main.value = axisT;
        }

        for (int v = 0; v < newOp.size(); ++v) {
            int index = newOp.size() - v - 1;
            iter      = mNet->oplists.insert(iter, std::unique_ptr<MNN::OpT>(newOp[index]));
        }
    }
}

void PostTreatUtils::changeBatchnNorm2Scale() {
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto& op                 = *iter;
        const MNN::OpType opType = op->type;

        if (MNN::OpType_BatchNorm != opType) {
            iter++;
            continue;
        }
        const int inputSize = op->inputIndexes.size();
        DCHECK(inputSize == 1 || inputSize == 3) << "MNN BatchnNorm input size error!";
        // instance norm have three input tensors(input_tensor, mean, variance)
        if (inputSize == 3) {
            iter++;
            continue;
        }
        // DLOG(INFO) << "change BatchNorm to Scale: " << op->name;
        auto batchnormParam  = op->main.AsBatchNorm();
        auto scaleParam      = new MNN::ScaleT;
        scaleParam->channels = batchnormParam->channels;
        scaleParam->scaleData.resize(batchnormParam->channels);
        scaleParam->biasData.resize(batchnormParam->channels);
        const float* slopePtr    = batchnormParam->slopeData.data();
        const float* meanDataPtr = batchnormParam->meanData.data();
        const float* varDataPtr  = batchnormParam->varData.data();
        const float* biasDataPtr = batchnormParam->biasData.data();

        for (int i = 0; i < batchnormParam->channels; i++) {
            float sqrt_var           = sqrt(varDataPtr[i]);
            scaleParam->biasData[i]  = biasDataPtr[i] - slopePtr[i] * meanDataPtr[i] / sqrt_var;
            scaleParam->scaleData[i] = slopePtr[i] / sqrt_var;
        }

        op->type       = MNN::OpType_Scale;
        op->main.type  = MNN::OpParameter_Scale;
        op->main.value = scaleParam;
    }
}
void PostTreatUtils::pluginConvert() {
    for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
        auto op = iter->get();
        if (op->type == OpType_PLUGIN) {
            auto plugin = op->main.AsPlugin();
            if (plugin->type == "ShuffleChannel") {
                int currentTensorCount = (int)mNet->tensorName.size();
                std::unique_ptr<OpT> convertTo(new OpT);
                convertTo->type                               = OpType_ConvertTensor;
                convertTo->main.type                          = OpParameter_TensorConvertInfo;
                convertTo->main.value                         = new TensorConvertInfoT;
                convertTo->main.AsTensorConvertInfo()->source = MNN_DATA_FORMAT_NC4HW4;
                convertTo->main.AsTensorConvertInfo()->dest   = MNN_DATA_FORMAT_NHWC;
                convertTo->inputIndexes                       = op->inputIndexes;
                convertTo->name                               = op->name + "_ConvertToNHWC";
                convertTo->outputIndexes                      = {currentTensorCount + 0};
                mNet->tensorName.emplace_back(convertTo->name);

                auto group = plugin->buffer[0]->int32s[0];
                std::unique_ptr<OpT> reshape(new OpT);
                reshape->type                      = OpType_Reshape;
                reshape->name                      = op->name + "_Reshape";
                reshape->main.value                = new ReshapeT;
                reshape->main.type                 = OpParameter_Reshape;
                reshape->main.AsReshape()->dimType = MNN_DATA_FORMAT_NHWC;
                reshape->main.AsReshape()->dims    = {0, 0, 0, group, -1};
                reshape->inputIndexes              = {currentTensorCount + 0};
                reshape->outputIndexes             = {currentTensorCount + 1};
                mNet->tensorName.emplace_back(reshape->name);

                std::unique_ptr<OpT> constOp(new OpT);
                constOp->type          = OpType_Const;
                auto blob              = new BlobT;
                blob->int32s           = {0, 1, 2, 4, 3};
                blob->dataFormat       = MNN_DATA_FORMAT_NHWC;
                blob->dataType         = DataType_DT_INT32;
                blob->dims             = {5};
                constOp->main.value    = blob;
                constOp->main.type     = OpParameter_Blob;
                constOp->name          = op->name + "_Const";
                constOp->outputIndexes = {currentTensorCount + 2};
                mNet->tensorName.emplace_back(constOp->name);

                std::unique_ptr<OpT> permute(new OpT);
                permute->type                      = OpType_Transpose;
                permute->name                      = op->name + "_Transpose";
                permute->main.value                = new TransposeT;
                permute->main.type                 = OpParameter_Transpose;
                permute->main.AsTranspose()->Tperm = DataType_DT_INT32;
                permute->inputIndexes              = {currentTensorCount + 1, currentTensorCount + 2};
                permute->outputIndexes             = {currentTensorCount + 3};
                mNet->tensorName.emplace_back(permute->name);

                std::unique_ptr<OpT> reshapeR(new OpT);
                reshapeR->type                      = OpType_Reshape;
                reshapeR->name                      = op->name + "_ReshapeR";
                reshapeR->main.value                = new ReshapeT;
                reshapeR->main.type                 = OpParameter_Reshape;
                reshapeR->main.AsReshape()->dimType = MNN_DATA_FORMAT_NHWC;
                reshapeR->main.AsReshape()->dims    = {0, 0, 0, -1};
                reshapeR->inputIndexes              = {currentTensorCount + 3};
                reshapeR->outputIndexes             = {currentTensorCount + 4};
                mNet->tensorName.emplace_back(reshapeR->name);

                std::unique_ptr<OpT> convertFrom(new OpT);
                convertFrom->type                               = OpType_ConvertTensor;
                convertFrom->main.type                          = OpParameter_TensorConvertInfo;
                convertFrom->main.value                         = new TensorConvertInfoT;
                convertFrom->main.AsTensorConvertInfo()->source = MNN_DATA_FORMAT_NHWC;
                convertFrom->main.AsTensorConvertInfo()->dest   = MNN_DATA_FORMAT_NC4HW4;
                convertFrom->inputIndexes                       = {currentTensorCount + 4};
                convertFrom->outputIndexes                      = op->outputIndexes;
                convertFrom->name                               = op->name + "_ConvertToNC4HW4";

                iter = mNet->oplists.erase(iter);
                iter = mNet->oplists.insert(iter, std::move(convertFrom));
                iter = mNet->oplists.insert(iter, std::move(reshapeR));
                iter = mNet->oplists.insert(iter, std::move(permute));
                iter = mNet->oplists.insert(iter, std::move(constOp));
                iter = mNet->oplists.insert(iter, std::move(reshape));
                iter = mNet->oplists.insert(iter, std::move(convertTo));
                iter++;
                iter++;
                iter++;
                iter++;
                continue;
            }
        }
        iter++;
    }
}
