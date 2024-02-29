//
//  AddTensorFormatConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include "../Global.hpp"
#include "../SubGraphComplete.hpp"
#include "config.hpp"

using namespace MNN;
static void _setInputFormat(std::vector<MNN_DATA_FORMAT>& tensorFormat, int index, MNN_DATA_FORMAT newFormat) {
    if (tensorFormat[index] == MNN_DATA_FORMAT_UNKNOWN) {
        tensorFormat[index] = newFormat;
    }
}
enum FormatSetType {
    NC4HW4_SINGLE, // only first input / output is nc4hw4
    NC4HW4_FULL, // all nc4hw4
    COMPABILIT_SINGLE, // only first input / output is compability
    COMPABILIT_FULL, // all format should be same
    ORIGIN
};
static FormatSetType _getFormatType(const OpT* op, MNN_DATA_FORMAT originFormat) {
    switch (op->type) {
        // NC4HW4 Ops with multi-input
        case MNN::OpType_SeqLen2Spatial:
	case MNN::OpType_GroupNorm:
	case MNN::OpType_Convolution:
        case MNN::OpType_Convolution3D:
        case MNN::OpType_ConvolutionDepthwise:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_DeconvolutionDepthwise:
        case MNN::OpType_GridSample:
        case MNN::OpType_PReLU:
        case MNN::OpType_Dilation2D:
            return NC4HW4_SINGLE;
        case MNN::OpType_ConvInt8:
        case MNN::OpType_Pooling:
        case MNN::OpType_Pooling3D:
        case MNN::OpType_ROIPooling:
        case MNN::OpType_ROIAlign:
        case MNN::OpType_Resize:
        case MNN::OpType_SpatialProduct:
        case MNN::OpType_Proposal:
        case MNN::OpType_PriorBox:
        case MNN::OpType_DetectionOutput:
        case MNN::OpType_LRN:
        case MNN::OpType_Interp:
        case MNN::OpType_Crop:
        case MNN::OpType_Scale:
        case MNN::OpType_TfQuantizedConv2D:
        case MNN::OpType_QuantizedDepthwiseConv2D:
        case MNN::OpType_BatchNorm:
        case MNN::OpType_InstanceNorm:
        case MNN::OpType_Moments:
        case MNN::OpType_QuantizedAvgPool:
        case MNN::OpType_QuantizedAdd:
        case MNN::OpType_Int8ToFloat:
        case MNN::OpType_FloatToInt8:
        case MNN::OpType_DepthwiseConvInt8:
        case MNN::OpType_Interp3D:
            return NC4HW4_SINGLE;
        case MNN::OpType_ReLU:
        case MNN::OpType_ReLU6:
        case MNN::OpType_Permute:
        case MNN::OpType_Selu:
        case MNN::OpType_Sigmoid:
        case MNN::OpType_Cast:
        case MNN::OpType_BatchToSpaceND:
        case MNN::OpType_SpaceToBatchND:
        case MNN::OpType_TanH:
        case MNN::OpType_Padding:
        case MNN::OpType_ELU:
        case MNN::OpType_Dropout:
        case MNN::OpType_UnaryOp:
        case MNN::OpType_DepthToSpace:
        case MNN::OpType_SpaceToDepth:
            return COMPABILIT_SINGLE;
        case MNN::OpType_Reshape:
        {
            if (op->main.type == OpParameter_Reshape && op->main.AsReshape()->dims.size() == 4) {
                return COMPABILIT_SINGLE;
            }
            break;
        }
        case MNN::OpType_Slice:
        case MNN::OpType_Concat:
        case MNN::OpType_Eltwise:
            return COMPABILIT_FULL;
        default:
            break;
    }
    if (MNN_DATA_FORMAT_NCHW == originFormat) {
        switch (op->type) {
            case MNN::OpType_Transpose:
            case MNN::OpType_StridedSlice:
            case MNN::OpType_SliceTf:
            case MNN::OpType_Unsqueeze:
            case MNN::OpType_Squeeze:
            case MNN::OpType_Crop:
            case MNN::OpType_Tile:
            case MNN::OpType_Reshape:
            case MNN::OpType_Fill:
            case MNN::OpType_BroadcastTo:
            case MNN::OpType_Padding:
            case MNN::OpType_Flatten:
            case MNN::OpType_ExpandDims:
            case MNN::OpType_ReverseSequence:
                return COMPABILIT_SINGLE;
            case MNN::OpType_Pack:
            case MNN::OpType_Unpack:
            case MNN::OpType_BinaryOp:
                return COMPABILIT_FULL;
            default:
                break;
        }
    }
    return ORIGIN;
}
static MNN_DATA_FORMAT _getRequireFormat(FormatSetType type, int inputIndex, MNN_DATA_FORMAT outputFormat, MNN_DATA_FORMAT originFormat) {
    switch (type) {
        case COMPABILIT_FULL:
            return outputFormat;
        case COMPABILIT_SINGLE:
            if (inputIndex == 0) {
                return outputFormat;
            } else {
                return originFormat;
            }
            break;
        case ORIGIN:
            return originFormat;
        case NC4HW4_FULL:
            return MNN_DATA_FORMAT_NC4HW4;
        case NC4HW4_SINGLE:
            if (inputIndex == 0) {
                return MNN_DATA_FORMAT_NC4HW4;
            } else {
                return originFormat;
            }
            break;
        default:
            break;
    }
    MNN_ASSERT(false);
    return MNN_DATA_FORMAT_UNKNOWN;
}

static bool _computeTensorFormat(std::vector<MNN_DATA_FORMAT>& tensorFormat, std::vector<int32_t>& constTensorIndexs, const OpT* op, MNN_DATA_FORMAT originFormat, bool keepInput, bool lastChange) {
    if (op->type == OpType_Input) {
        if (keepInput) {
            tensorFormat[op->outputIndexes[0]] = originFormat;
        }
        // Always return true, don't treat input op
        return true;
    }
    if (op->type == OpType_Const) {
        tensorFormat[op->outputIndexes[0]] = op->main.AsBlob()->dataFormat;
        constTensorIndexs.emplace_back(op->outputIndexes[0]);
        return true;
    }
    if (op->type == OpType_BinaryOp) {
        // Change Binary const input format to nonconst input format
        auto binaryFormat = originFormat;
        for (auto index : op->inputIndexes) {
            auto result = find(constTensorIndexs.begin(), constTensorIndexs.end(), index);
            if (result == constTensorIndexs.end()) {
                binaryFormat = tensorFormat[index];
                break;
            }
        }
        for (auto index : op->inputIndexes) {
            auto result = find(constTensorIndexs.begin(), constTensorIndexs.end(), index);
            if (result != constTensorIndexs.end()) {
                tensorFormat[index] = binaryFormat;
            }
        }
    }    
    if (op->type == OpType_TrainableParam) {
        tensorFormat[op->outputIndexes[0]] = op->main.AsBlob()->dataFormat;
        return true;
    }
    // For the net has been insert convert tensor, use origin format
    if (op->type == OpType_ConvertTensor) {
        tensorFormat[op->outputIndexes[0]] = op->main.AsTensorConvertInfo()->dest;
        return true;
    }
    auto formatType = _getFormatType(op, originFormat);
    if (lastChange) {
        formatType = ORIGIN;
    }
    switch (formatType) {
        // NC4HW4 Ops with multi-input
        case NC4HW4_SINGLE:
        {
            _setInputFormat(tensorFormat, op->inputIndexes[0], MNN_DATA_FORMAT_NC4HW4);
            tensorFormat[op->outputIndexes[0]] = MNN_DATA_FORMAT_NC4HW4;
            for (int i=1; i<op->inputIndexes.size(); ++i) {
                _setInputFormat(tensorFormat, op->inputIndexes[i], originFormat);
            }
            return true;
        }
        case NC4HW4_FULL:
        {
            for (int i=0; i<op->inputIndexes.size(); ++i) {
                _setInputFormat(tensorFormat, op->inputIndexes[i], MNN_DATA_FORMAT_NC4HW4);
            }
            for (int i=0; i<op->outputIndexes.size(); ++i) {
                tensorFormat[op->outputIndexes[i]] = MNN_DATA_FORMAT_NC4HW4;
            }
            return true;
        }
        case COMPABILIT_SINGLE:
        {
            for (int i=1; i<op->inputIndexes.size(); ++i) {
                _setInputFormat(tensorFormat, op->inputIndexes[i], originFormat);
            }
            if (MNN_DATA_FORMAT_UNKNOWN != tensorFormat[op->inputIndexes[0]]) {
                for (auto index : op->outputIndexes) {
                    tensorFormat[index] = tensorFormat[op->inputIndexes[0]];
                }
                return true;
            }
            if (MNN_DATA_FORMAT_UNKNOWN != tensorFormat[op->outputIndexes[0]]) {
                _setInputFormat(tensorFormat, op->inputIndexes[0], tensorFormat[op->outputIndexes[0]]);
                return true;
            }
            return false;
        }
        case COMPABILIT_FULL:
        {
            bool inputValid = true;
            for (auto index : op->inputIndexes) {
                if (tensorFormat[index] == MNN_DATA_FORMAT_UNKNOWN) {
                    inputValid = false;
                    break;
                }
            }
            bool outputValid = true;
            for (auto index : op->outputIndexes) {
                if (tensorFormat[index] == MNN_DATA_FORMAT_UNKNOWN) {
                    outputValid = false;
                    break;
                }
            }
            if (((!inputValid) && (!outputValid))) {
                return false;
            }
            int originNumber = 0;
            int c4Number = 0;
            auto format = originFormat;
            if (inputValid) {
                // Find best format
                for (auto index : op->inputIndexes) {
                    if (tensorFormat[index] == originFormat) {
                        originNumber++;
                    } else {
                        c4Number++;
                    }
                }
            }
	    if (outputValid) {
                // Find best format
                for (auto index : op->outputIndexes) {
                    if (tensorFormat[index] == originFormat) {
                        originNumber++;
                    } else {
                        c4Number++;
                    }
                }
            }
            if (c4Number > originNumber) {
                format = MNN_DATA_FORMAT_NC4HW4;
            }
            for (auto index : op->outputIndexes) {
                tensorFormat[index] = format;
            }
            for (auto index : op->inputIndexes) {
                _setInputFormat(tensorFormat, index, format);
            }
            return true;
        }
        case ORIGIN:
        {
            // Default Set originFormat
            for (int i=0; i<op->inputIndexes.size(); ++i) {
                _setInputFormat(tensorFormat, op->inputIndexes[i], originFormat);
            }
            for (int i=0; i<op->outputIndexes.size(); ++i) {
                tensorFormat[op->outputIndexes[i]] = originFormat;
            }
            return true;
        }
        default:
            break;
    }
    return true;
}

static bool _OpNeedConvertContent(OpType type) {
    switch (type) {
        case OpType_Shape:
        case OpType_PriorBox:
        case OpType_Const:
        case OpType_Rank:
        case OpType_ConvertTensor:
            return false;
        default:
            break;
    }
    return true;
}
class AddTensorFormatConverter : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& mNet = net;
        if (mNet->sourceType == MNN::NetSource_CAFFE) {
            return true;
        }
        auto* ctx = Global<MNN::Express::OptimizeContext>::Get();

        auto originTensorType = MNN::MNN_DATA_FORMAT_NHWC;
        if (mNet->sourceType == MNN::NetSource_ONNX || mNet->sourceType == MNN::NetSource_TORCH) {
            originTensorType = MNN::MNN_DATA_FORMAT_NCHW;
        }
        for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
            auto op = iter->get();
            if (OpParameter_Blob == op->main.type) {
                if (op->main.AsBlob()->dataFormat != MNN_DATA_FORMAT_NC4HW4) {
                    op->main.AsBlob()->dataFormat = originTensorType;
                }
            }
        }

        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        // Compute All Tensor's format
        std::vector<MNN_DATA_FORMAT> tensorFormats(net->tensorName.size());
        std::vector<bool> readyMask(net->oplists.size());
        std::fill(tensorFormats.begin(), tensorFormats.end(), MNN_DATA_FORMAT_UNKNOWN);
        std::fill(readyMask.begin(), readyMask.end(), false);
        bool hasChange = false;
        bool complete = false;
        // Record Const Op Index
        std::vector<int32_t> constTensorIndexs;
        do {
            complete = true;
            hasChange = false;
            for (int i=0; i<readyMask.size(); ++i) {
                if (readyMask[i]) {
                    continue;
                }
                auto op = net->oplists[i].get();
                readyMask[i] = _computeTensorFormat(tensorFormats, constTensorIndexs, op, originTensorType, config->keepInputFormat, false);
                if (readyMask[i]) {
                    hasChange = true;
                } else {
                    complete = false;
                }
            }
        } while (hasChange);

        // Has can't determine one, force compability op use originFormat
        if (!complete) {
            for (int i=0; i<readyMask.size(); ++i) {
                if (readyMask[i]) {
                    continue;
                }
                auto op = net->oplists[i].get();
                readyMask[i] = _computeTensorFormat(tensorFormats, constTensorIndexs, op, originTensorType, config->keepInputFormat, true);
                MNN_ASSERT(readyMask[i] == true);
            }
        }
        // Insert Extra Converter
        std::map<int, int> convertMap;
        if (config->keepInputFormat) {
            // Change Output
            auto& outputs = mNet->outputName;
            std::vector<std::unique_ptr<MNN::OpT>> extraOp;
            for (auto& op : mNet->oplists) {
                for (int idx : op->outputIndexes) {
                    for (int j = 0; j < outputs.size(); j++) {
                        if (mNet->tensorName[idx] == outputs[j]) {
                            auto outputFormat = tensorFormats[idx];
                            if (outputFormat == MNN_DATA_FORMAT_NC4HW4) {
                                auto newOutputName = outputs[j] + "__before_tr";
                                mNet->tensorName[idx] = newOutputName;
                                // Append a convert op
                                MNN::OpT* transformOp = new MNN::OpT;
                                MNN::TensorConvertInfoT* tc = new MNN::TensorConvertInfoT;
                                tc->source                  = outputFormat;
                                tc->dest                    = originTensorType;
                                transformOp->main.type      = MNN::OpParameter_TensorConvertInfo;
                                transformOp->main.value     = tc;
                                transformOp->name           = newOutputName;
                                transformOp->inputIndexes.push_back(idx);
                                int newOutputIndex = (int)mNet->tensorName.size();
                                transformOp->outputIndexes.push_back(newOutputIndex);
                                tensorFormats.push_back(originTensorType);
                                mNet->tensorName.push_back(outputs[j]);
                                transformOp->type   = MNN::OpType_ConvertTensor;
                                extraOp.emplace_back(transformOp);
                            }
                        }
                    }
                }
            }
            for (auto&& op : extraOp) {
                mNet->oplists.emplace_back(std::move(op));
            }
        } else {
            // Change Input
            for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
                auto& op         = *iter;
                if (OpType_Input == op->type) {
                    auto originInputFormat = op->main.AsInput()->dformat;
                    op->main.AsInput()->dformat = tensorFormats[op->outputIndexes[0]];
                    if (originInputFormat == MNN_DATA_FORMAT_NHWC && op->main.AsInput()->dformat == MNN_DATA_FORMAT_NC4HW4 && op->main.AsInput()->dims.size() == 4 && ctx->first_run) {
                        int n = op->main.AsInput()->dims[0];
                        int h = op->main.AsInput()->dims[1];
                        int w = op->main.AsInput()->dims[2];
                        int c = op->main.AsInput()->dims[3];
                        op->main.AsInput()->dims = {n, c, h, w};
                    }
                }
            }
        }
        if (originTensorType == MNN_DATA_FORMAT_NHWC) {
            for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
                auto op = iter->get();
                // Insert Pretreat Op if needed
                if (op->type == OpType_Padding && tensorFormats[op->outputIndexes[0]] == MNN_DATA_FORMAT_NC4HW4 && ctx->first_run) {
                    const int padValueIndex = op->inputIndexes[1];
                    auto padValueOp         = PostTreatUtils::_findOpByOutputIndex(padValueIndex, mNet.get());
                    // Add Gather op for padding, turn nhwc -> nchw
                    std::unique_ptr<OpT> gatherIndex(new OpT);
                    gatherIndex->outputIndexes = {(int)mNet->tensorName.size()};
                    gatherIndex->type          = OpType_Const;
                    gatherIndex->name          = op->name + "_Gather_Index";
                    mNet->tensorName.emplace_back(gatherIndex->name);
                    tensorFormats.push_back(originTensorType);
                    gatherIndex->main.type                 = OpParameter_Blob;
                    gatherIndex->main.value                = new BlobT;
                    gatherIndex->main.AsBlob()->dataType   = DataType_DT_INT32;
                    gatherIndex->main.AsBlob()->dataFormat = originTensorType;
                    gatherIndex->main.AsBlob()->int32s     = {0, 3, 1, 2};
                    gatherIndex->main.AsBlob()->dims       = {4};

                    std::unique_ptr<OpT> gather(new OpT);
                    gather->outputIndexes = {(int)mNet->tensorName.size()};
                    gather->inputIndexes  = {op->inputIndexes[1], gatherIndex->outputIndexes[0]};

                    gather->type = OpType_GatherV2;
                    gather->name = op->name + "_Gather";
                    mNet->tensorName.emplace_back(gather->name);
                    tensorFormats.push_back(originTensorType);

                    op->inputIndexes[1]                       = gather->outputIndexes[0];
                    iter = mNet->oplists.insert(iter, std::move(gather));
                    iter = mNet->oplists.insert(iter, std::move(gatherIndex));
                    iter++;
                    iter++;
                    iter++;
                } else {
                    iter++;
                }
            }
        }

        for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
            auto& op         = *iter;
            if (op->inputIndexes.empty()) {
                iter++;
                continue;
            }
            if (!_OpNeedConvertContent(op->type)) {
                iter++;
                continue;
            }
            auto formatType  = _getFormatType(op.get(), originTensorType);
            std::vector<MNN::OpT*> transformOps;
            auto currentName         = op->name;
            for (int i = 0; i < op->inputIndexes.size(); ++i) {
                auto inputIndex = op->inputIndexes[i];
                if (inputIndex < 0) {
                    continue; // optional input, ignore it
                }
                auto type = tensorFormats[inputIndex];
                auto requireType = _getRequireFormat(formatType, i, tensorFormats[op->outputIndexes[0]], originTensorType);
                if (type == requireType) {
                    continue;
                }

                if (convertMap.find(op->inputIndexes[i]) != convertMap.end()) {
                    op->inputIndexes[i] = convertMap[op->inputIndexes[i]];
                    continue;
                }

                // Insert Transform op
                MNN::OpT* transformOp = new MNN::OpT;
                transformOps.push_back(transformOp);
                MNN::TensorConvertInfoT* tc = new MNN::TensorConvertInfoT;
                tc->source                  = type;
                tc->dest                    = requireType;
                transformOp->main.type      = MNN::OpParameter_TensorConvertInfo;
                transformOp->main.value     = tc;
                transformOp->name           = mNet->tensorName[inputIndex] + "___tr4" + op->name;
                // printf("Insert convert for %s, %s 's input %d\n", net->tensorName[inputIndex].c_str(),
                // op->name.c_str(), i);
                transformOp->inputIndexes.push_back(inputIndex);
                transformOp->outputIndexes.push_back(mNet->tensorName.size());
                convertMap[inputIndex] = transformOp->outputIndexes[0];
                tensorFormats.push_back(requireType);
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

        if (originTensorType == MNN_DATA_FORMAT_NCHW) {
            return true;
        }

        // For NHWC -> NC4HW4 op, should Reset axis map
        const int axisMap[4] = {0, 2, 3, 1};

        for (auto& op : mNet->oplists) {
            if (op->inputIndexes.empty()) {
                continue;
            }
            if (tensorFormats[op->outputIndexes[0]] != MNN_DATA_FORMAT_NC4HW4) {
                continue;
            }
            if (!ctx->first_run) {
                continue;
            }
            if (MNN::OpType_Input == op->type) {
                auto input        = op->main.AsInput();
                const int dimSize = input->dims.size();
                if (dimSize > 2) {
                    const int channel = input->dims[dimSize - 1];
                    for (int i = dimSize - 1; i > 1; --i) {
                        input->dims[i] = input->dims[i - 1];
                    }
                    input->dims[1] = channel;
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
                auto slice     = op->main.AsSlice();
                auto sliceAxis = slice->axis;
                if (sliceAxis < 0) {
                    sliceAxis = 4 + sliceAxis;
                }
                DCHECK(sliceAxis >= 0 && sliceAxis <= 3) << "Slice axis ERROR!";
                slice->axis = axisMap[sliceAxis];
            }
            if (MNN::OpType_Reshape == op->type) {
                auto reshape   = op->main.AsReshape();
                auto originDim = reshape->dims;
                for (int i = 0; i < reshape->dims.size(); ++i) {
                    CHECK(i >= 0 && i <= 3) << "Error";
                    reshape->dims[axisMap[i]] = originDim[i];
                }
            }
            if (MNN::OpType_ArgMax == op->type || MNN::OpType_ArgMin == op->type) {
                auto param      = op->main.AsArgMax();
                auto originAxis = param->axis;
                DCHECK(originAxis >= 0 && originAxis <= 3) << "ArgMax / Argmin axis ERROR!";
                param->axis = axisMap[originAxis];
            }
        }
        return true;
    }
};
static PostConverterRegister<AddTensorFormatConverter> __l("AddTensorFormatConverter");
