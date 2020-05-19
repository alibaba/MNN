//
//  AddTensorFormatConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace MNN;
const std::set<MNN::OpType> NC4HW4_OPs = {
    MNN::OpType_Convolution,
    MNN::OpType_Convolution3D,
    MNN::OpType_ConvolutionDepthwise,
    MNN::OpType_Pooling,
    MNN::OpType_Pooling3D,
    MNN::OpType_ROIPooling,
    MNN::OpType_Resize,
    MNN::OpType_LSTM,
    MNN::OpType_SpatialProduct,
    MNN::OpType_Deconvolution,
    MNN::OpType_DeconvolutionDepthwise,
    MNN::OpType_Proposal,
    MNN::OpType_PriorBox,
    MNN::OpType_DetectionOutput,
    MNN::OpType_LRN,
    MNN::OpType_Interp,
    MNN::OpType_Crop,
    MNN::OpType_Scale,
    MNN::OpType_TfQuantizedConv2D,
    MNN::OpType_QuantizedDepthwiseConv2D,
    MNN::OpType_BatchToSpaceND,
    MNN::OpType_BatchNorm,
    MNN::OpType_SpaceToBatchND,
    MNN::OpType_InstanceNorm,
    MNN::OpType_Moments,
    MNN::OpType_QuantizedAvgPool,
    MNN::OpType_QuantizedAdd,
    MNN::OpType_PReLU,
    MNN::OpType_Dilation2D,
};
const std::set<MNN::OpType> COMPABILITY_OPs = {MNN::OpType_ReLU,          MNN::OpType_ReLU6,   MNN::OpType_Concat,
                                               MNN::OpType_Slice,         MNN::OpType_Permute, MNN::OpType_Selu,
                                               MNN::OpType_ConvertTensor, MNN::OpType_Sigmoid, MNN::OpType_Cast,
                                               MNN::OpType_Reshape,       MNN::OpType_TanH, MNN::OpType_Eltwise,    MNN::OpType_Padding, MNN::OpType_ELU, MNN::OpType_Dropout};

static bool _OpNeedConvertContent(OpType type, int index) {
    switch (type) {
        case OpType_Shape:
        case OpType_PriorBox:
        case OpType_Const:
            return false;
        case OpType_Convolution:
        case OpType_Deconvolution:
        case OpType_ConvolutionDepthwise:
        case OpType_DeconvolutionDepthwise:
        case OpType_Convolution3D:
        case OpType_Interp:
        case OpType_Crop:
        case OpType_Reshape:
        case OpType_Resize:
        case OpType_Padding:
            if (1 <= index) {
                return false;
            }
            break;
        default:
            break;
    }
    return true;
}
class AddTensorFormatConverter : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& mNet = net;
        if (mNet->sourceType == MNN::NetSource_CAFFE || mNet->sourceType == MNN::NetSource_ONNX) {
            return true;
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
            } else if (NC4HW4_OPs.find(iter->type) != NC4HW4_OPs.end()) {
                type = MNN::MNN_DATA_FORMAT_NC4HW4;
            } else if (COMPABILITY_OPs.find(iter->type) != COMPABILITY_OPs.end()) {
                int nc4hw4TypeNumber = 0; // NC4HW4 number
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
                DCHECK(op->inputIndexes.size() == 2) << "Padding should have 2 inputs";
                const int padValueIndex = op->inputIndexes[1];
                auto padValueOp = PostTreatUtils::_findOpByOutputIndex(padValueIndex, mNet.get());
                if(opType.find(padValueOp->name)->second == MNN::MNN_DATA_FORMAT_NCHW){
                    iter++;
                    continue;
                }
                
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

                MNN::OpT* inputOp = PostTreatUtils::_findOpByOutputIndex(inputIndex, mNet.get());
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
                // printf("Insert convert for %s, %s 's input %d\n", net->tensorName[inputIndex].c_str(),
                // op->name.c_str(), i);
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
            return true;
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
