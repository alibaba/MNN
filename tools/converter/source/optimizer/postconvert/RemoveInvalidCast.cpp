//
//  RemoveInvalidCast.cpp
//  MNNConverter
//
//  Created by MNN on 2021/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include "../PostTreatUtils.hpp"
#include <MNN/MNNDefine.h>
using namespace MNN;
class RemoveInvalidCast : public PostConverter {
public:
    static bool outputBool(int operation) {
        if (operation == BinaryOpOperation_GREATER_EQUAL) {
            return true;
        }
        if (operation == BinaryOpOperation_GREATER) {
            return true;
        }
        if (operation == BinaryOpOperation_LESS) {
            return true;
        }
        if (operation == BinaryOpOperation_LESS_EQUAL) {
            return true;
        }
        if (operation == BinaryOpOperation_EQUAL) {
            return true;
        }
        if (operation == BinaryOpOperation_NOTEQUAL) {
            return true;
        }
        return false;
    }
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        if (net->sourceType == MNN::NetSource_TENSORFLOW || net->sourceType == MNN::NetSource_TFLITE) {
            // The two framework has valid src type for cast, don't need treat
            return true;
        }
        if (net->sourceType == MNN::NetSource_CAFFE) {
            // For caffe has no invalid cast op
            return true;
        }
        bool needTreat = false;
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            auto& op = *iter;
            if (op->type == MNN::OpType_Cast) {
                needTreat = true;
                break;
            }
        }
        if (!needTreat) {
            return true;
        }
        // Infer DataType for All Tensor
        std::vector<MNN::DataType> types(net->tensorName.size(), MNN::DataType_DT_INVALID);
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            auto& op = *iter;
            switch (op->type) {
                // Float Op
                case MNN::OpType_PReLU:
                case MNN::OpType_Softmax:
                case MNN::OpType_Convolution:
                case MNN::OpType_ConvolutionDepthwise:
                case MNN::OpType_Convolution3D:
                case MNN::OpType_Deconvolution:
                case MNN::OpType_DeconvolutionDepthwise:
                case MNN::OpType_Interp:
                case MNN::OpType_LSTM:
                case MNN::OpType_LSTMBlockCell:
                case MNN::OpType_GridSample:
                case MNN::OpType_RNNSequenceGRU:
                case MNN::OpType_MatMul:
                    types[op->inputIndexes[0]] = MNN::DataType_DT_FLOAT;
                    if (op->outputIndexes.size() == 1) {
                        // 4 is integer matmul
                        types[op->outputIndexes[0]] = MNN::DataType_DT_FLOAT;
                    }
                    break;
                default:
                    break;
            }
        }
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            auto& op = *iter;
            switch (op->type) {
                case MNN::OpType_Input:
                    types[op->outputIndexes[0]] = op->main.AsInput()->dtype;
                    break;
                case MNN::OpType_Cast:
                    types[op->outputIndexes[0]] = op->main.AsCastParam()->dstT;
                    break;
                case MNN::OpType_CastLike:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[1]];
                    break;
                case MNN::OpType_Const:
                case MNN::OpType_TrainableParam:
                    types[op->outputIndexes[0]] = op->main.AsBlob()->dataType;
                    break;
                case MNN::OpType_Fill:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[1]];
                    break;
                case MNN::OpType_Slice:
                case MNN::OpType_SliceTf:
                case MNN::OpType_Unpack:
                    for (auto v : op->outputIndexes) {
                        types[v] = types[op->inputIndexes[0]];
                    }
                    break;
                case MNN::OpType_GatherV2:
                case MNN::OpType_GatherND:
                case MNN::OpType_Reduction:
                case MNN::OpType_Range:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[0]];
                    break;
                case MNN::OpType_Shape:
                case MNN::OpType_Size:
                case MNN::OpType_Rank:
                case MNN::OpType_UnravelIndex:
                    types[op->outputIndexes[0]] = MNN::DataType_DT_INT32;
                    break;
                case MNN::OpType_Unique:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[0]];
                    for (int v=1; v<op->outputIndexes.size(); ++v) {
                        types[op->outputIndexes[v]] = MNN::DataType_DT_INT32;
                    }
                    break;
                case MNN::OpType_RandomUniform:
                    types[op->outputIndexes[0]] = op->main.AsRandomUniform()->type;
                    break;
                case MNN::OpType_ArgMax:
                    types[op->outputIndexes[0]] = MNN::DataType_DT_INT32;
                    break;
                case MNN::OpType_TopKV2:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[0]];
                    if (op->outputIndexes.size() > 1) {
                        types[op->outputIndexes[1]] = MNN::DataType_DT_INT32;
                    }
                    break;
                case MNN::OpType_ScatterNd:
                case MNN::OpType_Select:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[1]];
                    break;
                case MNN::OpType_OneHot:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[2]];
                    break;
                case MNN::OpType_Extra:
                case MNN::OpType_Plugin:
                    break;
                case MNN::OpType_BinaryOp:
                {
                    if (outputBool(op->main.AsBinaryOp()->opType)) {
                        types[op->outputIndexes[0]] = DataType_DT_BOOL;
                    } else {
                        types[op->outputIndexes[0]] = types[op->inputIndexes[0]];
                    }
                }
                    break;
                // Deform
                case MNN::OpType_Broastcast:
                case MNN::OpType_Concat:
                case MNN::OpType_Crop:
                case MNN::OpType_CropAndResize:
                case MNN::OpType_Col2Im:
                case MNN::OpType_DepthToSpace:
                case MNN::OpType_ExpandDims:
                case MNN::OpType_Flatten:
                case MNN::OpType_Interp:
                case MNN::OpType_Interp3D:
                case MNN::OpType_Im2Col:
                case MNN::OpType_Pack:
                case MNN::OpType_Padding:
                case MNN::OpType_Permute:
                case MNN::OpType_Reshape:
                case MNN::OpType_Resize:
                case MNN::OpType_StridedSlice:
                case MNN::OpType_SpaceToDepth:
                case MNN::OpType_Squeeze:
                case MNN::OpType_Transpose:
                case MNN::OpType_Unsqueeze:
                {
                    types[op->outputIndexes[0]] = types[op->inputIndexes[0]];
                }
                    break;
                default:
                    break;
            }
        }
        // Remove Useless Cast
        const MNN::NetT* const netPtr = net.get();
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op          = *iter;
            if (op->type != MNN::OpType_Cast && op->type != MNN::OpType_CastLike) {
                iter++;
                continue;
            }
            if (types[op->inputIndexes[0]] == MNN::DataType_DT_INVALID) {
                iter++;
                continue;
            }
            if (types[op->inputIndexes[0]] != types[op->outputIndexes[0]]) {
                auto type = types[op->outputIndexes[0]];
                if (op->type == MNN::OpType_CastLike) {
                    if (type != MNN::DataType_DT_INVALID) {
                        // Turn Castlike to cast
                        op->type = MNN::OpType_Cast;
                        op->inputIndexes = {op->inputIndexes[0]};
                        op->main.Reset();
                        op->main.value = new CastParamT;
                        op->main.type = OpParameter_CastParam;
                        op->main.AsCastParam()->dstT = type;
                    }
                }
                iter++;
                continue;
            }
            if (std::find(net->outputName.begin(), net->outputName.end(), net->tensorName[op->outputIndexes[0]]) != net->outputName.end()) {
                iter++;
                continue;
            }
            // Find the next op
            if (op->outputIndexes.empty() || op->inputIndexes.empty()) {
                iter = net->oplists.erase(iter);
                continue;
            }

            auto originInput  = op->inputIndexes[0];
            auto originOutputs = op->outputIndexes;
            for (auto subIter = net->oplists.begin(); subIter != net->oplists.end(); subIter++) {
                auto& subOp = *subIter;
                for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
                    if (std::find(originOutputs.begin(), originOutputs.end(), subOp->inputIndexes[v]) != originOutputs.end()) {
                        subOp->inputIndexes[v] = originInput;
                    }
                }
            }
            iter = net->oplists.erase(iter);
        }
        return true;
    }
};
static PostConverterRegister<RemoveInvalidCast> __l("RemoveInvalidCast");
