//
//  TransformInnerProduct.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"

class TransformInnerProduct : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        std::vector<MNN::OpT*> readyToDelete;
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op = *iter;
            if (op->type != MNN::OpType_InnerProduct) {
                iter++;
                continue;
            }
            for (int i = 1; i < op->inputIndexes.size(); ++i) {
                auto uselessConst = PostTreatUtils::_findOpByOutputIndex(op->inputIndexes[i], net.get());
                readyToDelete.emplace_back(uselessConst);
            }
            // ONNX Gemm will be mapped to InnerProduct, check whether is Flatten before Gemm
            // then delete Flatten(mapped to Reshape, and this Reshape will reshape tensor to be
            // two dimensions, such as [M,K], which is the input of Gemm)
            auto inputId       = op->inputIndexes[0];
            auto beforeGemm    = PostTreatUtils::_findOpByOutputIndex(inputId, net.get());
            auto refBeforeGemm = PostTreatUtils::_findOpByInputIndex(beforeGemm->outputIndexes[0], net.get());
            if (beforeGemm->type == MNN::OpType_Reshape && PostTreatUtils::_isSingleInputOutput(beforeGemm) &&
                refBeforeGemm.size() == 1) {
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
            if (net->sourceType == MNN::NetSource_TENSORFLOW) {
                reshapeP->dims[3] = -1;
                reshapeP->dims[1] = 1;
                reshapeP->dims[2] = 1;
            }

            reshapeT->main.type  = MNN::OpParameter_Reshape;
            reshapeT->type       = MNN::OpType_Reshape;
            reshapeT->main.value = reshapeP;

            // Net Tensor
            net->tensorName.push_back(reshapeT->name);
            int tempId = net->tensorName.size() - 1;

            reshapeT->inputIndexes.push_back(inputId);
            reshapeT->outputIndexes.push_back(tempId);
            auto opName      = op->name;
            bool needPermute = 1 != axis && net->sourceType == MNN::NetSource_CAFFE;

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
                net->tensorName.push_back(permuteBefore->name);
                tempId = net->tensorName.size() - 1;
                permuteBefore->outputIndexes.push_back(tempId);

                newOpPrevious.push_back(permuteBefore);
            }

            op->inputIndexes = {tempId};
            op->type         = MNN::OpType_Convolution;

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
            convP->common->inputCount  = originInner->weight.size() / originInner->outputCount;
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

            const int finalOutputIndex = op->outputIndexes[0];

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
                net->tensorName.push_back(permuteBefore->name);
                tempId = net->tensorName.size() - 1;
                permuteBefore->inputIndexes.push_back(tempId);
                permuteBefore->outputIndexes.push_back(finalOutputIndex);
                op->outputIndexes[0] = tempId;

                newOpPost.push_back(permuteBefore);
            }

            if (axis + 1 != 4) {
                MNN::OpT* afterReshapeT = new MNN::OpT;
                afterReshapeT->name     = "____reshape2____" + op->name;
                auto reshapeP           = new MNN::ReshapeT;
                reshapeP->dims.resize(axis + 1);
                for (int i = 0; i < axis; ++i) {
                    reshapeP->dims[i] = 0;
                }
                reshapeP->dims[axis]      = -1;
                afterReshapeT->main.type  = MNN::OpParameter_Reshape;
                afterReshapeT->type       = MNN::OpType_Reshape;
                afterReshapeT->main.value = reshapeP;

                net->tensorName.push_back(afterReshapeT->name);
                tempId = net->tensorName.size() - 1;
                afterReshapeT->inputIndexes.push_back(tempId);
                if (newOpPost.size() > 0) {
                    newOpPost[newOpPost.size() - 1]->outputIndexes[0] = tempId;
                } else {
                    op->outputIndexes[0] = tempId;
                }
                afterReshapeT->outputIndexes.push_back(finalOutputIndex);
                newOpPost.push_back(afterReshapeT);
            }

            for (int i = 0; i < newOpPrevious.size(); ++i) {
                iter =
                    net->oplists.insert(iter, std::unique_ptr<MNN::OpT>(newOpPrevious[newOpPrevious.size() - i - 1]));
            }

            for (;; iter++) {
                auto& op = *iter;
                if (op->name == opName) {
                    break;
                }
            }

            for (int i = 0; i < newOpPost.size(); ++i) {
                iter = net->oplists.insert(iter + 1, std::unique_ptr<MNN::OpT>(newOpPost[i]));
            }
        }
        for (auto op : readyToDelete) {
            PostTreatUtils::_removeOpInNet(op, net.get());
        }
        return true;
    }
};
static PostConverterRegister<TransformInnerProduct> __l("TransformInnerProduct");
