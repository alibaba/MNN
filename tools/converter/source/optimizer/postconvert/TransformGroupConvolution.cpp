//
//  TransformGroupConvolution.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
using namespace MNN;
class TransformGroupConvolution3D : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& mNet = net;
        // Delete Convolution With Grouop
        for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end();) {
            auto& op = *iter;
            if (op->type != MNN::OpType_Convolution3D) {
                iter++;
                continue;
            }
            auto conv3D  = op->main.AsConvolution3D();
            auto& common = conv3D->common;
            const int srcCount = common->inputCount;
            if (common->group == 1 || op->inputIndexes.size() > 1) {
                iter++;
                continue;
            }

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

            int partWeightSize = conv3D->weight.size() / common->group;
            int partBiasSize   = conv3D->bias.size() / common->group;

            // Create Sub Convolution
            for (int i = 0; i < common->group; ++i) {
                std::ostringstream opNameOs;
                auto newConvOp = new MNN::OpT;
                opNameOs << op->name << "__group__" << i;
                newConvOp->type      = op->type;
                newConvOp->name      = opNameOs.str();
                newConvOp->main.type = MNN::OpParameter_Convolution3D;
                newConvOp->inputIndexes.push_back(newConvolutionInputIndex[i]);
                newConvOp->outputIndexes.push_back(newConvolutionOutputIndex[i]);

                auto newConvolutionT    = new MNN::Convolution3DT;
                newConvOp->main.value   = newConvolutionT;
                newConvolutionT->common = std::unique_ptr<MNN::Convolution3DCommonT>(new MNN::Convolution3DCommonT);
                newConvolutionT->common->dilates     = common->dilates;
                newConvolutionT->common->strides     = common->strides;
                newConvolutionT->common->kernels     = common->kernels;
                newConvolutionT->common->pads        = common->pads;
                newConvolutionT->common->group       = 1;
                newConvolutionT->common->padMode     = common->padMode;
                newConvolutionT->common->outputCount = common->outputCount / common->group;
                newConvolutionT->common->inputCount  = common->inputCount / common->group;
                newConvolutionT->common->relu        = common->relu;
                newConvolutionT->common->relu6       = common->relu6;

                int startWeight = partWeightSize * i;
                int startBias   = partBiasSize * i;
                for (int v = 0; v < partWeightSize; ++v) {
                    newConvolutionT->weight.push_back(conv3D->weight[startWeight + v]);
                }
                for (int v = 0; v < partBiasSize; ++v) {
                    newConvolutionT->bias.push_back(conv3D->bias[startBias + v]);
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
        return true;
    }
};

class TransformGroupConvolution : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& mNet = net;
        for (auto iter = mNet->oplists.begin(); iter != mNet->oplists.end(); iter++) {
            auto& op           = *iter;
            const auto op_type = op->type;
            auto conv2D        = op->main.AsConvolution2D();
            if (op_type == MNN::OpType_Convolution || op_type == MNN::OpType_Deconvolution) {
                auto& common       = conv2D->common;
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
                    // const int srcCount =
                    //     conv2D->weight.size() * common->group / common->outputCount / common->kernelX /
                    //     common->kernelY;
                    // get srcCount from conv param common args: inputCount, not use weight to compute(in some case,
                    // weight is empty)
                    const int srcCount = common->inputCount;
                    turnConv2DW        = common->outputCount == common->group && srcCount == common->outputCount;
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
            const int srcCount = common->inputCount;
            const bool depthwiseLike = srcCount % common->group != 0 || common->outputCount % common->group != 0;
            if (common->group == 1 || depthwiseLike) {
                iter++;
                continue;
            }

            // int srcCount =
            //     conv2D->weight.size() * common->group / common->outputCount / common->kernelX / common->kernelY;

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
                sliceOp->inputIndexes  = {op->inputIndexes[0]};
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

            if(op->inputIndexes.size() > 1){
                std::vector<int> newConvolutionWeightInputIndex;
                std::vector<int> newConvolutionBiasInputIndex;
                // splice weight
                {
                    for (int i = 0; i < common->group; ++i) {
                        std::ostringstream newTensorNameOs;
                        newTensorNameOs << op->name << "___input___weight___" << i;
                        newConvolutionWeightInputIndex.push_back(mNet->tensorName.size());
                        mNet->tensorName.push_back(newTensorNameOs.str());
                    }

                    // Create slice op for weight
                    {
                        MNN::OpT* sliceOp      = new MNN::OpT;
                        sliceOp->type          = MNN::OpType_Slice;
                        sliceOp->name          = op->name + "_____weight_____slice";
                        sliceOp->inputIndexes  = {op->inputIndexes[1]};
                        sliceOp->outputIndexes = newConvolutionWeightInputIndex;
                        auto sliceT            = new MNN::SliceT;
                        sliceOp->main.type     = MNN::OpParameter_Slice;
                        sliceOp->main.value    = sliceT;
                        sliceT->axis           = 0;
                        for (int i = 0; i < common->group - 1; ++i) {
                            sliceT->slicePoints.push_back(common->outputCount / (common->group) * (i + 1));
                        }
                        newOp.push_back(sliceOp);
                    }
                }
                // slice bias
                if(op->inputIndexes.size() == 3){
                    for (int i = 0; i < common->group; ++i) {
                        std::ostringstream newTensorNameOs;
                        newTensorNameOs << op->name << "___input___bias___" << i;
                        newConvolutionBiasInputIndex.push_back(mNet->tensorName.size());
                        mNet->tensorName.push_back(newTensorNameOs.str());
                    }

                    // Create slice op for bias
                    {
                        MNN::OpT* sliceOp      = new MNN::OpT;
                        sliceOp->type          = MNN::OpType_Slice;
                        sliceOp->name          = op->name + "_____bias_____slice";
                        sliceOp->inputIndexes  = {op->inputIndexes[2]};
                        sliceOp->outputIndexes = newConvolutionBiasInputIndex;
                        auto sliceT            = new MNN::SliceT;
                        sliceOp->main.type     = MNN::OpParameter_Slice;
                        sliceOp->main.value    = sliceT;
                        sliceT->axis           = 0;
                        for (int i = 0; i < common->group - 1; ++i) {
                            sliceT->slicePoints.push_back(common->outputCount / (common->group) * (i + 1));
                        }
                        newOp.push_back(sliceOp);
                    }
                }
                // Create Sub Convolution
                flatbuffers::FlatBufferBuilder tmpBuilder;
                tmpBuilder.Finish(Convolution2DCommon::Pack(tmpBuilder, common.get()));
                auto originCommon = flatbuffers::GetRoot<Convolution2DCommon>(tmpBuilder.GetBufferPointer());
                for (int i = 0; i < common->group; ++i) {
                    std::ostringstream opNameOs;
                    auto newConvOp = new MNN::OpT;
                    opNameOs << op->name << "__group__" << i;
                    newConvOp->type      = op->type;
                    newConvOp->name      = opNameOs.str();
                    newConvOp->main.type = MNN::OpParameter_Convolution2D;
                    newConvOp->inputIndexes.push_back(newConvolutionInputIndex[i]);
                    newConvOp->inputIndexes.push_back(newConvolutionWeightInputIndex[i]);
                    if(op->inputIndexes.size() == 3){
                        newConvOp->inputIndexes.push_back(newConvolutionBiasInputIndex[i]);
                    }
                    newConvOp->outputIndexes.push_back(newConvolutionOutputIndex[i]);
                        
                    auto newConvolutionT    = new MNN::Convolution2DT;
                    newConvOp->main.value   = newConvolutionT;
                    newConvolutionT->common = std::unique_ptr<MNN::Convolution2DCommonT>(originCommon->UnPack());
                    newConvolutionT->common->group       = 1;
                    newConvolutionT->common->outputCount = common->outputCount / common->group;
                    newConvolutionT->common->inputCount  = common->inputCount / common->group;
                    newOp.push_back(newConvOp);
                }
            }else{
                int partWeightSize = conv2D->weight.size() / common->group;
                int partBiasSize   = conv2D->bias.size() / common->group;
                
                // Create Sub Convolution
                flatbuffers::FlatBufferBuilder tmpBuilder;
                tmpBuilder.Finish(Convolution2DCommon::Pack(tmpBuilder, common.get()));
                auto originCommon = flatbuffers::GetRoot<Convolution2DCommon>(tmpBuilder.GetBufferPointer());
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
                    newConvolutionT->common = std::unique_ptr<MNN::Convolution2DCommonT>(originCommon->UnPack());
                    newConvolutionT->common->group       = 1;
                    newConvolutionT->common->outputCount = common->outputCount / common->group;
                    newConvolutionT->common->inputCount  = common->inputCount / common->group;
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
        return true;
    }
};
static PostConverterRegister<TransformGroupConvolution> __l("TransformGroupConvolution");
static PostConverterRegister<TransformGroupConvolution3D> __l3d("TransformGroupConvolution3D");
