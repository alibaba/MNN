//
//  MergeToConvolution.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace MNN;
static bool _merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) {
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

static bool _merge2Convolution3D(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) {
    if (inplaceOp->type == MNN::OpType_ReLU && inplaceOp->main.AsRelu()->slope == 0.0f) {
        convolutionOp->main.AsConvolution3D()->common->relu = true;
        return true;
    }
    if (inplaceOp->type == MNN::OpType_ReLU6) {
        convolutionOp->main.AsConvolution3D()->common->relu6 = true;
        return true;
    }
    
    const auto& convCommon = convolutionOp->main.AsConvolution3D()->common;
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
        
        auto conv3D     = convolutionOp->main.AsConvolution3D();
        int outputCount = conv3D->common->outputCount;
        for (int i = 0; i < outputCount; ++i) {
            conv3D->bias[i] = conv3D->bias[i] * alpha[i] + bias[i];
        }
        
        int weightPartSize = conv3D->weight.size() / outputCount;
        for (int i = 0; i < outputCount; ++i) {
            float a = alpha[i];
            for (int j = 0; j < weightPartSize; ++j) {
                conv3D->weight[i * weightPartSize + j] *= a;
            }
        }
        return true;
    }
    
    return false;
}

class MergeToConvolution : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        // Merge Layer
        std::vector<MNN::OpT*> readyToDelete;
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            MNN::OpT& currentOp = *(iter->get());
            if (currentOp.type != MNN::OpType_Convolution
                && currentOp.type != MNN::OpType_Deconvolution
                && currentOp.type != MNN::OpType_ConvolutionDepthwise
                && currentOp.type != MNN::OpType_Convolution3D) {
                continue;
            }
            DCHECK(currentOp.outputIndexes.size() == 1) << "Conv output ERROR!";

            // merge Batchnorm/Relu/Relu6 to Convolution
            std::vector<MNN::OpT*> nextOp = PostTreatUtils::_findOpByInputIndex(currentOp.outputIndexes[0], net.get());
            while (1) {
                if (nextOp.size() != 1) {
                    break;
                }
                const int nextOutputIndex = nextOp[0]->outputIndexes[0];
                bool succ;
                if (currentOp.type == MNN::OpType_Convolution3D) {
                    succ = _merge2Convolution3D(nextOp[0], &currentOp);
                } else {
                    succ = _merge2Convolution(nextOp[0], &currentOp);
                }
                if (PostTreatUtils::_isSingleInputOutput(nextOp[0]) && succ) {
                    // LOG(INFO) << "Merge " << nextOp[0]->name.c_str()<< " into convolution: " <<
                    // currentOp.name.c_str();
                    currentOp.outputIndexes[0] = nextOp[0]->outputIndexes[0];
                    readyToDelete.push_back(nextOp[0]);
                    nextOp = PostTreatUtils::_findOpByInputIndex(nextOutputIndex, net.get());
                } else {
                    break;
                }
            }
        }
        for (auto op : readyToDelete) {
            PostTreatUtils::_removeOpInNet(op, net.get());
        }
        return true;
    }
};
static PostConverterRegister<MergeToConvolution> __l("MergeToConvolution");
