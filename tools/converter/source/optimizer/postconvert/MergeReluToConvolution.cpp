//
//  MergeReluToConvolution.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include "MergeToConvolution.hpp"

using namespace MNN;

class MergeReluToConvolution : public MergeToConvolution {
public:
    bool merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const {
        if (inplaceOp->type == MNN::OpType_ReLU && inplaceOp->main.AsRelu()->slope == 0.0f) {
            convolutionOp->main.AsConvolution2D()->common->relu = true;
            return true;
        }
        return false;
    }

    bool merge2Convolution3D(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const {
        if (inplaceOp->type == MNN::OpType_ReLU && inplaceOp->main.AsRelu()->slope == 0.0f) {
            convolutionOp->main.AsConvolution3D()->common->relu = true;
            return true;
        }
        return false;
    }
};
static PostConverterRegister<MergeReluToConvolution> __l("MergeReluToConvolution");
