//
//  ROIPooling.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
using namespace std;
class RoiPooling : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        auto roiPool          = new MNN::RoiPoolingT;
        auto roiPoolCaffe     = parameters.roi_pooling_param();
        roiPool->pooledHeight = roiPoolCaffe.pooled_h();
        roiPool->pooledWidth  = roiPoolCaffe.pooled_w();
        roiPool->spatialScale = roiPoolCaffe.spatial_scale();
        dstOp->main.value     = roiPool;
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_ROIPooling;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_RoiPooling;
    }
};

static OpConverterRegister<RoiPooling> a("ROIPooling");
