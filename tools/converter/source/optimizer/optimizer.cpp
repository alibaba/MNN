//
//  optimizer.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "optimizer.hpp"
#include "PostTreatUtils.hpp"

std::unique_ptr<MNN::NetT> optimizeNet(std::unique_ptr<MNN::NetT>& originNet) {
    if (originNet->oplists.size() <= 0) {
        return nullptr;
    }

    std::unique_ptr<PostTreatUtils> postTool = std::unique_ptr<PostTreatUtils>(new PostTreatUtils(originNet));
    postTool->removeInplaceOp();

    // Turn Innerproduct to Convolution
    if (postTool->mNet->sourceType == MNN::NetSource_CAFFE) {
        postTool->turnInnerProduct2Convolution();
        postTool->treatIm2Seq();
    }

    postTool->merge2Convolution();
    // after merge, change the BatchNorm to Scale
    postTool->changeBatchnNorm2Scale();
    // Delete convolution's group parameter
    postTool->turnGroupConvolution();

    postTool->removeDeconvolutionShapeInput();

    postTool->deleteUnusefulOp();
    postTool->addTensorType();
    postTool->addConverterForTensorFlowModel();
    postTool->reIndexTensor();

    return std::unique_ptr<MNN::NetT>(std::move(postTool->mNet));
}
