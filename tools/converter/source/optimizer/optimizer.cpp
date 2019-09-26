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
    std::vector<std::string> postConvertPass;
    postConvertPass = {
        // Seperate Tensor for inplace op
        "RemoveInplace",

        // Remove Unuseful Op such as NoOp, Identity, Dropout, Seq2Out,
        "RemoveUnusefulOp",

        // Turn InnerProduct from Caffe / Onnx to Convolution
        "TransformInnerProduct",

        // Turn Im2Seq from Caffe to Reshape
        "TransformIm2Seq",

        // Turn Caffe's ShuffleChannel to compose op
        "TransformShuffleChannel",

        // Turn Onnx's Pad to Tensorflow's Pad
        "TransformOnnxPad",

        // Turn BatchNormal to Scale When inference
        "TransformBatchNormal",

        // Merge Scale info Convolution
        "MergeToConvolution",

        // Remove unuseful shape op for tensorflow's deconvolution
        "RemoveDeconvolutionShapeInput",

        // conert some binary op(add, mul, sub...) to element wise op(sum, sub) accroding to input condition
        "ConvertBinaryToElementwise",

        // Turn group convolution to Slice - Convolution - Concat
        "TransformGroupConvolution",

        // Add tensor dimension format convert for NC4HW4 - NHWC / NC4HW4 - NCHW
        "AddTensorFormatConverter",

        // Depercrate
        "AddTensorType",

        // Remove unuseful tensor
        "ReIndexTensor",
    };
    for (auto pass : postConvertPass) {
        auto convert = PostConverter::get(pass);
        if (nullptr == convert) {
            LOG(INFO) << "Can't find pass of " << pass << "\n";
            continue;
        }
        bool valid = convert->onExecute(originNet);
        if (!valid) {
            LOG(INFO) << "Run " << pass << "Error\n";
        }
    }
    std::set<int> inputSet;
    for (auto& op : originNet->oplists) {
        if (op->type == MNN::OpType_Input) {
            LOG(INFO) << "Inputs: " << op->name;
            continue;
        }
        for (auto index : op->inputIndexes) {
            inputSet.insert(index);
        }
    }
    for (auto& op : originNet->oplists) {
        for (auto index : op->outputIndexes) {
            if (inputSet.find(index) == inputSet.end()) {
                LOG(INFO) << "Outputs: " << originNet->tensorName[index]
                          << ", Type = " << MNN::EnumNameOpType(op->type);
                break;
            }
        }
    }
    return std::move(originNet);
}
