//
//  PostConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PostConverter.hpp"
#include "PostTreatUtils.hpp"
#include "Program.hpp"
#include "TemplateMerge.hpp"
#include "Optimizer.hpp"
using namespace MNN::Express;

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
    
    auto program = MNN::Express::Program::create(originNet.get(), true);
    std::vector<std::string> optimizePass = {
        "Merge",
    };
    switch (originNet->sourceType) {
        case MNN::NetSource_TENSORFLOW:
            optimizePass.insert(optimizePass.begin(), "TFExtra");
            break;
        case MNN::NetSource_ONNX:
            optimizePass.insert(optimizePass.begin(), "OnnxExtra");
            break;
        default:
            break;
    }
    for (auto pass : optimizePass) {
        auto& merge  = MNN::Express::TemplateMerge::getInstance(pass);
        merge.onExecute(program->outputs());
    }
    if (program->needGenerateCode()) {
        MNN_PRINT("The Model Has Control / Extra Op, Please Compile the Code of model.cpp\n");
        std::ofstream code("model.cpp");
        code << "#include \"Expr.hpp\"\n";
        code << "#include \"ExprCreator.hpp\"\n";
        code << "using namespace MNN::Express;\n";
        code << "void extraCall(std::map<std::string, VARP>& varMap) {\n";
        program->emit(code);
        code << "}\n";
    }
    std::unique_ptr<MNN::NetT> newNet(new MNN::NetT);
    {
        auto outputs = program->outputs();
        newNet->sourceType = originNet->sourceType;
        newNet->bizCode = originNet->bizCode;
        Variable::save(outputs, newNet.get());
    }
    
    std::vector<std::string> afterProgramConvert = {
        // Turn BatchNormal to Scale When inference
        "TransformBatchNormal",

        // Merge Scale info Convolution
        "MergeToConvolution",

        // conert some binary op(add, mul, sub...) to element wise op(sum, sub) accroding to input condition
        "ConvertBinaryToElementwise",

        // Turn group convolution to Slice - Convolution - Concat
        "TransformGroupConvolution",

        // Add tensor dimension format convert for NC4HW4 - NHWC / NC4HW4 - NCHW
        "AddTensorFormatConverter",

        // Remove unuseful tensor
        "ReIndexTensor",
    };
    for (auto pass : afterProgramConvert) {
        auto convert = PostConverter::get(pass);
        if (nullptr == convert) {
            LOG(INFO) << "Can't find pass of " << pass << "\n";
            continue;
        }
        bool valid = convert->onExecute(newNet);
        if (!valid) {
            LOG(INFO) << "Run " << pass << "Error\n";
        }
    }
    
    std::set<int> inputSet;
    for (auto& op : newNet->oplists) {
        if (op->type == MNN::OpType_Input) {
            LOG(INFO) << "Inputs: " << newNet->tensorName[op->outputIndexes[0]];
            continue;
        }
        for (auto index : op->inputIndexes) {
            inputSet.insert(index);
        }
    }
    for (auto& op : newNet->oplists) {
        bool hasInput = false;
        for (auto index : op->outputIndexes) {
            if (inputSet.find(index) != inputSet.end()) {
                hasInput = true;
                break;
            }
        }
        if (!hasInput) {
            for (auto index : op->outputIndexes) {
                LOG(INFO) << "Outputs: " << newNet->tensorName[index]
                          << ", Type = " << MNN::EnumNameOpType(op->type);
            }
        }
    }
    
    return newNet;
}
