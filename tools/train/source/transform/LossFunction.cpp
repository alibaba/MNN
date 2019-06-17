//
//  LossFunction.cpp
//  MNN
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "LossFunction.hpp"
using namespace MNN;
MNN::OpT* LossFunction::addSubEclLoss(MNN::NetT* net, const MNN::OpT* lastOp,
                                      std::map<int, std::shared_ptr<Tensor>>& tensors) {
    MNN::OpT* lossOp = nullptr;
    OpConverter::Result result;
    auto targetOutput = lastOp->outputIndexes[0];
    // Add Real Input
    int realInputId   = 0;
    auto originTensor = tensors[targetOutput];
    auto dimType      = originTensor->getDimensionType();
    {
        std::unique_ptr<OpT> newOp(new OpT);
        newOp->type = OpType_Input;
        newOp->name = lastOp->name + "_Compare";
        newOp->outputIndexes.emplace_back(net->tensorName.size());
        realInputId       = newOp->outputIndexes[0];
        newOp->main.type  = OpParameter_Input;
        auto input        = new InputT;
        input->dtype      = DataType_DT_FLOAT;
        input->dims       = originTensor->shape();
        newOp->main.value = input;
        if (Tensor::CAFFE == dimType) {
            input->dformat = MNN_DATA_FORMAT_NC4HW4;
        } else if (Tensor::TENSORFLOW == dimType) {
            input->dformat = MNN_DATA_FORMAT_NHWC;
        }
        net->tensorName.emplace_back(newOp->name);
        std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>(originTensor->shape(), dimType));
        tensors[newOp->outputIndexes[0]] = (tensor);

        net->oplists.emplace_back(std::move(newOp));
    }
    // Add Loss Compute
    {
        std::unique_ptr<OpT> subOp(new OpT);
        subOp->type                      = OpType_BinaryOp;
        subOp->main.value                = new BinaryOpT;
        subOp->main.type                 = OpParameter_BinaryOp;
        subOp->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
        subOp->main.AsBinaryOp()->opType = BinaryOpOperation_SUB;
        subOp->name                      = lastOp->name + "_CompareSub";
        subOp->inputIndexes              = {targetOutput, realInputId};
        subOp->outputIndexes.emplace_back(net->tensorName.size());
        net->tensorName.emplace_back(subOp->name);
        {
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>(originTensor->shape(), dimType));
            tensors[subOp->outputIndexes[0]] = (tensor);
        }

        std::unique_ptr<OpT> mulOp(new OpT);
        mulOp->type                      = OpType_BinaryOp;
        mulOp->main.type                 = OpParameter_BinaryOp;
        mulOp->main.value                = new BinaryOpT;
        mulOp->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
        mulOp->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
        mulOp->name                      = lastOp->name + "_CompareMul";
        mulOp->inputIndexes              = {subOp->outputIndexes[0], subOp->outputIndexes[0]};
        mulOp->outputIndexes.emplace_back(net->tensorName.size());
        net->tensorName.emplace_back(mulOp->name);
        {
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>(originTensor->shape(), dimType));
            tensors[mulOp->outputIndexes[0]] = (tensor);
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->type                               = OpType_Reduction;
        newOp->name                               = "Loss";
        newOp->main.type                          = OpParameter_ReductionParam;
        newOp->main.value                         = new ReductionParamT;
        newOp->main.AsReductionParam()->keepDims  = false;
        newOp->main.AsReductionParam()->dType     = DataType_DT_FLOAT;
        newOp->main.AsReductionParam()->operation = ReductionType_SUM;

        newOp->inputIndexes = {mulOp->outputIndexes[0]};
        newOp->outputIndexes.emplace_back(net->tensorName.size());
        net->tensorName.emplace_back(newOp->name);
        {
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>({}));
            tensors[newOp->outputIndexes[0]] = (tensor);
        }
        lossOp = newOp.get();

        net->oplists.emplace_back(std::move(subOp));
        net->oplists.emplace_back(std::move(mulOp));
        net->oplists.emplace_back(std::move(newOp));
    }
    return lossOp;
}

MNN::OpT* LossFunction::addProbLoss(MNN::NetT* net, const MNN::OpT* lastOp,
                                    std::map<int, std::shared_ptr<MNN::Tensor>>& tensors) {
    MNN::OpT* lossOp = nullptr;
    OpConverter::Result result;
    auto targetOutput = lastOp->outputIndexes[0];
    // Add Real Input
    int realInputId   = 0;
    auto originTensor = tensors[targetOutput];
    auto dimType      = originTensor->getDimensionType();
    {
        std::unique_ptr<OpT> newOp(new OpT);
        newOp->type = OpType_Input;
        newOp->name = lastOp->name + "_Compare";
        newOp->outputIndexes.emplace_back(net->tensorName.size());
        realInputId       = newOp->outputIndexes[0];
        newOp->main.type  = OpParameter_Input;
        auto input        = new InputT;
        input->dtype      = DataType_DT_FLOAT;
        input->dims       = originTensor->shape();
        newOp->main.value = input;
        if (Tensor::CAFFE == dimType) {
            input->dformat = MNN_DATA_FORMAT_NC4HW4;
        } else if (Tensor::TENSORFLOW == dimType) {
            input->dformat = MNN_DATA_FORMAT_NHWC;
        }
        net->tensorName.emplace_back(newOp->name);
        std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>(originTensor->shape(), dimType));
        tensors[newOp->outputIndexes[0]] = (tensor);

        net->oplists.emplace_back(std::move(newOp));
    }
    // Add Loss Compute
    {
        int currenctOutput = 0;
        std::unique_ptr<OpT> mulOp(new OpT);
        mulOp->type                      = OpType_BinaryOp;
        mulOp->main.value                = new BinaryOpT;
        mulOp->main.type                 = OpParameter_BinaryOp;
        mulOp->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
        mulOp->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
        mulOp->name                      = lastOp->name + "_CompareMul";
        mulOp->inputIndexes              = {targetOutput, realInputId};
        mulOp->outputIndexes.emplace_back(net->tensorName.size());
        net->tensorName.emplace_back(mulOp->name);
        {
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>(originTensor->shape(), dimType));
            tensors[mulOp->outputIndexes[0]] = (tensor);
        }
        currenctOutput = mulOp->outputIndexes[0];
        net->oplists.emplace_back(std::move(mulOp));
        if (dimType == Tensor::CAFFE) {
            std::unique_ptr<OpT> convertOp(new OpT);
            convertOp->type                               = OpType_ConvertTensor;
            convertOp->main.value                         = new TensorConvertInfoT;
            convertOp->main.type                          = OpParameter_TensorConvertInfo;
            convertOp->main.AsTensorConvertInfo()->source = MNN_DATA_FORMAT_NC4HW4;
            convertOp->main.AsTensorConvertInfo()->dest   = MNN_DATA_FORMAT_NHWC;
            convertOp->name                               = lastOp->name + "_CompareMul_Convert";
            convertOp->inputIndexes                       = {currenctOutput};
            convertOp->outputIndexes.emplace_back(net->tensorName.size());
            net->tensorName.emplace_back(convertOp->name);
            {
                std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>(
                    {originTensor->batch(), originTensor->height(), originTensor->width(), originTensor->channel()},
                    dimType));
                tensors[convertOp->outputIndexes[0]] = (tensor);
            }
            currenctOutput = convertOp->outputIndexes[0];
            net->oplists.emplace_back(std::move(convertOp));
        }

        std::unique_ptr<OpT> LossSum(new OpT);
        LossSum->type                               = OpType_Reduction;
        LossSum->name                               = "LossSum";
        LossSum->main.type                          = OpParameter_ReductionParam;
        LossSum->main.value                         = new ReductionParamT;
        LossSum->main.AsReductionParam()->keepDims  = false;
        LossSum->main.AsReductionParam()->dType     = DataType_DT_FLOAT;
        LossSum->main.AsReductionParam()->operation = ReductionType_SUM;
        LossSum->main.AsReductionParam()->dim       = {1, 2, 3};

        LossSum->inputIndexes = {currenctOutput};
        LossSum->outputIndexes.emplace_back(net->tensorName.size());
        net->tensorName.emplace_back(LossSum->name);
        {
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>({originTensor->batch()}));
            tensors[LossSum->outputIndexes[0]] = (tensor);
        }

        std::unique_ptr<OpT> LogLoss(new OpT);
        LogLoss->type                     = OpType_UnaryOp;
        LogLoss->name                     = "LogLoss";
        LogLoss->main.type                = OpParameter_UnaryOp;
        LogLoss->main.value               = new UnaryOpT;
        LogLoss->main.AsUnaryOp()->T      = DataType_DT_FLOAT;
        LogLoss->main.AsUnaryOp()->opType = UnaryOpOperation_LOG;

        LogLoss->inputIndexes = {LossSum->outputIndexes[0]};
        LogLoss->outputIndexes.emplace_back(net->tensorName.size());
        net->tensorName.emplace_back(LogLoss->name);
        {
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>({originTensor->batch()}));
            tensors[LogLoss->outputIndexes[0]] = (tensor);
        }

        std::unique_ptr<OpT> LossNeg(new OpT);
        LossNeg->type                               = OpType_Reduction;
        LossNeg->name                               = "LossNeg";
        LossNeg->main.type                          = OpParameter_ReductionParam;
        LossNeg->main.value                         = new ReductionParamT;
        LossNeg->main.AsReductionParam()->keepDims  = false;
        LossNeg->main.AsReductionParam()->dType     = DataType_DT_FLOAT;
        LossNeg->main.AsReductionParam()->operation = ReductionType_SUM;
        LossNeg->main.AsReductionParam()->dim       = {0};

        LossNeg->inputIndexes = {LogLoss->outputIndexes[0]};
        LossNeg->outputIndexes.emplace_back(net->tensorName.size());
        net->tensorName.emplace_back(LossNeg->name);
        {
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>({}));
            tensors[LossNeg->outputIndexes[0]] = (tensor);
        }

        std::unique_ptr<OpT> Loss(new OpT);
        Loss->type                     = OpType_UnaryOp;
        Loss->name                     = "Loss";
        Loss->main.type                = OpParameter_UnaryOp;
        Loss->main.value               = new UnaryOpT;
        Loss->main.AsUnaryOp()->T      = DataType_DT_FLOAT;
        Loss->main.AsUnaryOp()->opType = UnaryOpOperation_NEG;

        Loss->inputIndexes = {LossNeg->outputIndexes[0]};
        Loss->outputIndexes.emplace_back(net->tensorName.size());
        net->tensorName.emplace_back(Loss->name);
        {
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>({}));
            tensors[Loss->outputIndexes[0]] = (tensor);
        }
        lossOp = Loss.get();

        net->oplists.emplace_back(std::move(LossSum));
        net->oplists.emplace_back(std::move(LogLoss));
        net->oplists.emplace_back(std::move(LossNeg));
        net->oplists.emplace_back(std::move(Loss));
    }
    return lossOp;
}
