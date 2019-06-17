//
//  backwardTest.cpp
//  MNN
//
//  Created by MNN on 2019/06/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <fstream>
#include "Interpreter.hpp"
#include "OpConverter.hpp"
#include "OpGrad.hpp"
#include "TensorUtils.hpp"
using namespace std;
using namespace MNN;

int main(int argc, const char* argv[]) {
    std::vector<int> inputSize  = {5, 1, 5, 5};
    std::vector<int> kernelSize = {6, 1, 1, 1};
    std::unique_ptr<NetT> net(new NetT);
    std::unique_ptr<OpT> inputOp(new OpT);
    inputOp->type                    = OpType_Input;
    inputOp->name                    = "input";
    inputOp->outputIndexes           = {0};
    inputOp->main.value              = new InputT;
    inputOp->main.type               = OpParameter_Input;
    inputOp->main.AsInput()->dims    = inputSize;
    inputOp->main.AsInput()->dtype   = DataType_DT_FLOAT;
    inputOp->main.AsInput()->dformat = MNN_DATA_FORMAT_NC4HW4;
    net->tensorName.emplace_back(inputOp->name);
    net->oplists.emplace_back(std::move(inputOp));

    std::unique_ptr<OpT> weight(new OpT);
    weight->name                      = "weight";
    weight->type                      = OpType_Const;
    weight->outputIndexes             = {1};
    weight->main.type                 = OpParameter_Blob;
    weight->main.value                = new BlobT;
    weight->main.AsBlob()->dims       = kernelSize;
    weight->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NHWC;
    {
        int kernelCount = 1;
        for (int i = 0; i < kernelSize.size(); ++i) {
            kernelCount *= kernelSize[i];
        }
        weight->main.AsBlob()->float32s.resize(kernelCount);
        for (int i = 0; i < kernelCount; ++i) {
            weight->main.AsBlob()->float32s[i] = 1.0f;
        }
    }
    net->tensorName.emplace_back(weight->name);
    net->oplists.emplace_back(std::move(weight));

    std::unique_ptr<OpT> bias(new OpT);
    bias->name                = "bias";
    bias->type                = OpType_Const;
    bias->outputIndexes       = {2};
    bias->main.type           = OpParameter_Blob;
    bias->main.value          = new BlobT;
    bias->main.AsBlob()->dims = {kernelSize[0]};
    {
        bias->main.AsBlob()->float32s.resize(kernelSize[0]);
        for (int i = 0; i < kernelSize[0]; ++i) {
            bias->main.AsBlob()->float32s[i] = 0.0f;
        }
    }
    net->tensorName.emplace_back(bias->name);
    net->oplists.emplace_back(std::move(bias));

    std::unique_ptr<OpT> conv(new OpT);
    conv->name          = "conv";
    conv->type          = OpType_Convolution;
    conv->outputIndexes = {3};
    conv->inputIndexes  = {0, 1, 2};
    conv->main.type     = OpParameter_Convolution2D;
    {
        conv->main.value = new Convolution2DT;
        auto conv2D      = conv->main.AsConvolution2D();
        conv2D->common.reset(new Convolution2DCommonT);
        auto common         = conv2D->common.get();
        common->kernelX     = kernelSize[2];
        common->kernelY     = kernelSize[1];
        common->padMode     = PadMode_VALID;
        common->outputCount = kernelSize[0];
        common->inputCount  = kernelSize[3];
    }
    net->tensorName.emplace_back(conv->name);
    auto convOp = conv.get();
    net->oplists.emplace_back(std::move(conv));

    std::unique_ptr<OpT> mul(new OpT);
    mul->name          = "mul";
    mul->inputIndexes  = {3, 3};
    mul->outputIndexes = {4};
    {
        mul->type                      = OpType_BinaryOp;
        mul->main.value                = new BinaryOpT;
        mul->main.type                 = OpParameter_BinaryOp;
        mul->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
        mul->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
    }
    net->tensorName.emplace_back(mul->name);
    net->oplists.emplace_back(std::move(mul));

    // Grad
    {
        std::map<int, std::vector<int> > backwardTensors;
        backwardTensors.insert(std::make_pair(3, std::vector<int>{4}));
        auto creator = OpGrad::get(OpType_Convolution);
        std::unique_ptr<OpGrad> grad(creator->onCreate(convOp, {}, {}));
        grad->onGradCommon(net.get(), convOp, backwardTensors);
    }

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = Net::Pack(builder, net.get());
    builder.Finish(offset);

    std::unique_ptr<Interpreter> interpre(Interpreter::createFromBuffer(builder.GetBufferPointer(), builder.GetSize()));
    ScheduleConfig config;
    auto session = interpre->createSession(config);
    auto input   = interpre->getSessionInput(session, nullptr);

    auto elesize = input->elementSize();
    for (int v = 0; v < elesize; ++v) {
        input->host<float>()[v] = 0.03f;
    }
    TensorCallBack begin = [](const std::vector<Tensor*>& inputs, const std::string& oname) {
        std::string name = oname;
        for (int i = 0; i < name.size(); ++i) {
            if (name[i] == '/') {
                name[i] = '_';
            }
        }
        for (int index = 0; index < inputs.size(); ++index) {
            if (inputs[index]->getType().code != halide_type_float) {
                continue;
            }
            auto origin0 = inputs[index]->host<float>();
            std::ofstream prob("output/" + name + "_input_" + numberToString(index));
            auto size = inputs[index]->elementSize();
            for (int i = 0; i < size; ++i) {
                prob << origin0[i] << "\n";
            }
        }
        return true;
    };
    TensorCallBack after = [](const std::vector<Tensor*>& output, const std::string& oname) {
        std::string name = oname;
        for (int i = 0; i < name.size(); ++i) {
            if (name[i] == '/') {
                name[i] = '_';
            }
        }
        float maxValue = 0.0f;
        for (int index = 0; index < output.size(); ++index) {
            if (output[index]->getType().code != halide_type_float) {
                continue;
            }
            std::ofstream prob("output/" + name + "_" + numberToString(index));
            auto origin0 = output[index]->host<float>();
            auto size    = output[index]->elementSize();
            for (int i = 0; i < size; ++i) {
                maxValue = std::max(maxValue, fabsf(origin0[i]));
                prob << origin0[i] << "\n";
            }
        }
        if (maxValue > 10000.0f) {
            MNN_PRINT("Invalid value : %f, %s\n", maxValue, oname.c_str());
        }
        return true;
    };
    interpre->runSessionWithCallBack(session, begin, after);

    return 0;
}
