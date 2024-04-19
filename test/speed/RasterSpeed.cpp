//
//  RasterSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2020/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/Execution.hpp"
#include "core/Backend.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "MNNTestSuite.h"
using namespace MNN;
#define CHANNEL 32
#define HEIGHT 64
#define WIDTH 128
#define TIME 100
class RasterSpeed : public MNNTestCase {
public:
    void RasterTranspose_102(std::unique_ptr<Execution>& exe, std::vector<Tensor*>& inputs, std::vector<Tensor*>& outputs) {
        AUTOTIME;
        for (int i = 0; i < TIME; i++) {
            exe->onExecute(inputs, outputs);
        }
    }
    void RasterTranspose_021(std::unique_ptr<Execution>& exe, std::vector<Tensor*>& inputs, std::vector<Tensor*>& outputs) {
        AUTOTIME;
        for (int i = 0; i < TIME; i++) {
            exe->onExecute(inputs, outputs);
        }
    }
    void RasterTranspose_210(std::unique_ptr<Execution>& exe, std::vector<Tensor*>& inputs, std::vector<Tensor*>& outputs) {
        AUTOTIME;
        for (int i = 0; i < TIME; i++) {
            exe->onExecute(inputs, outputs);
        }
    }
    virtual bool run(int precision) {
        // prepare CPU backend
        ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        BackendConfig backendConfig;
        backendConfig.precision = BackendConfig::Precision_High;
        config.backendConfig = &backendConfig;
        Backend::Info compute;
        compute.type = config.type;
        compute.numThread = config.numThread;
        compute.user = config.backendConfig;
        const RuntimeCreator* runtimeCreator(MNNGetExtraRuntimeCreator(compute.type));
        std::unique_ptr<Runtime> runtime(runtimeCreator->onCreate(compute));
        std::unique_ptr<Backend> backend(runtime->onCreate());
        // build Op
        std::unique_ptr<OpT> opt(new OpT);
        opt->type = OpType_Raster;
        flatbuffers::FlatBufferBuilder builder(1024);
        builder.ForceDefaults(true);
        auto len = Op::Pack(builder, opt.get());
        builder.Finish(len);
        auto buffer = builder.GetBufferPointer();
        const Op* op = flatbuffers::GetMutableRoot<Op>(buffer);
        // build Tensors
        std::unique_ptr<Tensor> tensors[3];
        for (int i = 0; i < 3; i++) {
            tensors[i].reset(new Tensor(4, Tensor::CAFFE));
            auto tensor = tensors[i].get();
            tensor->setType(DataType_DT_FLOAT);
            tensor->setLength(0, 1);
            tensor->setLength(1, CHANNEL);
            tensor->setLength(2, HEIGHT);
            tensor->setLength(3, WIDTH);
            if (i == 1) {
                auto des = TensorUtils::getDescribe(tensor);
                des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                Tensor::InsideDescribe::Region region;
                region.origin = tensors[0].get();
                des->regions.push_back(region);
            } else {
                backend->onAcquireBuffer(tensor, Backend::STATIC);
                TensorUtils::getDescribeOrigin(tensor)->setBackend(backend.get());
            }
        }
        auto middle = tensors[1].get();
        auto& region = TensorUtils::getDescribe(middle)->regions[0];
        std::vector<Tensor*> ins = {middle}, outs = {tensors[2].get()};
        std::unique_ptr<Execution> exe(backend->onCreate(ins, outs, op));
        // transpose(1, 0, 2)
        region.size[0] = HEIGHT;
        region.size[1] = CHANNEL;
        region.size[2] = WIDTH;
        region.src.offset = 0;
        region.src.stride[0] = WIDTH;
        region.src.stride[1] = HEIGHT * WIDTH;
        region.src.stride[2] = 1;
        region.dst.offset = 0;
        region.dst.stride[0] = CHANNEL * WIDTH;
        region.dst.stride[1] = WIDTH;
        region.dst.stride[2] = 1;
        exe->onResize(ins, outs);
        RasterTranspose_102(exe, ins, outs);
        // transpose(0, 2, 1)
        region.size[0] = CHANNEL;
        region.size[1] = WIDTH;
        region.size[2] = HEIGHT;
        region.src.offset = 0;
        region.src.stride[0] = HEIGHT * WIDTH;
        region.src.stride[1] = 1;
        region.src.stride[2] = WIDTH;
        region.dst.offset = 0;
        region.dst.stride[0] = HEIGHT * WIDTH;
        region.dst.stride[1] = HEIGHT;
        region.dst.stride[2] = 1;
        exe->onResize(ins, outs);
        RasterTranspose_021(exe, ins, outs);
        // transpose(2, 1, 0)
        region.size[0] = WIDTH;
        region.size[1] = HEIGHT;
        region.size[2] = CHANNEL;
        region.src.offset = 0;
        region.src.stride[0] = 1;
        region.src.stride[1] = WIDTH;
        region.src.stride[2] = HEIGHT * WIDTH;
        region.dst.offset = 0;
        region.dst.stride[0] = HEIGHT * CHANNEL;
        region.dst.stride[1] = CHANNEL;
        region.dst.stride[2] = 1;
        exe->onResize(ins, outs);
        RasterTranspose_210(exe, ins, outs);
        return true;
    }
};
MNNTestSuiteRegister(RasterSpeed, "speed/Raster");
