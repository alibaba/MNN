//
//  GeometryLRN.cpp
//  MNN
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {
class GeometryLRN : public GeometryComputer {
public:
    bool computeForNormalize(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             Context& context, CommandBuffer& res) const {
        auto normalize      = op->main_as_Normalize();
        auto mAcrossSpatial = normalize->acrossSpatial();
        auto mChannelShared = normalize->channelShared();
        Tensor* eps         = nullptr;
        Tensor* scale       = nullptr;
        auto cache          = context.searchConst(op);
        if (!cache.empty()) {
            eps   = cache[0].get();
            scale = cache[1].get();
        } else {
            auto mEps              = normalize->eps();
            auto epsT              = context.allocConst(op, {}, halide_type_of<float>());
            epsT->host<float>()[0] = mEps;
            eps                    = epsT.get();
            auto mScale = context.allocConst(op, {1, (int)normalize->scale()->size(), 1}, halide_type_of<float>());
            ::memcpy(mScale->host<float>(), normalize->scale()->data(), normalize->scale()->size() * sizeof(float));
            scale = mScale.get();
        }
        auto inputTensor = inputs[0];
        // Across channel
        int inside  = inputTensor->width() * inputTensor->height();
        int axis    = inputTensor->channel();
        int outside = inputTensor->batch();

        {
            // 1, axis, 1 -> outside, axis, inside
            std::shared_ptr<Tensor> broadCastScale(Tensor::createDevice<float>({outside, axis, inside}, Tensor::CAFFE));
            res.extras.emplace_back(broadCastScale);
            auto des = TensorUtils::getDescribe(broadCastScale.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions.resize(1);
            auto& reg = des->regions[0];
            reg.size[0] = outside;
            reg.size[1] = axis;
            reg.size[2] = inside;
            reg.src.offset = 0;
            reg.src.stride[0] = 0;
            reg.src.stride[1] = 1;
            reg.src.stride[2] = 0;
            reg.dst.offset = 0;
            reg.dst.stride[0] = axis * inside;
            reg.dst.stride[1] = inside;
            reg.dst.stride[2] = 1;
            reg.origin = scale;
            scale = broadCastScale.get();
        }

        // Across Spatial
        if (mAcrossSpatial) {
            inside  = 1;
            axis    = inputTensor->width() * inputTensor->height() * inputTensor->channel();
        }
        std::shared_ptr<Tensor> inputRaw(Tensor::createDevice<float>({outside, axis, inside}, Tensor::CAFFE));
        res.extras.emplace_back(inputRaw);
        std::shared_ptr<Tensor> inputRawSquare(Tensor::createDevice<float>({outside, axis, inside}, Tensor::CAFFE));
        res.extras.emplace_back(inputRawSquare);
        GeometryComputerUtils::makeRawAddressRef(inputRaw.get(), inputTensor, 0, outside * axis * inside);
        res.command.emplace_back(
            GeometryComputerUtils::makeUnary(UnaryOpOperation_SQUARE, inputRaw.get(), inputRawSquare.get()));
        std::shared_ptr<Tensor> summer(Tensor::createDevice<float>({outside, 1, inside}, Tensor::CAFFE));
        res.extras.emplace_back(summer);
        res.command.emplace_back(
            GeometryComputerUtils::makeReduce(ReductionType_SUM, inputRawSquare.get(), summer.get()));
        std::shared_ptr<Tensor> temp0(Tensor::createDevice<float>({outside, 1, inside}, Tensor::CAFFE));
        res.extras.emplace_back(temp0);
        std::shared_ptr<Tensor> temp1(Tensor::createDevice<float>({outside, 1, inside}, Tensor::CAFFE));
        res.extras.emplace_back(temp1);
        res.command.emplace_back(
            GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, summer.get(), eps, temp0.get()));
        res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_RSQRT, temp0.get(), temp1.get()));

        std::shared_ptr<Tensor> scaleFirst(Tensor::createDevice<float>({outside, axis, inside}, Tensor::CAFFE));
        res.extras.emplace_back(scaleFirst);
        {
            // Broadcast scale
            auto des = TensorUtils::getDescribe(scaleFirst.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions.resize(1);
            auto& reg = des->regions[0];
            reg.size[0] = outside;
            reg.size[1] = axis;
            reg.size[2] = inside;
            reg.src.offset = 0;
            reg.src.stride[0] = inside;
            reg.src.stride[1] = 0;
            reg.src.stride[2] = 1;
            reg.dst.offset = 0;
            reg.dst.stride[0] = axis * inside;
            reg.dst.stride[1] = inside;
            reg.dst.stride[2] = 1;
            reg.origin = temp1.get();
        }

        std::shared_ptr<Tensor> output0(Tensor::createDevice<float>({outside, axis, inside}, Tensor::CAFFE));
        res.extras.emplace_back(output0);
        std::shared_ptr<Tensor> output1(Tensor::createDevice<float>({outside, axis, inside}, Tensor::CAFFE));
        res.extras.emplace_back(output1);
        res.command.emplace_back(
            GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, inputRaw.get(), scaleFirst.get(), output0.get()));
        res.command.emplace_back(
            GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, output0.get(), scale, output1.get()));

        GeometryComputerUtils::makeRawAddressRef(outputs[0], output1.get(), 0, inside * outside * axis);
        return true;
    }
    bool computeForLRN(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       Context& context, CommandBuffer& res) const {
        auto parameter = op->main_as_LRN();
        // Across channel
        auto alpha  = parameter->alpha();
        auto beta   = parameter->beta();
        auto bias = parameter->bias();
        auto input  = inputs[0];
        int outside = input->length(0);
        int channel = input->length(1);
        int inside  = 1;
        for (int i = 2; i < input->dimensions(); ++i) {
            inside *= input->length(i);
        }
        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            std::shared_ptr<Tensor> newInput(new Tensor);
            newInput->buffer().type = input->getType();
            TensorUtils::copyShape(input, newInput.get(), true);
            TensorUtils::getDescribe(newInput.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            res.extras.emplace_back(newInput);
            GeometryComputerUtils::makeRawAddressRef(newInput.get(), input, 0, inside * outside * channel);
            input = newInput.get();
        }
        // 1. y = x^2
        std::shared_ptr<Tensor> squareInput(new Tensor);
        squareInput->buffer().type = input->getType();
        TensorUtils::copyShape(input, squareInput.get(), true);
        res.extras.emplace_back(squareInput);
        res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SQUARE, input, squareInput.get()));
        // 2. z = filter(y, 1)
        std::shared_ptr<Tensor> filterOutput(new Tensor);
        filterOutput->buffer().type = input->getType();
        TensorUtils::copyShape(input, filterOutput.get());
        res.extras.emplace_back(filterOutput);

        if (parameter->regionType() == 0) {
            // 2.1 NCHW -> N, H*W, 1, localsize /2 + C + localsize / 2
            std::shared_ptr<Tensor> squareInputTranspose(new Tensor);
            {
                auto pad                                  = parameter->localSize() / 2;
                squareInputTranspose->buffer().type       = input->getType();
                squareInputTranspose->buffer().dimensions = 4;
                squareInputTranspose->setLength(0, outside);
                squareInputTranspose->setLength(1, inside);
                squareInputTranspose->setLength(2, 1);
                squareInputTranspose->setLength(3, channel + 2 * pad);
                auto des             = TensorUtils::getDescribe(squareInputTranspose.get());
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
                des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->regions.resize(1);
                auto& reg         = des->regions[0];
                reg.origin        = squareInput.get();
                reg.size[0]       = outside;
                reg.size[1]       = inside;
                reg.size[2]       = channel;
                reg.src.offset    = 0;
                reg.src.stride[0] = inside * channel;
                reg.src.stride[1] = 1;
                reg.src.stride[2] = inside;
                reg.dst.offset    = pad;
                reg.dst.stride[0] = inside * (channel + 2 * pad);
                reg.dst.stride[1] = channel + 2 * pad;
                reg.dst.stride[2] = 1;
            }
            res.extras.emplace_back(squareInputTranspose);
            // 2.2 Filter, Use AVE pool to compute
            std::shared_ptr<Tensor> avgTensor(new Tensor);
            TensorUtils::copyShape(squareInputTranspose.get(), avgTensor.get(), true);
            avgTensor->setLength(3, channel);
            avgTensor->buffer().type = squareInputTranspose->getType();
            res.extras.emplace_back(avgTensor);
            {
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(GeometryComputerUtils::makePool(builder, std::make_pair(parameter->localSize(), 1), std::make_pair(1, 1), PoolType_AVEPOOL, PoolPadType_VALID, std::make_pair(0, 0), false));
                res.command.emplace_back(
                    GeometryComputerUtils::makeCommand(builder, {squareInputTranspose.get()}, {avgTensor.get()}));
            }
            // 2.3 N, H*W, 1, C -> NCHW
            {
                auto des             = TensorUtils::getDescribe(filterOutput.get());
                des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
                des->regions.resize(1);
                auto& reg         = des->regions[0];
                reg.origin        = avgTensor.get();
                reg.size[0]       = outside;
                reg.size[1]       = channel;
                reg.size[2]       = inside;
                reg.src.offset    = 0;
                reg.src.stride[0] = inside * channel;
                reg.src.stride[1] = 1;
                reg.src.stride[2] = channel;
                reg.dst.offset    = 0;
                reg.dst.stride[0] = inside * channel;
                reg.dst.stride[1] = inside;
                reg.dst.stride[2] = 1;
            }
        } else {
            // 2.1 NCHW -> N, C, H+localsize-1, W+localSize-1
            std::shared_ptr<Tensor> squareInputTranspose(new Tensor);
            {
                auto pad                                  = parameter->localSize() / 2;
                squareInputTranspose->buffer().type       = input->getType();
                squareInputTranspose->buffer().dimensions = 4;
                squareInputTranspose->setLength(0, outside);
                squareInputTranspose->setLength(1, channel);
                squareInputTranspose->setLength(2, input->length(2) + 2 * pad);
                squareInputTranspose->setLength(3, input->length(3) + 2 * pad);
                auto des             = TensorUtils::getDescribe(squareInputTranspose.get());
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
                des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->regions.resize(1);
                auto& reg         = des->regions[0];
                reg.origin        = squareInput.get();
                reg.size[0]       = outside * channel;
                reg.size[1]       = input->length(2);
                reg.size[2]       = input->length(3);
                reg.src.offset    = 0;
                reg.src.stride[0] = input->length(3) * input->length(2);
                reg.src.stride[1] = input->length(3);
                reg.src.stride[2] = 1;
                reg.dst.offset    = pad * squareInputTranspose->length(3) + pad;
                reg.dst.stride[0] = squareInputTranspose->length(2) * squareInputTranspose->length(3);
                reg.dst.stride[1] = squareInputTranspose->length(3);
                reg.dst.stride[2] = 1;
            }
            res.extras.emplace_back(squareInputTranspose);
            // 2.2 Filter, Use AVE pool to compute
            std::shared_ptr<Tensor> avgTensor(new Tensor);
            TensorUtils::copyShape(squareInputTranspose.get(), avgTensor.get(), true);
            avgTensor->setLength(3, input->length(3));
            avgTensor->setLength(2, input->length(2));
            avgTensor->buffer().type = squareInputTranspose->getType();
            res.extras.emplace_back(avgTensor);
            {
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(GeometryComputerUtils::makePool(builder, std::make_pair(parameter->localSize(), parameter->localSize()), std::make_pair(1, 1), PoolType_AVEPOOL, PoolPadType_VALID, std::make_pair(0, 0), false));
                res.command.emplace_back(
                    GeometryComputerUtils::makeCommand(builder, {squareInputTranspose.get()}, {avgTensor.get()}));
            }
            // 2.3 N, C4, HW, 4 -> NCHW
            {
                GeometryComputerUtils::makeRawAddressRef(filterOutput.get(), avgTensor.get(), 0,
                                                         outside * inside * channel);
            }
        }

        // 3. filter = filter * beta + alpha
        std::shared_ptr<Tensor> temp0(new Tensor);
        temp0->buffer().type = input->getType();
        std::shared_ptr<Tensor> temp1(new Tensor);
        temp1->buffer().type = input->getType();
        std::shared_ptr<Tensor> temp2(new Tensor);
        temp2->buffer().type = input->getType();
        TensorUtils::copyShape(filterOutput.get(), temp0.get(), true);
        TensorUtils::copyShape(filterOutput.get(), temp1.get(), true);
        TensorUtils::copyShape(filterOutput.get(), temp2.get(), true);
        res.extras.emplace_back(temp0);
        res.extras.emplace_back(temp1);
        res.extras.emplace_back(temp2);

        {
            Tensor* Alpha     = nullptr;
            Tensor* Beta      = nullptr;
            Tensor* Bias       = nullptr;
            auto constTensors = context.searchConst(op);
            if (!constTensors.empty()) {
                Alpha = constTensors[0].get();
                Beta  = constTensors[1].get();
                Bias  = constTensors[2].get();
            } else {
                auto t0              = context.allocConst(op, {}, halide_type_of<float>());
                auto t1              = context.allocConst(op, {}, halide_type_of<float>());
                auto t2              = context.allocConst(op, {}, halide_type_of<float>());
                t0->host<float>()[0] = alpha;
                t1->host<float>()[0] = -beta; // turn input / pow(filter, beta) -> input * pow(filter, -beta)
                t2->host<float>()[0] = bias;
                Alpha                = t0.get();
                Beta                 = t1.get();
                Bias                 = t2.get();
            }
            res.command.emplace_back(
                GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, filterOutput.get(), Alpha, temp0.get()));
            res.command.emplace_back(
                GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, temp0.get(), Bias, temp1.get()));
            res.command.emplace_back(
                GeometryComputerUtils::makeBinary(BinaryOpOperation_POW, temp1.get(), Beta, temp2.get()));
        }
        // 4. output = input * filter
        std::shared_ptr<Tensor> output(new Tensor);
        output->buffer().type = input->getType();
        TensorUtils::copyShape(input, output.get(), true);
        res.extras.emplace_back(output);

        res.command.emplace_back(
            GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, input, temp2.get(), output.get()));
        GeometryComputerUtils::makeRawAddressRef(outputs[0], output.get(), 0, outside * inside * channel);
        return true;
    }

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (op->type() == OpType_Normalize) {
            return computeForNormalize(op, inputs, outputs, context, res);
        }
        return computeForLRN(op, inputs, outputs, context, res);
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryLRN);
    GeometryComputer::registerGeometryComputer(comp, {OpType_LRN, OpType_Normalize});
}

REGISTER_GEOMETRY(GeometryLRN, _create);

} // namespace MNN
