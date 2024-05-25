//
//  GeometryReshape.cpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
namespace MNN {
class GeometryReshape : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input     = inputs[0];
        auto output    = outputs[0];
        auto inputDes  = TensorUtils::getDescribe(input);
        auto outputDes = TensorUtils::getDescribe(output);
        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            auto midFormat = op->main_as_Reshape()->dimType();
            if (MNN_DATA_FORMAT_NHWC == midFormat) {
                // Convert to NHWC, reshape, and then convert to NC4HW4
                std::shared_ptr<Tensor> nhwc(new Tensor);
                TensorUtils::setupTensorInfo(input, nhwc.get(), MNN_DATA_FORMAT_NHWC);
                ConvertUtils::compute(input, nhwc.get(), res);
                res.extras.emplace_back(nhwc);
                std::shared_ptr<Tensor> nhwc2(new Tensor);
                TensorUtils::setupTensorInfo(output, nhwc2.get(), MNN_DATA_FORMAT_NHWC);
                res.extras.emplace_back(nhwc2);
                {
                    auto inputSlice = TensorUtils::getDescribe(nhwc.get())->regions;
                    if (inputSlice.empty()) {
                        // Create Full Refence
                        Tensor::InsideDescribe::Region totalSlice = TensorUtils::makeFullSlice(nhwc.get());
                        inputSlice.emplace_back(std::move(totalSlice));
                    }
                    TensorUtils::getDescribe(nhwc2.get())->regions    = std::move(inputSlice);
                    TensorUtils::getDescribe(nhwc2.get())->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                }
                ConvertUtils::compute(nhwc2.get(), output, res);
                return true;
            }
        }
        outputDes->regions = {TensorUtils::makeFullSlice(input)};
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        return true;
    }
};
class SingleGeometryComputer : public GeometryComputer {
public:
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             Context& context, CommandBuffer& cmd) const override {
        auto inputDes = TensorUtils::getDescribe(inputs[0]);
        auto des = TensorUtils::getDescribe(outputs[0]);
        if (des->regions.size() != 1 || inputDes->regions.size() > 0) {
            return false;
        }
        des->regions[0].origin = inputs[0];
        des->regions[0].size[0] = 1;
        des->regions[0].size[1] = 1;
        des->regions[0].size[2] = 1;
        for (int i = 0; i < inputs[0]->dimensions(); ++i) {
            des->regions[0].size[2] *= inputs[0]->length(i);
        }
        des->regions[0].src.stride[2] = 1;
        des->regions[0].dst.stride[2] = 1;
        des->regions[0].src.offset = 0;
        des->regions[0].dst.offset = 0;
        des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        return true;
    }

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input      = inputs[0];
        auto output     = outputs[0];
        auto inputDes   = TensorUtils::getDescribe(input);
        auto outputDes  = TensorUtils::getDescribe(output);
        outputDes->regions = {TensorUtils::makeFullSlice(input)};
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        return true;
    }
};
class CopyGeometryComputer : public GeometryComputer {
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        for (int v=0; v<inputs.size(); ++v) {
            auto input      = inputs[v];
            auto output     = outputs[v];
            auto inputDes   = TensorUtils::getDescribe(input);
            auto outputDes  = TensorUtils::getDescribe(output);
            if (inputDes->tensorArrayAttr != nullptr) {
                outputDes->tensorArrayAttr = inputDes->tensorArrayAttr;
                return true;
            }
            outputDes->regions = {TensorUtils::makeFullSlice(input)};
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryReshape);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Reshape});
    std::shared_ptr<GeometryComputer> _comp(new SingleGeometryComputer);
    GeometryComputer::registerGeometryComputer(_comp, {OpType_Squeeze, OpType_Unsqueeze, OpType_ExpandDims, OpType_Flatten, OpType_QuantizedReshape});
    std::shared_ptr<GeometryComputer> copycomp(new CopyGeometryComputer);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Identity});
}

REGISTER_GEOMETRY(GeometryReshape, _create);
}; // namespace MNN
