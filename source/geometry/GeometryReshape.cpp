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
        auto inputSlice = inputDes->regions;
        if (inputSlice.empty()) {
            // Create Full Refence
            Tensor::InsideDescribe::Region totalSlice = TensorUtils::makeFullSlice(input);
            inputSlice.emplace_back(std::move(totalSlice));
        }
        outputDes->regions    = std::move(inputSlice);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        return true;
    }
};
class SingleGeometryComputer : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input      = inputs[0];
        auto output     = outputs[0];
        auto inputDes   = TensorUtils::getDescribe(input);
        auto outputDes  = TensorUtils::getDescribe(output);
        auto inputSlice = inputDes->regions;
        if (inputSlice.empty()) {
            // Create Full Refence
            Tensor::InsideDescribe::Region totalSlice = TensorUtils::makeFullSlice(input);
            inputSlice.emplace_back(std::move(totalSlice));
        }
        outputDes->regions    = std::move(inputSlice);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryReshape);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Reshape});
    std::shared_ptr<GeometryComputer> _comp(new SingleGeometryComputer);
    GeometryComputer::registerGeometryComputer(_comp, {OpType_Squeeze, OpType_Unsqueeze, OpType_ExpandDims, OpType_Flatten, OpType_QuantizedReshape});
}

REGISTER_GEOMETRY(GeometryReshape, _create);
}; // namespace MNN
