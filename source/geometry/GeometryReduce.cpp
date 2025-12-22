//
//  GeometryReduce.cpp
//  MNN
//
//  Created by MNN on 2020/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
static std::vector<std::tuple<int, int, int>> _computeReduceDims(const std::vector<Tensor*>& inputs,
                                                                               std::vector<int>& axises) {

   auto totalSize = TensorUtils::getRawSize(inputs[0]);
   if (axises.empty()) {
       return {std::make_tuple(1, totalSize, 1)};
   }
   for (int i = 0; i < axises.size(); ++i) {
       if (axises[i] < 0) {
           if (axises[i] < 0) {
               return {std::make_tuple(1, totalSize, 1)};
           }
       }
   }
   // Cache for input's dims
   std::vector<int> lengths(inputs[0]->dimensions());
   for (int i = 0; i < lengths.size(); ++i) {
       lengths[i] = inputs[0]->length(i);
   }
   std::vector<std::pair<int, int>> groupAxises;
   {
       // Merge adj axis
       std::sort(axises.begin(), axises.end());
       int lastAxis = axises[0];
       int length   = 1;
       int start    = axises[0];
       for (int i = 1; i < axises.size(); ++i) {
           // MNN_PRINT("%d - %d\n", axises[i], lastAxis);
           if (axises[i] - lastAxis == 1) {
               length++;
           } else {
               groupAxises.emplace_back(std::make_pair(start, length));
               length = 1;
               start  = axises[i];
           }
           lastAxis = axises[i];
       }
       groupAxises.emplace_back(std::make_pair(start, length));
   }

   // Compute inside-outside-axis
   std::vector<std::tuple<int, int, int>> result;

   for (int i = 0; i < groupAxises.size(); ++i) {
       int outsideSize = 1;
       int insideSize  = 1;
       int axisSize    = 1;
       auto start      = groupAxises[i].first;
       auto length     = groupAxises[i].second;
       if (start >= (int)lengths.size()) {
           break;
       }
       for (int j = 0; j < start; ++j) {
           outsideSize *= lengths[j];
       }
       for (int j = start; j < start + length; ++j) {
           if (j >= (int)lengths.size()) {
               break;
           }
           axisSize *= lengths[j];
           lengths[j] = 1;
       }
       for (int j = start + length; j < lengths.size(); ++j) {
           insideSize *= lengths[j];
       }
       if (1 == axisSize) {
           continue;
       }
       result.emplace_back(std::make_tuple(outsideSize, axisSize, insideSize));
   }
   // FUNC_PRINT(result.size());
   if (result.empty()) {
       result.emplace_back(std::make_tuple(1, 1, totalSize));
   }
   return result;
}

class GeometryReduce : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(inputs.size() >= 1);
        auto reduct          = op->main_as_ReductionParam();
        auto reductOp        = reduct->operation();
        std::vector<int> axises;
        if (inputs.size() >= 2) {
            auto size = inputs[1]->elementSize();
            auto dims = inputs[1]->host<int32_t>();
            for (int i = 0; i < size; ++i) {
                axises.emplace_back(dims[i]);
            }
        } else {
            auto reduct = op->main_as_ReductionParam();
            if (nullptr != reduct->dim()) {
                for (int i = 0; i < reduct->dim()->size(); ++i) {
                    axises.emplace_back(reduct->dim()->data()[i]);
                }
            }
        }
        for (int i = 0; i < axises.size(); ++i) {
            if (axises[i] < 0) {
                axises[i] = inputs[0]->dimensions() + axises[i];
            }
        }
        if (1 == axises.size() && TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 && TensorUtils::getDescribe(outputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            auto cmd = GeometryComputerUtils::makeReduce(reductOp, inputs[0], outputs[0], axises[0]);
            res.command.emplace_back(std::move(cmd));
            return true;
        }
        // prod([]) = 1
        if (inputs[0]->elementSize() == 0) {
            if(!context.allocTensor(outputs[0])) {
                return false;
            }
            float res;
            switch (reductOp) {
                case ReductionType_PROD:
                    res = 1.0f;
                    break;
                default:
                    res = 0.0f;
                    break;
            }
            if (outputs[0]->getType() == halide_type_of<float>()) {
                outputs[0]->host<float>()[0] = (float)res;
            } else {
                outputs[0]->host<int>()[0] = (int)res;
            }
            return true;
        }
        auto reduceDims = _computeReduceDims(inputs, axises);
        Tensor* currentInput = inputs[0];
        MNN_ASSERT(reduceDims.size() > 0);
        auto dimType = currentInput->getDimensionType();
        for (int i = 0; i < reduceDims.size(); ++i) {
            auto& iter   = reduceDims[i];
            auto inside  = std::get<2>(iter);
            auto outside = std::get<0>(iter);
            auto axis    = std::get<1>(iter);
            
            std::shared_ptr<Tensor> inputTensor(
                Tensor::createDevice({outside, axis, inside}, inputs[0]->getType(), dimType));
            auto des        = TensorUtils::getDescribe(inputTensor.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions    = {TensorUtils::makeFullSlice(currentInput)};
            res.extras.emplace_back(inputTensor);
            std::shared_ptr<Tensor> outputTensor(
                Tensor::createDevice({outside, 1, inside}, inputs[0]->getType(), dimType));
            res.extras.emplace_back(outputTensor);

            // Create Command
            {
                auto cmd = GeometryComputerUtils::makeReduce(reductOp, inputTensor.get(), outputTensor.get());
                res.command.emplace_back(std::move(cmd));
            }
            currentInput = outputTensor.get();
            // Ref output
            if (i == reduceDims.size() - 1) {
                auto outputDes        = TensorUtils::getDescribe(outputs[0]);
                outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                outputDes->regions    = {TensorUtils::makeFullSlice(outputTensor.get())};
            }
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryReduce);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Reduction});
}

REGISTER_GEOMETRY(GeometryReduce, _create);

} // namespace MNN
