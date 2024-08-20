//
//  GeometryLayerNorm.cpp
//  MNN
//
//  Created by MNN on 2020/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {

class GeometryLayerNorm : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        /* Target: Ensure reduce dimensions must be a sequence subset [-rank,...,rank-1] */
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(1 == inputs.size());
        auto layernorm          = op->main_as_LayerNorm();
        if (!layernorm->axis()) {
            std::shared_ptr<Command> cmdP(new Command);
            auto& cmd = *cmdP;
            cmd.op      = op;
            cmd.inputs  = {inputs[0]};
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmdP));
            return true;
        }
        auto reduceDims = layernorm->axis()->data();
        int reduceDimensionCount = layernorm->axis()->size();
        auto inputShape = inputs[0]->shape();
        auto outputShape = outputs[0]->shape();
        int rank = static_cast<int32_t>(inputShape.size());

        // Case1: Do not need permute
        bool needPermute = true;
        if (reduceDims[0] < 0 && reduceDims[reduceDimensionCount - 1] == -1) { // [-r,-r+1...]
            needPermute = false;
        }
        if (reduceDims[reduceDimensionCount - 1] > 0 && reduceDims[reduceDimensionCount - 1] == rank - 1 ) { // [...,r-2,r-1]
            needPermute = false;
        }
        if (reduceDims[0] == 0 && rank == 1) { // reduce dim:[0], input dimensions=1
            needPermute = false;
        }
        std::vector<int> lastdims(reduceDimensionCount);
        for (int i = 0; i < reduceDimensionCount; ++i) {
            lastdims[i] = (reduceDims[i] + rank) % rank;
        }
        if (false == needPermute) {
            std::shared_ptr<Command> cmdP(new Command);
            auto& cmd = *cmdP;
            cmd.op      = op;
            cmd.inputs  = {inputs[0]};
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmdP));
            return true;
        }
        // Case2 : Need permute
        int oldorder[MNN_MAX_TENSOR_DIM];
        int neworder[MNN_MAX_TENSOR_DIM];

        {
            int di = 0;
            int idx = 0;
            while (di < rank) {
                if (di < lastdims[0] || di > lastdims[reduceDimensionCount - 1]) {
                    neworder[idx++] = di;
                }
                di++;
            }
            for (int i = 0; i < reduceDimensionCount; ++i) {
                neworder[idx++] = lastdims[i];
            }
        }
        {
            int idx = 0;
            for (int i = 0; i < rank; ++i) {
                int j = 0;
                while (i != neworder[j]) {
                    ++j;
                }
                oldorder[idx++] = j;
            }
        }
        std::vector<int> newshape;
        for (int i = 0; i < rank; ++i) {
            newshape.emplace_back(inputShape[neworder[i]]);
        }
        
        std::shared_ptr<Tensor> outputTensorPermute(Tensor::createDevice(newshape, inputs[0]->getType(), inputs[0]->getDimensionType()));
        res.extras.emplace_back(outputTensorPermute);
        GeometryComputer::ComputePermuteRegion(inputs[0], outputTensorPermute.get(), neworder, rank);
        
        // Create LayerNorm command
        auto currentInput = outputTensorPermute.get();
        float epislon = layernorm->epsilon();
        bool  useRMS  = layernorm->useRMSNorm();
        std::vector<int64_t> externalData;
        if (layernorm->external()) {
            externalData.resize(3);
            externalData = {layernorm->external()->data()[0], layernorm->external()->data()[1], layernorm->external()->data()[2]};
        }
        std::vector<float> gamma, beta;
        if (layernorm->gamma() && layernorm->beta()) {
            int   gammaSize = layernorm->gamma()->size();
            gamma.resize(gammaSize);
            beta.resize(gammaSize);
            int   group     = layernorm->group();
            ::memcpy(gamma.data(), layernorm->gamma()->data(), gammaSize * sizeof(float));
            ::memcpy(beta.data(), layernorm->beta()->data(), gammaSize * sizeof(float));
        }
        std::shared_ptr<Tensor> inputTensorLayernorm(Tensor::createDevice(newshape, inputs[0]->getType(), inputs[0]->getDimensionType()));
        auto des = TensorUtils::getDescribe(inputTensorLayernorm.get());
        des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->regions    = {TensorUtils::makeFullSlice(currentInput)};
        res.extras.emplace_back(inputTensorLayernorm);
        std::shared_ptr<Tensor> outputTensorLayerNorm(Tensor::createDevice(newshape, inputs[0]->getType(), inputs[0]->getDimensionType()));
        res.extras.emplace_back(outputTensorLayerNorm);
        {
            auto cmd = GeometryComputerUtils::makeLayerNorm(inputTensorLayernorm.get(), outputTensorLayerNorm.get(), lastdims, epislon, gamma, beta, externalData, 1, useRMS);
            res.command.emplace_back(std::move(cmd));
        }
        
        GeometryComputer::ComputePermuteRegion(outputTensorLayerNorm.get(), outputs[0], oldorder, rank);
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryLayerNorm);
    GeometryComputer::registerGeometryComputer(comp, {OpType_LayerNorm});
}

REGISTER_GEOMETRY(GeometryLayerNorm, _create);

} // namespace MNN
