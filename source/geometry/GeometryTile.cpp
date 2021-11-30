//
//  GeometryTile.cpp
//  MNN
//
//  Created by MNN on 2020/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryTile : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        auto multiples = inputs[1];
        auto output    = outputs[0];
        auto input     = inputs[0];

        // Compute Remain size and stride because region can only support up to 3
        int remainSize = 1;
        std::vector<int> remainDims;
        for (int i = 0; i < input->dimensions() - 3; ++i) {
            remainSize *= input->length(i);
            remainDims.emplace_back(input->length(i));
        }
        std::vector<int32_t> mod(remainDims.size());
        OpCommonUtils::computeStride(mod.data(), remainDims.data(), remainDims.size());

        // Compute Multiply Stride
        auto mulPtr   = multiples->host<int32_t>();
        int copyTimes = 1;
        for (int i = 0; i < input->dimensions(); ++i) {
            copyTimes *= mulPtr[i];
        }
        auto modMultiSize = input->dimensions();
        int32_t modMulti[MNN_MAX_TENSOR_DIM];
        for (int i = 0; i < modMultiSize; ++i) {
            int value = 1;
            for (int j = i + 1; j < input->dimensions(); ++j) {
                value *= mulPtr[j];
            }
            modMulti[i] = value;
        }

        // Compute input and output stride
        // input stride use for remainSize split
        // output stride use for remainSize split and tile split
        std::vector<int> inputStrides(input->dimensions());
        std::vector<int> outputStrides(input->dimensions());
        {
            int strides    = 1;
            int outStrides = 1;
            for (int i = input->dimensions() - 1; i >= 0; --i) {
                inputStrides[i] = strides;
                strides *= input->length(i);
                outputStrides[i] = outStrides;
                outStrides *= output->length(i);
            }
        }
        // Compute regions, first iter copyTimes, second iter remainSize
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->regions.resize(copyTimes * remainSize);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        int coordinates[MNN_MAX_TENSOR_DIM];
        for (int u = 0; u < copyTimes; ++u) {
            int dstOffset = 0;
            OpCommonUtils::unravelIndexHelper(coordinates, modMulti, modMultiSize, u);
            for (int i = 0; i < modMultiSize; ++i) {
                dstOffset += coordinates[i] * input->length(i) * outputStrides[i];
            }
            for (int v = 0; v < remainSize; ++v) {
                auto& region      = outputDes->regions[u * remainSize + v];
                region.src.offset = 0;
                region.origin     = input;
                auto value        = v;
                region.dst.offset = dstOffset;
                for (int i = 0; i < 3; ++i) {
                    auto match = input->dimensions() - i - 1;
                    if (match < 0) {
                        continue;
                    }
                    region.size[3 - i - 1]       = input->length(match);
                    region.src.stride[3 - i - 1] = inputStrides[match];
                    region.dst.stride[3 - i - 1] = outputStrides[match];
                }
                for (int i = 0; i < remainDims.size(); ++i) {
                    auto coordinate = value / mod[i];
                    region.src.offset += coordinate * inputStrides[i];
                    region.dst.offset += coordinate * outputStrides[i];
                    value = value % mod[i];
                }
            }
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryTile);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Tile});
}

REGISTER_GEOMETRY(GeometryTile, _create);

} // namespace MNN
