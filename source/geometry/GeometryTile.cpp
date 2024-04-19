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
/**
 Status 0 : input = 1 and multi = 1
 Status 1 ：multi > 1
 Status 2 ：input > 1

 Input = 1 , multi = 1 : No change
 Input = 1 , multi > 1 :
 - Status 0 / 1 : multi * prevmulti，set status 1
 - Status 2 : Export Input，set status 1
 Input > 1 , multi = 1 ：
 - Status 0 / 2 ：input * previnput，set status  2
 - Status 1 ：Export multi，set status 2
 Input > 1 , multi > 1 ：
 - Status 0 ：Export multi and input，Set status 0
 - Status 1 ：multi * prevmulti，Export multi and input, set status  0
 - Status 2 ：Export prevInput，Export mult, Export input，set status  0
 */

class GeometryTile : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        auto multiples = inputs[1];
        auto output    = outputs[0];
        auto input     = inputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outputDes->regions.clear();

        int validLength = 0;
        int status = 0;
        int inputStrides[MNN_MAX_TENSOR_DIM];
        int outputStrides[MNN_MAX_TENSOR_DIM];
        {
            int shapes[MNN_MAX_TENSOR_DIM];
            for (int i = 0; i < input->dimensions(); ++i) {
                shapes[i] = input->length(i);
            }
            OpCommonUtils::computeStride(inputStrides, shapes, input->dimensions());
            for (int i = 0; i < output->dimensions(); ++i) {
                shapes[i] = output->length(i);
            }
            OpCommonUtils::computeStride(outputStrides, shapes, input->dimensions());
        }
        
        int size[MNN_MAX_TENSOR_DIM];
        int srcStride[MNN_MAX_TENSOR_DIM];
        int dstStride[MNN_MAX_TENSOR_DIM];
        int prevInput = 1;
        int prevMulti = 1;
        int prevIndex = 0;
        auto mulPtr   = multiples->host<int32_t>();
        for (int i = 0; i < input->dimensions(); ++i) {
            auto il = input->length(i);
            auto ml = mulPtr[i];
            if (il == 0 || ml == 0) {
                // Zero shape
                return true;
            }
            if (il == 1 && ml == 1) {
                continue;
            }
            if (il == 1 && ml > 1) {
                switch (status) {
                    case 0:
                        prevMulti = 1;
                    case 1:
                        prevMulti = prevMulti * ml;
                        prevIndex = i;
                        break;
                    case 2:
                        size[validLength] = prevInput;
                        srcStride[validLength] = inputStrides[prevIndex];
                        dstStride[validLength] = outputStrides[prevIndex];
                        validLength++;
                        prevIndex = i;
                        prevMulti = ml;
                        break;
                    default:
                        break;
                }
                status = 1;
                continue;
            }
            if (il > 1 && ml == 1) {
                switch (status) {
                    case 0:
                        prevInput = 1;
                    case 2:
                        prevInput = prevInput * il;
                        prevIndex = i;
                        break;
                    case 1:
                        size[validLength] = prevMulti;
                        srcStride[validLength] = 0;
                        dstStride[validLength] = input->length(prevIndex) * outputStrides[prevIndex];
                        validLength++;
                        prevIndex = i;
                        prevInput = il;
                        break;
                    default:
                        break;
                }
                status = 2;
                continue;
            }
            // il > 1 and ml > 1
            if (1 == status) {
                ml = ml * prevMulti;
            } else if (2 == status) {
                size[validLength] = prevInput;
                srcStride[validLength] = inputStrides[prevIndex];
                dstStride[validLength] = outputStrides[prevIndex];
                validLength++;
            }
            size[validLength] = ml;
            srcStride[validLength] = 0;
            dstStride[validLength] = il * outputStrides[i];
            validLength++;
            size[validLength] = il;
            srcStride[validLength] = inputStrides[i];
            dstStride[validLength] = outputStrides[i];
            validLength++;
            status = 0;
        }
        // Check remain input length / multi
        switch (status) {
            case 1:
                size[validLength] = prevMulti;
                srcStride[validLength] = 0;
                dstStride[validLength] = input->length(prevIndex) * outputStrides[prevIndex];
                validLength++;
                break;
            case 2:
                size[validLength] = prevInput;
                srcStride[validLength] = inputStrides[prevIndex];
                dstStride[validLength] = outputStrides[prevIndex];
                validLength++;
                break;
            default:
                break;
        }
        
        // Pad to 3 if not larger than 3
        for (int i=validLength; i<3; ++i) {
            size[i] = 1;
            srcStride[i] = 0;
            dstStride[i] = 0;
        }
        validLength = ALIMAX(validLength, 3);

        // Compute Remain size and stride because region can only support up to 3
        int remainSize = 1;
        int remainDims[MNN_MAX_TENSOR_DIM];
        int remainDimSize = validLength - 3;
        for (int i = 0; i < validLength - 3; ++i) {
            remainSize *= size[i];
            remainDims[i] = size[i];
        }
        int mod[MNN_MAX_TENSOR_DIM];
        OpCommonUtils::computeStride(mod, remainDims, remainDimSize);
        outputDes->regions.reserve(remainSize);
        int coordinates[MNN_MAX_TENSOR_DIM];
        for (int u = 0; u < remainSize; ++u) {
            OpCommonUtils::unravelIndexHelper(coordinates, mod, remainDimSize, u);
            Tensor::InsideDescribe::Region region;
            region.origin     = input;
            region.src.offset = 0;
            region.dst.offset = 0;
            for (int v=0; v<remainDimSize; ++v) {
                region.src.offset += srcStride[v] * coordinates[v];
                region.dst.offset += dstStride[v] * coordinates[v];
            }
            for (int v=0; v<3; ++v) {
                auto ov = v + remainDimSize;
                region.src.stride[v] = srcStride[ov];
                region.dst.stride[v] = dstStride[ov];
                region.size[v] = size[ov];
            }
            outputDes->regions.emplace_back(std::move(region));
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
