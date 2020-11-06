//
//  GeometryCropAndResize.cpp
//  MNN
//
//  Created by MNN on 2020/08/5.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "ConvertUtils.hpp"

namespace MNN {
class GeometryCropAndResize : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(4 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto img       = inputs[0];
        auto boxes     = inputs[1];
        auto box_ind   = inputs[2];
        auto crop_size = inputs[3];
        auto output    = outputs[0];
        auto extrapolation = op->main_as_CropAndResize()->extrapolationValue();
        auto method = op->main_as_CropAndResize()->method();
        // resizeType of Interp : 1-NEAREST, 2-BILINEAR
        const int resizeType = method == CropAndResizeMethod_BILINEAR ? 2 : 1;

        int batch = img->length(0), ih = img->length(1), iw = img->length(2),
                  depth = img->length(3), boxNum = boxes->length(0);
        const int cropHeight = crop_size->host<uint32_t>()[0],
                  cropWidth = crop_size->host<uint32_t>()[1];

        auto des             = TensorUtils::getDescribe(output);
        des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        des->regions.clear();
        des->regions.reserve(boxNum);
        for (int i = 0; i < boxNum; i++) {
            const float y1 = boxes->host<float>()[i*4];
            const float x1 = boxes->host<float>()[i*4+1];
            const float y2 = boxes->host<float>()[i*4+2];
            const float x2 = boxes->host<float>()[i*4+3];
            const int ind = box_ind->host<uint32_t>()[i];
            const float ch = (y2 - y1) * (ih - 1), cw = (x2 - x1) * (iw - 1);
            const float yScale = ch / static_cast<float>(cropHeight - 1);
            const float xScale = cw / static_cast<float>(cropWidth - 1);
            const float yOffset = y1 * (ih - 1), xOffset = x1 * (iw - 1);
            // select croped image from images, convert it's format from NHWC to NC4HW4
            std::shared_ptr<Tensor> cropValue(new Tensor);
            {
                cropValue->buffer().type = halide_type_of<float>();
                cropValue->buffer().dimensions = 4;
                cropValue->setLength(0, 1);
                cropValue->setLength(1, depth);
                cropValue->setLength(2, ih);
                cropValue->setLength(3, iw);
                auto des             = TensorUtils::getDescribe(cropValue.get());
                des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
                des->regions.clear();
                Tensor::InsideDescribe::Region region;
                region.origin        = img;
                region.size[1]       = depth;
                region.size[2]       = ih * iw;
                region.src.offset    = ind * ih * iw * depth;
                region.dst.offset    = 0;
                region.src.stride[1] = 1;
                region.src.stride[2] = depth;
                region.dst.stride[1] = ih * iw;
                region.dst.stride[2] = 1;
                des->regions.emplace_back(std::move(region));
                res.extras.emplace_back(cropValue);
            }
            // using Interp Op deal with crop and resize for selected image
            std::shared_ptr<Tensor> resizeValue;
            {
                resizeValue.reset(Tensor::createDevice<float>({1, depth, cropHeight, cropWidth}));
                auto des             = TensorUtils::getDescribe(resizeValue.get());
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
                std::unique_ptr<OpT> interp(new OpT);
                interp->type                          = OpType_Interp;
                interp->main.type                     = OpParameter_Interp;
                interp->main.value                    = new InterpT;
                interp->main.AsInterp()->widthScale   = xScale;
                interp->main.AsInterp()->heightScale  = yScale;
                interp->main.AsInterp()->widthOffset  = xOffset;
                interp->main.AsInterp()->heightOffset = yOffset;
                interp->main.AsInterp()->alignCorners = false;
                interp->main.AsInterp()->resizeType   = resizeType;
                auto cmd = GeometryComputerUtils::makeCommand(interp.get(), {cropValue.get()}, {resizeValue.get()});
                res.extras.emplace_back(resizeValue);
                res.command.emplace_back(cmd);
            }
            // convert resize image's format from NC4HW4 to NHWC, add it to output's batch
            {
                Tensor::InsideDescribe::Region region;
                region.origin        = resizeValue.get();
                region.size[1]       = cropHeight * cropWidth;
                region.size[2]       = depth;
                region.src.offset    = 0;
                region.dst.offset    = i * cropHeight * cropWidth * depth;
                region.src.stride[1] = 1;
                region.src.stride[2] = cropHeight * cropWidth;
                region.dst.stride[1] = depth;
                region.dst.stride[2] = 1;
                des->regions.emplace_back(std::move(region));
            }
        }

        return true;
    }
    virtual std::vector<bool> onGetOutputVirtual(const Op* op, const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) const override {
        //return {false};
        return {true};
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryCropAndResize);
    // GeometryComputer::registerGeometryComputer(comp, {OpType_CropAndResize});
}

REGISTER_GEOMETRY(GeometryCropAndResize, _create);

} // namespace MNN
