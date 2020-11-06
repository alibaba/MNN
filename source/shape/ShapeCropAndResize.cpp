//
//  ShapeCropAndResize.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class CropAndResizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // The shape of 'image' is [batch_size, image_height, image_width,
        // channels].
        const Tensor* image = inputs[0];
        // The shape of 'boxes' is [num_boxes, 4].
        const Tensor* boxes = inputs[1];
        // The shape of 'box_index' is [num_boxes].
        const Tensor* box_index = inputs[2];
        // The shape of 'crop_size' is [2].
        Tensor* crop_size = inputs[3];

        MNN_ASSERT(4 == image->buffer().dimensions);

        const int image_height = image->buffer().dim[1].extent;
        const int image_width  = image->buffer().dim[2].extent;
        const int depth        = image->buffer().dim[3].extent;

        MNN_ASSERT(image_height > 0 && image_width > 0);
        MNN_ASSERT(1 == crop_size->buffer().dimensions && 2 == crop_size->buffer().dim[0].extent);

        int num_boxes = 0;
        if (boxes->length(0) == 0 && box_index->length(0) == 0) {
            num_boxes = 0;
        } else {
            num_boxes = boxes->buffer().dim[0].extent;
        }

        MNN_ASSERT(4 == boxes->buffer().dim[1].extent && 1 == box_index->buffer().dimensions &&
                   num_boxes == box_index->buffer().dim[0].extent);

        auto crop_size_vec = crop_size->host<int32_t>();

        const int32_t crop_height = crop_size_vec[0];
        const int32_t crop_width  = crop_size_vec[1];
        MNN_ASSERT(crop_height > 0 && crop_width > 0);

        outputs[0]->buffer().dimensions    = 4;
        outputs[0]->buffer().dim[0].extent = num_boxes;
        outputs[0]->buffer().dim[1].extent = crop_height;
        outputs[0]->buffer().dim[2].extent = crop_width;
        outputs[0]->buffer().dim[3].extent = depth;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        outputs[0]->buffer().type = inputs[0]->getType();

        return true;
    }
};

REGISTER_SHAPE_INPUTS(CropAndResizeComputer, OpType_CropAndResize, {3});
} // namespace MNN
