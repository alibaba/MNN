//
//  VulkanTensor.cpp
//  MNN
//
//  Created by MNN on 2020/03/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanTensor.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
std::array<int, 4> VulkanTensor::tensorShapeFormat(const Tensor *input) {    
    int iN = (0 != input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
    int iC = (0 != input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
    int iH = (0 != input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
    int iW = (0 != input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;

    if(input->buffer().dimensions > 4)//more than 4 dimensions put to N dimension
    {
        for(int i = 4; i < input->buffer().dimensions; i++)
        {
            iW *= input->buffer().dim[i].extent;
        }
    }
    
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        iN = (0 < input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
        iH = (0 < input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
        iW = (0 < input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
        iC = (0 < input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;
        if (input->dimensions() == 3) {
            // Use C instead of W for last Dim
            iC = input->buffer().dim[2].extent;
            iW = 1;
        }
        if(input->buffer().dimensions > 4)//more than 4 dimensions put to N dimension
        {
            for(int i = 4; i < input->buffer().dimensions; i++)
            {
                iC *= input->buffer().dim[i].extent;
            }
        }
    }
    if (input->buffer().dimensions == 2) {
        iN = input->buffer().dim[0].extent;
        iH = 1;
        iW = 1;
        iC = input->buffer().dim[1].extent;
    }
    if (input->buffer().dimensions == 1) {
        iN = 1;
        iH = 1;
        iW = 1;
        iC = input->buffer().dim[0].extent;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("tensorShapeFormat : [%d, %d, %d, %d] \n", iN, iH, iW, iC);
#endif
    return {iN, iH, iW, iC};
}
int VulkanTensor::getAlignSize(const Tensor* tensor) {
    auto format      = TensorUtils::getDescribe(tensor)->dimensionFormat;
    auto elementSize = tensor->elementSize();
    return ALIGN_UP4(elementSize);
}


VulkanTensor::VulkanTensor(const Tensor* shape, VkFormat format, const VulkanMemoryPool& pool, const VkPhysicalDeviceLimits& limits, bool separate) {
    auto nhwc = tensorShapeFormat(shape);
    auto width = UP_DIV(nhwc[3], 4) * nhwc[2];
    auto height = nhwc[0] * nhwc[1];
    int unit = limits.maxImageDimension2D;
    mBlocks[0] = UP_DIV(width, unit);
    mBlocks[1] = UP_DIV(height, unit);
    mSize = std::move(nhwc);
    mImage.resize(mBlocks[0] * mBlocks[1]);
    for (int y=0; y<mBlocks[1]; ++y) {
        auto ySta = y * unit;
        auto yFin = std::min(height, ySta + unit);
        auto hReal = yFin - ySta;
        for (int x=0; x<mBlocks[0]; ++x) {
            auto xSta = x * unit;
            auto xFin = std::min(width, xSta + unit);
            auto wReal = xFin - xSta;
            mImage[y*mBlocks[0] + x] = std::make_shared<VulkanImage>(pool, separate, std::vector<int>{wReal, hReal}, format);
        }
    }
}
void VulkanTensor::release() {
    for (auto img : mImage) {
        img->release();
    }
}
}
