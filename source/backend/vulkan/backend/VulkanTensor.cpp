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
int VulkanTensor::getAlignSize(const Tensor* tensor) {
    auto format      = TensorUtils::getDescribe(tensor)->dimensionFormat;
    auto elementSize = tensor->elementSize();
    return ALIGN_UP4(elementSize);
}

VulkanTensor::VulkanTensor(const Tensor* shape, const VulkanMemoryPool& pool, const VkPhysicalDeviceLimits& limits, bool seperate) {
    auto dims = shape->dimensions();
    int width = 1, height = 1, depth = 1;
    switch (dims) {
        case 0:
            break;
        case 1:
            depth = UP_DIV(shape->length(0), 4);
            break;
        case 2:
            depth = UP_DIV(shape->length(1), 4);
            width = shape->length(0);
            break;
        default:
        {
            depth = UP_DIV(shape->length(1), 4) * shape->length(0);
            height = shape->length(2);
            for (int i=3; i < dims; ++i) {
                width *= shape->length(i);
            }
        }
            break;
    }
    mBlocks.resize(3);
    int wUnit = limits.maxImageDimension3D;
    int hUnit = limits.maxImageDimension3D;
    int cUnit = limits.maxImageDimension3D;
    mBlocks[0] = UP_DIV(width, wUnit);
    mBlocks[1] = UP_DIV(height, hUnit);
    mBlocks[2] = UP_DIV(depth, cUnit);
    
    mImage.resize(mBlocks[0] * mBlocks[1] * mBlocks[2]);
    for (int z=0; z<mBlocks[2]; ++z) {
        auto zSta = z * cUnit;
        auto zFin = std::min(depth, zSta + cUnit);
        auto depthReal = zFin  - zSta;
        for (int y=0; y<mBlocks[1]; ++y) {
            auto ySta = y * hUnit;
            auto yFin = std::min(height, ySta + hUnit);
            auto hReal = yFin - ySta;
            for (int x=0; x<mBlocks[0]; ++x) {
                auto xSta = x * wUnit;
                auto xFin = std::min(width, xSta + wUnit);
                auto wReal = xFin - xSta;
                mImage[z*mBlocks[1]*mBlocks[0] + y*mBlocks[0] + x] = std::make_shared<VulkanImage>(pool, seperate, std::vector<int>{wReal, hReal, depthReal}, shape->getType());
            }
        }
    }
}
void VulkanTensor::release() {
    for (auto img : mImage) {
        img->release();
    }
}
}
