//
//  RasterPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/11'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "RasterPlugin.hpp"
#include "Raster.cuh"
namespace MNN {
int RasterPlugin::onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, cudaStream_t stream) {
    auto rasterInfo  = mPlugin->main_as_RasterInfo();
    auto outputShape = mPlugin->outputs()->GetAs<MNNTRTPlugin::Shape>(0);
    int bytes       = 0;
    if (dataType == nvinfer1::DataType::kFLOAT){
        bytes = 4;
    }else{
        bytes = 2;
    }

    auto dest        = (uint8_t*)outputs[0];
    if (rasterInfo->extra() == MNNTRTPlugin::ExtraType_Fill) {
        size_t total = bytes;
        for (int i = 0; i < outputShape->dim()->size(); ++i) {
            total *= outputShape->dim()->data()[i];
        }
        cudaMemset(dest, 0, total);
    }
    for (int i = 0; i < rasterInfo->regions()->size(); ++i) {
        auto regionInfo = rasterInfo->regions()->GetAs<MNNTRTPlugin::Region>(i);
        Tensor::InsideDescribe::Region region;
        region.size[0]       = regionInfo->size()->data()[0];
        region.size[1]       = regionInfo->size()->data()[1];
        region.size[2]       = regionInfo->size()->data()[2];
        region.src.stride[0] = regionInfo->src()->stride()->data()[0];
        region.src.stride[1] = regionInfo->src()->stride()->data()[1];
        region.src.stride[2] = regionInfo->src()->stride()->data()[2];
        region.src.offset    = regionInfo->src()->offset();

        region.dst.stride[0] = regionInfo->dst()->stride()->data()[0];
        region.dst.stride[1] = regionInfo->dst()->stride()->data()[1];
        region.dst.stride[2] = regionInfo->dst()->stride()->data()[2];
        region.dst.offset    = regionInfo->dst()->offset();

        auto source = (const uint8_t*)inputs[regionInfo->index()] + bytes * regionInfo->src()->offset();
        auto tmpDst = dest + bytes * regionInfo->dst()->offset();
        RasterBlit(dataType, tmpDst, source, region, bytes, stream);
    }
    return 0;
}

}; // namespace MNN
