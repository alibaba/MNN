//
//  CoreMLRaster.metal
//  MNN
//
//  Created by MNN on 2021/04/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <metal_stdlib>
using namespace metal;

struct SamplerInfo {
    uint4 stride;   //stride[3] + offset
    uint4 size;     //size[3] + totalSize
    uint4 extent;   //dstStride[3]+dstOffset
    uint4 imageSize;
};

kernel void raster_texture(texture2d_array<half, access::read> in   [[texture(0)]],
                           texture2d_array<half, access::write> out [[texture(1)]],
                           constant SamplerInfo &info               [[buffer(0)]],
                           uint3 gid                                [[thread_position_in_grid]]) {
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        // out[int(dstOffset)] = in[int(srcOffset)];
        // do raster on texture
    }
}

kernel void raster(const device int *in         [[buffer(0)]],
                   device int *out              [[buffer(1)]],
                   constant SamplerInfo &info   [[buffer(2)]],
                   uint3 gid                    [[thread_position_in_grid]]) {
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        out[int(dstOffset)] = in[int(srcOffset)];
    }
}
