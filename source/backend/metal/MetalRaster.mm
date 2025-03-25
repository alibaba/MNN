//
//  MetalRaster.mm
//  MNN
//
//  Created by MNN on 2020/05/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalRaster.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

struct SamplerInfo {
    unsigned int stride[4];//stride[3] + offset
    unsigned int size[4];//size[3] + totalSize
    unsigned int extent[4];//dstStride[3]+dstOffset
};

static void writeSamplerInfo(SamplerInfo& info, const Tensor::InsideDescribe::Region& sampler) {
    int sizeTotal = 1;
    for (int i=0; i<3; ++i) {
        info.size[i] = sampler.size[i];
        info.stride[i] = sampler.src.stride[i];
        info.extent[i] = sampler.dst.stride[i];
        sizeTotal *= info.size[i];
    }
    info.size[3] = sizeTotal;
    info.stride[3] = sampler.src.offset;
    info.extent[3] = sampler.dst.offset;
}
    
static std::string getUnitName(int bytes) {
    std::string unitName;
    switch (bytes) {
        case 1:
            unitName = "uchar";
            break;
        case 2:
            unitName = "short";
            break;
        case 4:
            unitName = "int";
            break;
        case 8:
            unitName = "short4";
            break;
        case 16:
            unitName = "int4";
            break;
        default:
            FUNC_PRINT(bytes);
            break;
    }
    return unitName;
}

static const char* gMultiBlitMetal = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct SamplerInfo {
    uint4 stride;//stride[3] + offset
    uint4 size;//size[3] + totalSize
    uint4 extent;//dstStride[3]+dstOffset
};
kernel void main0(const device T *in [[buffer(0)]],
                       device T *out [[buffer(1)]],
                       const device uint4* buf [[buffer(2)]],
                       uint3 tgid [[thread_position_in_grid]]) {
    uint4 limit = buf[0];
    const device SamplerInfo* infoP = (const device SamplerInfo*)(buf + 1);
    uint3 gid = tgid;
    gid.x = tgid.x % limit.x;
    uint n = tgid.x / limit.x;
    if (n < limit.y) {
        SamplerInfo info = infoP[n];
        if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
            uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
            uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
            out[int(dstOffset)] = in[int(srcOffset)];
        }
    }
}
)metal";

static const char* gSingleBlitMetal = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct SamplerInfo {
    uint4 stride;//stride[3] + offset
    uint4 size;//size[3] + totalSize
    uint4 extent;//dstStride[3]+dstOffset
};
kernel void main0(const device T *in [[buffer(0)]],
                       device T *out [[buffer(1)]],
                       constant SamplerInfo &info [[buffer(2)]],
                       uint3 gid [[thread_position_in_grid]]) {
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        out[int(dstOffset)] = in[int(srcOffset)];
    }
}
)metal";

static const char* gMultiRasterTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct SamplerInfo {
    uint4 stride;//stride[3] + offset
    uint4 size;//size[3] + totalSize
    uint4 extent;//dstStride[3]+dstOffset
};
kernel void main0(const device T *in [[buffer(0)]],
                       device T *out [[buffer(1)]],
                       const device uint4* buf [[buffer(2)]],
                       uint3 tgid [[thread_position_in_grid]]) {
    
    uint4 limit = buf[2];
    const device SamplerInfo* infoP = (const device SamplerInfo*)(buf + 3);
    uint3 gid = tgid;
    gid.x = tgid.x % limit.x;
    uint n = tgid.x / limit.x;
    if (n < limit.y) {
        SamplerInfo info = infoP[n];

        if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
            uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
            uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        #ifdef INPUT_FORMAT_NCHW
            int srcOffsetReal = srcOffset;
        #elif INPUT_FORMAT_NHWC
            int srcOffsetReal = srcOffset;
        #elif INPUT_FORMAT_C4NHW4
            uint4 src_shape = buf[0];//src nchw
            int src_batch   = src_shape.x;
            int src_channel = src_shape.y;
            int src_height  = src_shape.z;
            int src_width   = src_shape.w;
            int in_w = srcOffset % src_width; srcOffset /= src_width;
            int in_h = srcOffset % src_height; srcOffset /= src_height;
            int in_c = srcOffset % src_channel;
            int in_b = srcOffset / src_channel;
            int srcOffsetReal = (((in_b + (in_c / 4) * src_batch) * src_height + in_h) * src_width + in_w) * 4 + (in_c % 4);
        #endif

        #ifdef OUTPUT_FORMAT_NCHW
            int dstOffsetReal = dstOffset;
        #elif OUTPUT_FORMAT_NHWC
            int dstOffsetReal = dstOffset;
        #elif OUTPUT_FORMAT_C4NHW4
            uint4 dst_shape = buf[1];//dst nchw
            int dst_batch   = dst_shape.x;
            int dst_channel = dst_shape.y;
            int dst_height  = dst_shape.z;
            int dst_width   = dst_shape.w;
            int out_w = dstOffset % dst_width; dstOffset /= dst_width;
            int out_h = dstOffset % dst_height; dstOffset /= dst_height;
            int out_c = dstOffset % dst_channel;
            int out_b = dstOffset / dst_channel;
            int dstOffsetReal = (((out_b + (out_c / 4) * dst_batch) * dst_height + out_h) * dst_width + out_w) * 4 + (out_c % 4);
        #endif
            out[dstOffsetReal] = in[srcOffsetReal];
        }
    }
}
)metal";

static const char* gSingleRasterTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct SamplerInfo {
    uint4 stride;//stride[3] + offset
    uint4 size;//size[3] + totalSize
    uint4 extent;//dstStride[3]+dstOffset
};
kernel void main0(const device T *in [[buffer(0)]],
                       device T *out [[buffer(1)]],
                       const device uint4* buf [[buffer(2)]],
                       uint3 gid [[thread_position_in_grid]]) {
    SamplerInfo info = *((const device SamplerInfo*)(buf + 3));
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
    #ifdef INPUT_FORMAT_NCHW
        int srcOffsetReal = srcOffset;
    #elif INPUT_FORMAT_NHWC
        int srcOffsetReal = srcOffset;
    #elif INPUT_FORMAT_C4NHW4
        uint4 src_shape = buf[0];//src nchw
        int src_batch   = src_shape.x;
        int src_channel = src_shape.y;
        int src_height  = src_shape.z;
        int src_width   = src_shape.w;
        int in_w = srcOffset % src_width; srcOffset /= src_width;
        int in_h = srcOffset % src_height; srcOffset /= src_height;
        int in_c = srcOffset % src_channel;
        int in_b = srcOffset / src_channel;
        int srcOffsetReal = (((in_b + (in_c / 4) * src_batch) * src_height + in_h) * src_width + in_w) * 4 + (in_c % 4);
    #endif

    #ifdef OUTPUT_FORMAT_NCHW
        int dstOffsetReal = dstOffset;
    #elif OUTPUT_FORMAT_NHWC
        int dstOffsetReal = dstOffset;
    #elif OUTPUT_FORMAT_C4NHW4
        uint4 dst_shape = buf[1];//dst nchw
        int dst_batch   = dst_shape.x;
        int dst_channel = dst_shape.y;
        int dst_height  = dst_shape.z;
        int dst_width   = dst_shape.w;
        int out_w = dstOffset % dst_width; dstOffset /= dst_width;
        int out_h = dstOffset % dst_height; dstOffset /= dst_height;
        int out_c = dstOffset % dst_channel;
        int out_b = dstOffset / dst_channel;
        int dstOffsetReal = (((out_b + (out_c / 4) * dst_batch) * dst_height + out_h) * dst_width + out_w) * 4 + (out_c % 4);
    #endif
        out[dstOffsetReal] = in[srcOffsetReal];
    }
}
)metal";
    
static const char* gFillInt4 = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct MemsetInfo {
    int4 value;
    uint4 size;
};
kernel void main0(device int4 *out   [[buffer(0)]],
                       constant MemsetInfo &info        [[buffer(1)]],
                       uint3 gid                 [[thread_position_in_grid]]) {
    if (gid.x < info.size.x) {
        out[gid.x] = info.value;
    }
}
)metal";

id<MTLComputePipelineState> MetalRaster::getBlitPipeline(int bytes, Backend* backend, bool multiRegion) {
    auto mtbn = static_cast<MetalBackend*>(backend);
    std::string pipelineName;
    std::string unitName = getUnitName(bytes);
    if (multiRegion) {
        pipelineName = "blit_multi";
    } else {
        pipelineName = "blit";
    }
    
    std::vector<std::string> keys = {
        unitName,
        pipelineName
    };
    auto pipeline = mtbn->runtime()->findPipeline(keys);
    if (nil == pipeline) {
        MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
        compileOptions.preprocessorMacros = @{
            @"T" : @(unitName.c_str()),
        };
        if (multiRegion) {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gMultiBlitMetal, "main0", compileOptions);
        } else {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gSingleBlitMetal, "main0", compileOptions);
        }
        mtbn->runtime()->insertPipeline(keys, pipeline);
    }
    return pipeline;
}

MetalRaster::MetalRaster(Backend *backend) : MetalExecution(backend) {
    // Do nothing
}
MetalRaster::~MetalRaster() {
    auto mtbn = static_cast<MetalBackend*>(backend());
    if (nil != mZeroCopy) {
        mtbn->returnConstBuffer(mZeroCopy);
    }
    auto bufferAlloc = mtbn->getStaticBufferPool();
    for(auto& iter : mTempInputCopy) {
        bufferAlloc->free(iter.second.blit);
    }
}
struct MemsetInfo {
    int value[4];
    uint32_t size[4];
};
ErrorCode MetalRaster::onResize(const std::vector<Tensor *> &____inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(outputs.size() == 1);
    OpCommonUtils::rasterInputReset(____inputs, outputs[0]);
    auto output = outputs[0];
    auto outputDes = TensorUtils::getDescribe(output);
    auto des = outputDes;
    mNeedZero = !TensorUtils::regionIsFull(output);
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && output->length(1) % 4 != 0) {
        mNeedZero = true;
    }
    auto context  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend())->context();
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto bufferAlloc = mtbn->getStaticBufferPool();
    auto bytes = outputs[0]->getType().bytes();
    if (outputs[0]->getType().code == halide_type_float) {
        if (mtbn->useFp16InsteadFp32()) {
            bytes = 2;
        }
    }
    if (mNeedZero) {
        std::vector<std::string> keys = {
            "fill_int4"
        };
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gFillInt4, "main0", nil);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        mZeroPipeline = pipeline;
        if (nil == mZeroCopy) {
            mZeroCopy = mtbn->getConstBuffer(sizeof(MemsetInfo));
        }
    }

    for (auto& iter : mTempInputCopy) {
        bufferAlloc->free(iter.second.blit);
    }
    mTempInputCopy.clear();
    mOutputPtr = output;
#ifndef MNN_METAL_FORBID_RASTER_C4
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        bool fast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                fast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output, 4, true)) {
                fast = false;
                break;
            }
        }
        if (fast) {
            mBlitPipeline.resize(1);
            mBlitPipeline[0] = getBlitPipeline(bytes * 4, backend(), true);
            std::map<Tensor*, std::vector<int>> collectForTensor;
            for (int i=0; i< des->regions.size(); ++i) {
                auto& slice = des->regions[i];
                Tensor* t = slice.origin;
                auto coliter = collectForTensor.find(t);
                if (coliter == collectForTensor.end()) {
                    collectForTensor.insert(std::make_pair(t, std::vector<int>{i}));
                } else {
                    coliter->second.emplace_back(i);
                }
            }
            for (auto& iter : collectForTensor) {
                BlitInfo blit;
                auto memory = bufferAlloc->alloc(sizeof(SamplerInfo) * iter.second.size() + 4 * sizeof(uint32_t));
                blit.blit = std::make_pair(memory.first, memory.second);
                auto buffer = ((MetalRuntimeAllocator::MetalBufferAlloc*)memory.first)->getBuffer();

                auto infoP = (SamplerInfo*)((uint8_t*)[buffer contents] + 4 * sizeof(uint32_t) + memory.second);
                uint32_t maxSize[3] = {1, 1, 1};
                for (int v=0; v<iter.second.size(); ++v) {
                    auto& oldr = des->regions[iter.second[v]];
                    Tensor::InsideDescribe::Region slice;
                    OpCommonUtils::turnToPackRegion(oldr, slice, output, 4, true);
                    slice.dst.offset /= 4;
                    slice.src.offset /= 4;
                    writeSamplerInfo(infoP[v], slice);
                    maxSize[0] = ALIMAX(maxSize[0], slice.size[0]);
                    maxSize[1] = ALIMAX(maxSize[1], slice.size[1]);
                    maxSize[2] = ALIMAX(maxSize[2], slice.size[2]);
                }
                ((uint32_t*)((uint8_t*)[buffer contents] + memory.second))[0] = maxSize[0];
                 ((uint32_t*)((uint8_t*)[buffer contents] + memory.second))[1] = iter.second.size();
                auto local = [context computeBestGroupAndLocal:mBlitPipeline[0] threads:MTLSizeMake(maxSize[0] * iter.second.size(), maxSize[1], maxSize[2])];
                blit.global = local.first;
                blit.local = local.second;
                mTempInputCopy.insert(std::make_pair(iter.first, blit));
            }
            return NO_ERROR;
        }
    }
#endif
    
    std::map<Tensor*, std::vector<int>> collectForTensor;
    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        if (nullptr == slice.origin) {
            continue;
        }
        Tensor* t = slice.origin;
        auto coliter = collectForTensor.find(t);
        if (coliter == collectForTensor.end()) {
            collectForTensor.insert(std::make_pair(t, std::vector<int>{i}));
        } else {
            coliter->second.emplace_back(i);
        }
    }
    
    NSString* input_format;
    NSString* output_format;
    if(outputDes->dimensionFormat == MNN_DATA_FORMAT_NCHW) {
        output_format = @"OUTPUT_FORMAT_NCHW";
    } else if(outputDes->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
        output_format = @"OUTPUT_FORMAT_NHWC";
    } else {
        output_format = @"OUTPUT_FORMAT_C4NHW4";
    }
    std::string unitName = getUnitName(bytes);
    mBlitPipeline.resize(collectForTensor.size());
    int index = 0;
    for (auto& iter : collectForTensor) {
        auto origin = iter.first;

        if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NCHW) {
            input_format = @"INPUT_FORMAT_NCHW";
        } else if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
            input_format = @"INPUT_FORMAT_NHWC";
        } else {
            input_format = @"INPUT_FORMAT_C4NHW4";
        }
        std::vector<std::string> keys = {
            std::string([input_format UTF8String]),
            std::string([output_format UTF8String]),
            unitName,
        };
        if(iter.second.size() == 1) {
            keys.emplace_back("direct_raster_single");
        } else {
            keys.emplace_back("direct_raster_multi");
        }
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        
        if(nullptr == pipeline) {
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            options.preprocessorMacros = @{
                input_format : @"1",
                output_format : @"1",
                @"T" : @(unitName.c_str()),
            };
            if(iter.second.size() == 1) {
                pipeline = mtbn->makeComputePipelineWithSourceOption(gSingleRasterTemplate, "main0", options);
            } else {
                pipeline = mtbn->makeComputePipelineWithSourceOption(gMultiRasterTemplate, "main0", options);
            }
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        mBlitPipeline[index] = pipeline;
        
        BlitInfo blit;
        auto memory = bufferAlloc->alloc(sizeof(SamplerInfo) * iter.second.size() + 12 * sizeof(uint32_t));
        blit.blit = std::make_pair(memory.first, memory.second);
        auto buffer = ((MetalRuntimeAllocator::MetalBufferAlloc*)memory.first)->getBuffer();

        auto infoP = (SamplerInfo*)((uint8_t*)[buffer contents] + 12 * sizeof(uint32_t) + memory.second);

        uint32_t maxSize[3] = {1, 1, 1};
        for (int v=0; v<iter.second.size(); ++v) {
            auto& slice = des->regions[iter.second[v]];
            writeSamplerInfo(infoP[v], slice);
            maxSize[0] = ALIMAX(maxSize[0], slice.size[0]);
            maxSize[1] = ALIMAX(maxSize[1], slice.size[1]);
            maxSize[2] = ALIMAX(maxSize[2], slice.size[2]);
        }
        
        uint32_t* shape = (uint32_t*)((uint8_t*)[buffer contents] + memory.second);
        int origin_area = 1;
        for(int i = 2; i < origin->dimensions(); i++) {
            origin_area *= origin->shape()[i];
        }
        int output_area = 1;
        for(int i = 2; i < output->dimensions(); i++) {
            output_area *= output->length(i);
        }
        shape[0] = ALIMAX(1, origin->length(0));
        shape[1] = ALIMAX(1, origin->length(1));
        shape[2] = ALIMAX(1, origin_area);
        shape[3] = 1;
        shape[4] = ALIMAX(1, output->length(0));
        shape[5] = ALIMAX(1, output->length(1));
        shape[6] = ALIMAX(1, output_area);
        shape[7] = 1;
        shape[8] = maxSize[0];
        shape[9] = iter.second.size();

        auto local = [context computeBestGroupAndLocal:mBlitPipeline[index++] threads:MTLSizeMake(maxSize[0] * iter.second.size(), maxSize[1], maxSize[2])];
        blit.global = local.first;
        blit.local = local.second;
        mTempInputCopy.insert(std::make_pair(iter.first, blit));
    }
    return NO_ERROR;
}

void MetalRaster::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    if (mNeedZero) {
        size_t sizeInBytes = backend->getTensorSizeInBytes(outputs[0]);
        size_t size = sizeInBytes / (4 * sizeof(int32_t));
        auto ptr = (MemsetInfo*)[mZeroCopy contents];
        ptr->size[0] = (uint32_t)size;
        [encoder setComputePipelineState:mZeroPipeline];
        MetalBackend::setTensor(mOutputPtr, encoder, 0);
        [encoder setBuffer: mZeroCopy offset:0 atIndex: 1];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(size, 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    
    bool singlePipeline = false;
    int index = 0;
    if(mBlitPipeline.size() == 1) {
        singlePipeline = true;
        [encoder setComputePipelineState:mBlitPipeline[0]];
    } else {
        MNN_ASSERT(mTempInputCopy.size() == mBlitPipeline.size());
    }
    for (auto& iter : mTempInputCopy) {
        if(!singlePipeline) {
            [encoder setComputePipelineState:mBlitPipeline[index++]];
        }
        MetalBackend::setTensor(iter.first, encoder, 0);
        MetalBackend::setTensor(mOutputPtr, encoder, 1);
        auto& blit = iter.second;
        auto buffer = ((MetalRuntimeAllocator::MetalBufferAlloc*)blit.blit.first)->getBuffer();
        [encoder setBuffer: buffer offset:blit.blit.second atIndex: 2];

        [encoder dispatchThreadgroups:blit.global threadsPerThreadgroup:blit.local];
    }
}

class MetalRasterCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {

        return new MetalRaster(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalRasterCreator, OpType_Raster);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
