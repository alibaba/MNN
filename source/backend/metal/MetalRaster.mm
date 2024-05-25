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
    std::string unitName;
    if (multiRegion) {
        pipelineName = "blit_multi";
    } else {
        pipelineName = "blit";
    }
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
    for (auto b : mShapeTemp) {
        mtbn->returnConstBuffer(b);
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
    mTempInput.clear();
    mTempInputCopy.clear();
    mTempOutput = nullptr;
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
            mBlitPipeline = getBlitPipeline(bytes * 4, backend(), true);
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
                auto local = [context computeBestGroupAndLocal:mBlitPipeline threads:MTLSizeMake(maxSize[0] * iter.second.size(), maxSize[1], maxSize[2])];
                blit.global = local.first;
                blit.local = local.second;
                mTempInputCopy.insert(std::make_pair(iter.first, blit));
            }
            return NO_ERROR;
        }
    }
#endif
    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        auto origin = slice.origin;
        if (TensorUtils::getDescribe(origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            continue;
        }
        if (mTempInput.find(origin)!=mTempInput.end()) {
            continue;
        }
        std::shared_ptr<Tensor> newTensor(new Tensor);
        TensorUtils::copyShape(origin, newTensor.get());
        TensorUtils::getDescribe(newTensor.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        newTensor->buffer().type = origin->getType();
        TensorUtils::setLinearLayout(newTensor.get());
        mTempInput.insert(std::make_pair(origin, newTensor));
    }
    if (MNN_DATA_FORMAT_NC4HW4 == outputDes->dimensionFormat) {
        mTempOutput.reset(new Tensor);
        TensorUtils::setupTensorInfo(output, mTempOutput.get(), MNN_DATA_FORMAT_NCHW);
    }
    if (nullptr != mTempOutput) {
        auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mOutputPtr = mTempOutput.get();
    }
    for (auto& iter : mTempInput) {
        auto res = backend()->onAcquireBuffer(iter.second.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
    }
    for (auto& iter : mTempInput) {
        backend()->onReleaseBuffer(iter.second.get(), Backend::DYNAMIC);
    }
    if (nullptr != mTempOutput) {
        backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }
    mBlitPipeline = getBlitPipeline(bytes, backend(), true);
    std::map<Tensor*, std::vector<int>> collectForTensor;
    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        if (nullptr == slice.origin) {
            continue;
        }
        auto iter = mTempInput.find(slice.origin);
        Tensor* t = slice.origin;
        if (iter != mTempInput.end()) {
            t = iter->second.get();
        }
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

        blit.blit = std::make_pair(memory.first, memory.second);
        uint32_t maxSize[3] = {1, 1, 1};
        for (int v=0; v<iter.second.size(); ++v) {
            auto& slice = des->regions[iter.second[v]];
            writeSamplerInfo(infoP[v], slice);
            maxSize[0] = ALIMAX(maxSize[0], slice.size[0]);
            maxSize[1] = ALIMAX(maxSize[1], slice.size[1]);
            maxSize[2] = ALIMAX(maxSize[2], slice.size[2]);
        }
        ((uint32_t*)((uint8_t*)[buffer contents] + memory.second))[0] = maxSize[0];
         ((uint32_t*)((uint8_t*)[buffer contents] + memory.second))[1] = iter.second.size();
        auto local = [context computeBestGroupAndLocal:mBlitPipeline threads:MTLSizeMake(maxSize[0] * iter.second.size(), maxSize[1], maxSize[2])];
        blit.global = local.first;
        blit.local = local.second;
        mTempInputCopy.insert(std::make_pair(iter.first, blit));
    }
    for (auto b : mShapeTemp) {
        mtbn->returnConstBuffer(b);
    }
    mShapeTemp.clear();
    for (int i = 0; i < mTempInput.size(); ++i) {
        id<MTLBuffer> shape = mtbn->getConstBuffer(0);
        mShapeTemp.emplace_back(std::move(shape));
    }
    if (nullptr != mTempOutput) {
        mShapeTemp.emplace_back(mtbn->getConstBuffer(0));
    }
    return NO_ERROR;
}

void MetalRaster::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    int out_offset = TensorUtils::getDescribe(outputs[0])->extra.offset;
    if (nullptr != mTempOutput) {
        out_offset = TensorUtils::getDescribe(mTempOutput.get())->extra.offset;
    }
    if (mNeedZero) {
        size_t sizeInBytes;
        if (mTempOutput != nullptr) {
            sizeInBytes = backend->getTensorSizeInBytes(mTempOutput.get());
        } else {
            sizeInBytes = backend->getTensorSizeInBytes(outputs[0]);
        }
        size_t size = sizeInBytes / (4 * sizeof(int32_t));
        auto ptr = (MemsetInfo*)[mZeroCopy contents];
        ptr->size[0] = (uint32_t)size;
        [encoder setComputePipelineState:mZeroPipeline];
        MetalBackend::setTensor(mOutputPtr, encoder, 0);
        [encoder setBuffer: mZeroCopy offset:0 atIndex: 1];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(size, 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    int index = 0;
    for (auto& iter : mTempInput) {
        backend->onCopyBuffer(iter.first, iter.second.get(), encoder, mShapeTemp[index++]);
    }

    [encoder setComputePipelineState:mBlitPipeline];
    for (auto& iter : mTempInputCopy) {
        MetalBackend::setTensor(iter.first, encoder, 0);
        MetalBackend::setTensor(mOutputPtr, encoder, 1);
        auto& blit = iter.second;
        auto buffer = ((MetalRuntimeAllocator::MetalBufferAlloc*)blit.blit.first)->getBuffer();
        [encoder setBuffer: buffer offset:blit.blit.second atIndex: 2];
        [encoder dispatchThreadgroups:blit.global threadsPerThreadgroup:blit.local];
    }
    if (nullptr != mTempOutput) {
        backend->onCopyBuffer(mTempOutput.get(), outputs[0], encoder, mShapeTemp[index]);
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
