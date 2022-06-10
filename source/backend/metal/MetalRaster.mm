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
    unsigned int imageSize[4];
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
MetalRaster::MetalRaster(Backend *backend) : Execution(backend) {
    // Do nothing
}
ErrorCode MetalRaster::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    auto des = TensorUtils::getDescribe(input);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(input);
    auto context  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend())->context();
    auto bytes = outputs[0]->getType().bytes();

    mTempInput.clear();
    mTempOutput = nullptr;
    mOutputPtr = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)(output->deviceId()))->getBuffer();
#ifndef MNN_METAL_FORBID_RASTER_C4
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        bool fast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                fast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output)) {
                fast = false;
                break;
            }
        }
        mFast = fast;
        if (fast) {
            NSString* kernelName = nil;
            switch (bytes) {
                case 4:
                    kernelName = @"blit_intx4";
                    break;
                case 2:
                    kernelName = @"blit_int64";
                    break;
                case 1:
                    kernelName = @"blit_int";
                    break;
                default:
                    break;
            }
            if (outputs[0]->getType().code == halide_type_float) {
#if MNN_METAL_FULL_PRECISION
                kernelName = @"blit_intx4";
#else
                kernelName = @"blit_int64";
#endif
            }
            mBlitPipeline = [context pipelineWithName:kernelName];

            for (int i=0; i< des->regions.size(); ++i) {
                auto& slice = des->regions[i];
                Tensor::InsideDescribe::Region newRegion;
                OpCommonUtils::turnToPackRegion(slice, newRegion, output, 4);
                newRegion.dst.offset /= 4;
                newRegion.src.offset /= 4;
                SamplerInfo info;
                writeSamplerInfo(info, newRegion);
                auto local = [context computeBestGroupAndLocal:mBlitPipeline threads:MTLSizeMake(newRegion.size[0], newRegion.size[1], newRegion.size[2])];
                auto buffer = [context newDeviceBuffer:sizeof(SamplerInfo) bytes:&info access:CPUWriteOnly];
                mTempInputCopy.emplace_back(std::make_tuple(( id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)newRegion.origin->deviceId())->getBuffer(), buffer, local.first, local.second, TensorUtils::getDescribe(newRegion.origin)->extra.offset));
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
        mOutputPtr = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)(mTempOutput->deviceId()))->getBuffer();

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
    NSString* kernelName = nil;
    switch (bytes) {
        case 4:
            kernelName = @"blit_int";
            break;
        case 2:
            kernelName = @"blit_int16";
            break;
        case 1:
            kernelName = @"blit_int8";
            break;
        default:
            break;
    }
    if (outputs[0]->getType().code == halide_type_float) {
#if MNN_METAL_FULL_PRECISION
        kernelName = @"blit_int";
#else
        kernelName = @"blit_int16";
#endif
    }
    mBlitPipeline = [context pipelineWithName:kernelName];
    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        if (nullptr == slice.origin) {
            continue;
        }
        SamplerInfo info;
        writeSamplerInfo(info, slice);
        auto buffer = [context newDeviceBuffer:sizeof(SamplerInfo) bytes:&info access:CPUWriteOnly];

        auto iter = mTempInput.find(slice.origin);
        auto local = [context computeBestGroupAndLocal:mBlitPipeline threads:MTLSizeMake(slice.size[0], slice.size[1], slice.size[2])];
        if (iter != mTempInput.end()) {
            mTempInputCopy.emplace_back(std::make_tuple(( id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)iter->second->deviceId())->getBuffer(), buffer, local.first, local.second, TensorUtils::getDescribe(iter->second.get())->extra.offset));
            continue;
        }
        mTempInputCopy.emplace_back(std::make_tuple(( id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)(slice.origin->deviceId()))->getBuffer(), buffer, local.first, local.second, TensorUtils::getDescribe(slice.origin)->extra.offset));
    }
    mShapeTemp.clear();
    for (int i = 0; i < mTempInput.size(); ++i) {
        id<MTLBuffer> shape = [context newDeviceBuffer:4*sizeof(int) access:CPUWriteOnly];
        mShapeTemp.emplace_back(std::move(shape));
    }
    if (nullptr != mTempOutput) {
        mShapeTemp.emplace_back([context newDeviceBuffer:4*sizeof(int) access:CPUWriteOnly]);
    }
    return NO_ERROR;
}

ErrorCode MetalRaster::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    
    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }

    auto func = [=](){
        int out_offset = TensorUtils::getDescribe(outputs[0])->extra.offset;
        if (nullptr != mTempOutput) {
            out_offset = TensorUtils::getDescribe(mTempOutput.get())->extra.offset;
        }
        if (mNeedZero) {
            backend->flushEncoder();
            auto size = outputs[0]->elementSize();
            if (mTempOutput != nullptr) {
                size = mTempOutput->elementSize();
            }
            size = ((size + 3) / 4) * 4 * sizeof(metal_float);
            auto blitEncode = [context encoderBlit_net];
            [blitEncode fillBuffer:mOutputPtr range:NSMakeRange(out_offset, size) value:0];
            [blitEncode endEncoding];
        }
        auto encoder   = backend->encoder();
        int index = 0;
        for (auto& iter : mTempInput) {
            backend->onCopyBuffer(iter.first, iter.second.get(), encoder, mShapeTemp[index++]);
        }

        [encoder setComputePipelineState:mBlitPipeline];
        for (auto& iter : mTempInputCopy) {
            [encoder setBuffer: std::get<0>(iter) offset: std::get<4>(iter) atIndex: 0];
            [encoder setBuffer: mOutputPtr offset:out_offset atIndex: 1];
            [encoder setBuffer: std::get<1>(iter) offset:0 atIndex: 2];
            [encoder dispatchThreadgroups:std::get<2>(iter) threadsPerThreadgroup:std::get<3>(iter)];
        }
        if (nullptr != mTempOutput) {
            backend->onCopyBuffer(mTempOutput.get(), outputs[0], encoder, mShapeTemp[index]);
        }

        if(backend->isCmdBufferCommit()) {
            backend->flushEncoder();
            [context commit_net];
        }
    };
    func();
    backend->addOpEncoder(func);
    return NO_ERROR;
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
