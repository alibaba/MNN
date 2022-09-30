//
//  RasterExecution.cpp
//  MNN
//
//  Created by MNN on b'2020/04/02'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "RasterExecution.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/BufferAllocator.hpp"
#include "Raster.cuh"
#include "Transpose.cuh"
#include "MNNCUDADefine.hpp"
namespace MNN {
namespace CUDA {


static void getBatchChannelArea(const Tensor* t, int& batch, int& channel, int& area) {
    batch = t->batch();
    if (t->dimensions() == 4) {
        channel = t->channel();
        area = t->width() * t->height();
    } else if (t->dimensions() == 3) {
        auto format = TensorUtils::getDescribe(t)->dimensionFormat;
        if (format == MNN_DATA_FORMAT_NHWC) {
            channel = t->length(2);
            area    = t->length(1);
        } else {
            channel = t->length(1);
            area    = t->length(2);
        }
    } else {
        auto format = TensorUtils::getDescribe(t)->dimensionFormat;
        if (format == MNN_DATA_FORMAT_NHWC) {
            for (int i = t->dimensions() - 1; i > 0; i--) {
                int len = t->length(i);
                if (len > 1) {
                    if (channel == 1) {
                        channel = len;
                    } else {
                        area *= len;
                    }
                }
            }
        } else {
            for (int i = 1; i < t->dimensions(); i++) {
                int len = t->length(i);
                if (len > 1) {
                    if (channel == 1) {
                        channel = len;
                    } else {
                        area *= len;
                    }
                }
            }
        }
    }
}

static int _singleConvert(const Tensor::InsideDescribe::Region& region, const Tensor* dest) {
    auto origin = region.origin;
    auto srcFormat = TensorUtils::getDescribe(origin)->dimensionFormat;
    auto dstFormat = TensorUtils::getDescribe(dest)->dimensionFormat;
    if (srcFormat == dstFormat) {
        return 0;
    }
    if (0 != region.src.offset || 0 != region.dst.offset) {
        return 0;
    }
    int dstBatch = 1, dstChannel = 1, dstArea = 1,
        srcBatch = 1, srcChannel = 1, srcArea = 1;
    getBatchChannelArea(origin, srcBatch, srcChannel, srcArea);
    getBatchChannelArea(dest, dstBatch, dstChannel, dstArea);
    if (dstBatch != srcBatch) {
        return 0;
    }
    if (dstChannel != srcChannel) {
        return 0;
    }
    if (dstArea != srcArea) {
        return 0;
    }
    auto totalSize = dstBatch * dstChannel * dstArea;
    int srcSize = 1;
    int dstSize = 1;
    int res = 1;
    for (int i=0; i<3; ++i) {
        if (region.size[i] == 1) {
            continue;
        }
        if (region.src.stride[i] != region.dst.stride[i]) {
            if (dstArea == 1) {
                // Batch / Channel transpose
                return 0;
            }
            res = 2;
        }
        srcSize += (region.size[i] - 1) * region.src.stride[i];
        dstSize += (region.size[i] - 1) * region.dst.stride[i];
    }
    if (srcSize != totalSize || dstSize != totalSize ) {
        return 0;
    }
    // Check If it can be described as NHWC <-> NC4HW4 transpose
    if (2 == res) {
        int srcChannelStride;
        int dstChannelStride;
        int srcAreaStride;
        int dstAreaStride;
        if (MNN_DATA_FORMAT_NC4HW4 == srcFormat) {
            srcChannelStride = srcArea;
            srcAreaStride = 1;
            dstChannelStride = 1;
            dstAreaStride = srcChannel;
        } else {
            srcChannelStride = 1;
            srcAreaStride = srcChannel;
            dstAreaStride = 1;
            dstChannelStride = srcArea;
        }
        for (int i=0; i<3; ++i) {
            if (region.size[i] == 1) {
                continue;
            }
            if (region.size[i] == dstBatch) {
                if (region.src.stride[i] != region.dst.stride[i]) {
                    return 0;
                }
                continue;
            }
            if (region.size[i] == srcChannel) {
                if (region.src.stride[i] != srcChannelStride || region.dst.stride[i] != dstChannelStride) {
                    return 0;
                }
            }
            if (region.size[i] == srcArea) {
                if (region.src.stride[i] != srcAreaStride || region.dst.stride[i] != dstAreaStride) {
                    return 0;
                }
            }
        }
        return 2;
    }
    return 1;
}

static bool _equalSizeStride(const Tensor::InsideDescribe::Region& slice0, const Tensor::InsideDescribe::Region& slice1) {
    if (slice0.src.stride[0] != slice1.src.stride[0] || slice0.dst.stride[0] != slice1.dst.stride[0]) {
        //MNN_PRINT("Raster total:%d, index:%d, src stride0:%d-%d, , dst stride0:%d-%d\n", mTempInputCopy.size(), i, slice.src.stride[0], slice0.src.stride[0], slice.dst.stride[0], slice0.dst.stride[0]);
        return false;
    }
    if (slice0.src.stride[1] != slice1.src.stride[1] || slice0.dst.stride[1] != slice1.dst.stride[1]) {
        //MNN_PRINT("Raster total:%d, index:%d, src stride1:%d-%d, , dst stride1:%d-%d\n", mTempInputCopy.size(), i, slice.src.stride[1], slice0.src.stride[1], slice.dst.stride[1], slice0.dst.stride[1]);
        return false;
    }      
    if (slice0.src.stride[2] != slice1.src.stride[2] || slice0.dst.stride[2] != slice1.dst.stride[2]) {
        //MNN_PRINT("Raster total:%d, index:%d, src stride2:%d-%d, , dst stride2:%d-%d\n", mTempInputCopy.size(), i, slice.src.stride[2], slice0.src.stride[2], slice.dst.stride[2], slice0.dst.stride[2]);
        return false;
    }
    if (slice0.size[0] != slice1.size[0] || slice0.size[1] != slice1.size[1] || slice0.size[2] != slice1.size[2]) {
        //MNN_PRINT("Raster total:%d, index:%d, copy size:%d-%d-%d, %d-%d-%d\n", mTempInputCopy.size(), i, slice.size[0], slice.size[1], slice.size[2], slice0.size[0], slice0.size[1], slice0.size[2]);
        return false;
    }
    return true;
}

static bool _directBlitC4(const Tensor::InsideDescribe::Region& slice0, const Tensor::InsideDescribe::Region& slice1) {
    if(slice0.size[1] % PACK_NUMBER != 0 || slice0.size[0] != 1) {
        return false;
    }
    if(slice1.size[1] % PACK_NUMBER != 0 || slice1.size[0] != 1) {
        return false;
    }
    if(slice0.dst.offset % (slice0.size[1] * slice0.size[0]) != 0) {
        return false;
    }
    if(slice1.dst.offset % (slice1.size[1] * slice1.size[0]) != 0) {
        return false;
    }
    return _equalSizeStride(slice0, slice1); 
}

static void _turnToNewRegion(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& newRegion, int multiStride) {
    newRegion.size[0] = region.size[0];
    newRegion.size[1] = region.size[2];
    newRegion.size[2] = region.size[1];

    newRegion.src.stride[0] = region.src.stride[0];
    newRegion.src.stride[1] = region.src.stride[2] * region.size[1];
    newRegion.src.stride[2] = region.src.stride[1] / region.size[2];

    newRegion.dst.stride[0] = region.dst.stride[0] * multiStride;
    newRegion.dst.stride[1] = region.dst.stride[2] * region.size[1] * multiStride;
    newRegion.dst.stride[2] = region.dst.stride[1] / region.size[2];

    newRegion.src.offset = region.src.offset / region.size[2];
    newRegion.dst.offset = region.dst.offset / region.size[2];
}

ErrorCode RasterExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    auto des = TensorUtils::getDescribe(input);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(input);
    mTempInputCopy.clear();
    mTempInput.clear();

    mTempOutput = nullptr;
    mOutputPtr = output; 

    mFast = false;
    int pack = PACK_NUMBER;
    // all_srcFormat == dstFormat == NC4HW4 : Fast Exe
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mFast = true;
        auto& slice0 = des->regions[0];
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            //MNN_PRINT("%d-%d-%d, %d-%d-%d-%d\n", slice.size[0], slice.size[1], slice.size[2], slice.src.stride[1], slice.src.stride[2], slice.dst.stride[1], slice.dst.stride[2]);
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
            if(!_directBlitC4(slice0, slice)) {
                mFast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output, pack, false, true)) {
                mFast = false;
                break;
            }
        }
        //MNN_PRINT("raster fast:%d\n", mFast);
        if (mFast) {
            int multiStride = 1;
            for (int i=0; i< des->regions.size(); ++i) {
                auto& slice = des->regions[i];
                if(slice.dst.offset / (slice.size[0] * slice.size[1]) >= 1) {
                    int batchChannel = slice.dst.offset / (slice.size[1] * slice.size[2]) + 1;
                    multiStride = multiStride > batchChannel ? multiStride : batchChannel;
                }
            }
            for (int i=0; i< des->regions.size(); ++i) {
                auto& slice = des->regions[i];
                if (slice.origin == nullptr) {
                    continue;
                }
                Tensor::InsideDescribe::Region newRegion;
                _turnToNewRegion(slice, newRegion, multiStride);
                mFastBlit.emplace_back(std::make_pair(slice.origin, std::move(newRegion)));
            }
            return NO_ERROR;
        }
    }

    mSingleConvert = 0;
    // srcNum == 1 && srcFormat != dstFormat : Single Convert
    if (des->regions.size() == 1) {
        mSingleConvert = _singleConvert(des->regions[0], output);
        if (mSingleConvert > 0) {
            return NO_ERROR;
        }
    }

    for(int i = 0; i < des->regions.size(); i++) {
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

    for (int i = 0; i < des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        if (nullptr == slice.origin) {
            continue;
        }
        auto iter = mTempInput.find(slice.origin);
        if (iter != mTempInput.end()) {
            mTempInputCopy.emplace_back(std::make_pair(iter->second.get(), &slice));
            continue;
        }
        mTempInputCopy.emplace_back(std::make_pair(slice.origin, &slice));
    }


    //MNN_PRINT("Raster copy size:%d\n", mTempInputCopy.size());
    if(mTempInputCopy.size() > 1) {
        mFuseRaster.first = 1;
        mFuseRaster.second = mTempInputCopy.size();
        auto& slice0 = *mTempInputCopy[0].second;
        for (int i = 1; i < mTempInputCopy.size(); ++i) {
            auto& slice = *mTempInputCopy[i].second;
            if (mTempInputCopy[i].first != mTempInputCopy[0].first) {
                mFuseRaster.first = 0;
                //MNN_PRINT("Raster total:%d, index:%d, origin:%p-%p\n", mTempInputCopy.size(), i, mTempInputCopy[i].first, mTempInputCopy[0].first);
                break;
            }
            if(!_equalSizeStride(slice0, slice)) {
                mFuseRaster.first = 0;
            }
        }
    }
    if(mFuseRaster.first > 0) {
        auto& slice0 = *mTempInputCopy[0].second;
        auto tensor = mTempInputCopy[0].first;
        int regionSize = mTempInputCopy.size();
        std::vector<int32_t> temp(2*regionSize, 0);
        // TODO: Reduce logic for these code
        mFuseRaster.first = 4;
        for (int i = 0; i < regionSize; ++i) {
            auto& slice = *mTempInputCopy[i].second;
            temp[i] = slice.src.offset;
            temp[regionSize+i] = slice.dst.offset;
            if (temp[i] % 4 != 0 || temp[regionSize+i] % 4 != 0) {
                mFuseRaster.first = 1;
            }
            //printf("%d-%d-%d\n", regionSize, temp[i], temp[regionSize+i]);
        }
        //save srcOffset/dstOffset to Device
        offsetTensor.reset(Tensor::createDevice<int32_t>({2*regionSize}));
        backend()->onAcquireBuffer(offsetTensor.get(), Backend::STATIC);
        mOffset = (void *)offsetTensor.get()->buffer().device;
        cuda_check(cudaMemcpy(mOffset, temp.data(), 2*regionSize*sizeof(int32_t), cudaMemcpyHostToDevice));
        mTempInputCopy.clear();
        mTempInputCopy.emplace_back(std::make_pair(tensor, &slice0));
    }

    for (auto& iter : mTempInput) {
        backend()->onReleaseBuffer(iter.second.get(), Backend::DYNAMIC);
    }
    if (nullptr != mTempOutput) {
        backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

void RasterExecution::executeFaster(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) const {
    auto bn = static_cast<CUDABackend*>(backend());
    auto input = inputs[0];
    auto output = outputs[0];
    auto bytes = bn->getBytes(output);
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    if (mNeedZero) {
        auto size = static_cast<CUDABackend*>(backend())->realSize(output) * bytes;
        cudaMemset((uint8_t*)output->deviceId(), 0, size);
    }
    // Use mFastBlit
    for (auto& iter : mFastBlit) {
        auto srcPtr = (uint8_t*)iter.first->deviceId() + iter.second.src.offset * bytes;
        auto dstPtr = (uint8_t*)output->deviceId() + iter.second.dst.offset * bytes;
        RasterBlit(dstPtr, srcPtr, iter.second.size, iter.second.src.stride, iter.second.dst.stride, bytes, runtime);
    }
}

ErrorCode RasterExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mFast) {
        executeFaster(inputs, outputs);
        return NO_ERROR;
    }
    auto bn = static_cast<CUDABackend*>(backend());
    auto input = inputs[0];
    auto output = outputs[0];
    auto bytes = bn->getBytes(output);
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    // printf("raster format:%d -> %d, addr:%p %p\n", TensorUtils::getDescribe(input)->dimensionFormat, \
    //     TensorUtils::getDescribe(output)->dimensionFormat, \
    //     input->deviceId(), output->deviceId());

    if (mSingleConvert > 0) {
        auto realInput = TensorUtils::getDescribe(input)->regions[0].origin;
        int srcBatch = 1, srcChannel = 1, srcArea = 1;
        getBatchChannelArea(realInput, srcBatch, srcChannel, srcArea);
        auto sourceFormat = TensorUtils::getDescribe(realInput)->dimensionFormat;
        auto destFormat = TensorUtils::getDescribe(output)->dimensionFormat;
        int batchStride = srcChannel * srcArea * bytes;
        int inputBatchStride = batchStride;
        int outputBatchStride = batchStride;
        PackInfo pack;
        pack.inside = srcArea;
        pack.axis = srcChannel;
        pack.unit = PACK_NUMBER;
        pack.outside = srcBatch;
        if (mSingleConvert == 1) {
            pack.axisStride = srcArea;
            pack.insideStride = 1;
        } else if (mSingleConvert == 2) {
            pack.axisStride = 1;
            pack.insideStride = srcChannel;
        }
        auto srcPtr = (void*)realInput->deviceId();
        auto dstPtr = (void*)output->deviceId();
        if (MNN_DATA_FORMAT_NC4HW4 == sourceFormat) {
            if (realInput->dimensions() <= 1) {
                cudaMemcpy(dstPtr, srcPtr, bn->realSize(realInput) * bytes, cudaMemcpyDeviceToDevice);
                return NO_ERROR;
            }
            UnpackBuffer(dstPtr, srcPtr, &pack, bytes, runtime);            
        } else {
            if (output->dimensions() <= 1) {
                cudaMemcpy(dstPtr, srcPtr, bn->realSize(realInput) * bytes, cudaMemcpyDeviceToDevice);
                return NO_ERROR;
            }
            PackBuffer(dstPtr, srcPtr, &pack, bytes, runtime);            
        }
        return NO_ERROR;
    }

    if (mNeedZero) {
        auto size = static_cast<CUDABackend*>(backend())->realSize(mOutputPtr) * bytes;
        cudaMemset((uint8_t*)mOutputPtr->deviceId(), 0, size);
    }
    for (auto& iter : mTempInput) {
        backend()->onCopyBuffer(iter.first, iter.second.get());
    }
    //printf("\n%d\n", mFuseRaster.first);
    if(mFuseRaster.first > 0) {
        MNN_ASSERT(mTempInputCopy.size() == 1);
        auto& iter  = mTempInputCopy[0];
        auto& slice = *(iter.second);
        auto srcPtr = (uint8_t*)iter.first->deviceId();
        auto dstPtr = (uint8_t*)mOutputPtr->deviceId();
        //printf("fuseRaster:%p-%p\n", mSrcOffset, mDstOffset);

        FuseRasterBlit(dstPtr, srcPtr, slice.size, slice.src.stride, slice.dst.stride, mFuseRaster.second, mOffset, bytes, runtime, mFuseRaster.first);
    } else {
        for (auto& iter : mTempInputCopy) {
            auto srcPtr = (uint8_t*)iter.first->deviceId() + iter.second->src.offset * bytes;
            auto dstPtr = (uint8_t*)mOutputPtr->deviceId() + iter.second->dst.offset * bytes;
            RasterBlit(dstPtr, srcPtr, iter.second->size, iter.second->src.stride, iter.second->dst.stride, bytes, runtime);
        }
    }

    if (nullptr != mTempOutput) {
        backend()->onCopyBuffer(mTempOutput.get(), output);
    }
    return NO_ERROR;
}

class RasterExecutionFactory : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new RasterExecution(backend);
    }
};

static CUDACreatorRegister<RasterExecutionFactory> __init(OpType_Raster);

}
}