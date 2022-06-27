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
// Detect if the region is a transpose
static bool _transpose(const Tensor::InsideDescribe::Region& region) {
    int srcOne = -1, dstOne = -1;
    for (int i = 0; i < 3; i++) {
        if (region.src.stride[i] == 1 && region.size[i] != 1) {
            if (srcOne >= 0 || region.size[i] < 4) {
                return false;
            }
            srcOne = i;
        }
        if (region.dst.stride[i] == 1 && region.size[i] != 1) {
            if (dstOne >= 0 || region.size[i] < 4) {
                return false;
            }
            dstOne = i;
        }
    }
    return srcOne >= 0 && dstOne >= 0 && srcOne != dstOne;
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

ErrorCode RasterExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    auto des = TensorUtils::getDescribe(input);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(input);
    mZeroPoint = 0;
    mTempInput.clear();
    mFastBlit.clear();
    mFuseRaster.first = 0;
    mTempOutput = nullptr;
    auto midFormat = MNN_DATA_FORMAT_NCHW;
    mTempInputCopy.clear();
    mOutputPtr = output;
    mFast = false;
    int pack = PACK_NUMBER;
    // all_srcFormat == dstFormat == NC4HW4 : Fast Exe
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mFast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output, pack, true)) {
                mFast = false;
                break;
            }
        }
        if (mFast) {
            for (int i=0; i< des->regions.size(); ++i) {
                auto& slice = des->regions[i];
                if (slice.origin == nullptr) {
                    continue;
                }
                Tensor::InsideDescribe::Region newRegion;
                OpCommonUtils::turnToPackRegion(slice, newRegion, output, pack, true);
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
    // Acquire Buffer for temp output
    // TODO: optimize it
    if (MNN_DATA_FORMAT_NC4HW4 == outputDes->dimensionFormat) {
        mTempOutput.reset(new Tensor);
        TensorUtils::setupTensorInfo(output, mTempOutput.get(), midFormat);
    }
    if (nullptr != mTempOutput) {
        auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mOutputPtr = mTempOutput.get();
    }
    // input is NC4HW4 add Convert
    std::vector<Tensor*> forRelease;
    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        auto origin = slice.origin;
        if (slice.mask != 0) {
            mTempInputCopy.emplace_back(std::make_pair(origin, &slice));
            continue;
        }
        // if tensor is not NC4HW4 or has been merged, don't need deal
        if (TensorUtils::getDescribe(origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            mTempInputCopy.emplace_back(std::make_pair(origin, &slice));
            continue;
        }
        // if NC4HW4's C%4 == 0, change convert to transpose and fuse it
        if (origin->batch() == 1 && origin->channel() % pack == 0) {
            int channel = origin->channel();
            int area = 1;
            // conv3d/pool3d will has 5 dims, area = depth * width * height, otherwise area = width * height
            for (int d = 2; d < origin->dimensions(); d++) {
                area *= origin->length(d);
            }
            Tensor::InsideDescribe::Region regionTmp;
            regionTmp.src.offset = 0;
            regionTmp.src.stride[0] = area * pack;
            regionTmp.src.stride[1] = 1;
            regionTmp.src.stride[2] = pack;
            regionTmp.dst.offset = 0;
            regionTmp.dst.stride[0] = area * pack;
            regionTmp.dst.stride[1] = area;
            regionTmp.dst.stride[2] = 1;
            regionTmp.size[0] = channel / pack;
            regionTmp.size[1] = pack;
            regionTmp.size[2] = area;
            regionTmp.origin = slice.origin;
            bool merge = TensorUtils::fuseRegion(regionTmp, slice);
            if (merge) {
                // cache the merged tensor
                slice.mask = 1;
                mTempInputCopy.emplace_back(std::make_pair(origin, &slice));
                continue;
            }
        }
        auto cache = static_cast<CUDABackend*>(backend())->getCache();
        auto tempTensor = cache->findCacheTensor(origin, midFormat);
        if (nullptr == tempTensor) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(origin, newTensor.get());
            TensorUtils::getDescribe(newTensor.get())->dimensionFormat = midFormat;
            newTensor->buffer().type = origin->getType();
            TensorUtils::setLinearLayout(newTensor.get());
            mTempInput.insert(std::make_pair(origin, newTensor.get()));
            auto res = backend()->onAcquireBuffer(newTensor.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
            tempTensor = newTensor.get();
            TensorUtils::getDescribe(tempTensor)->useCount = TensorUtils::getDescribe(origin)->useCount;
            cache->pushCacheTensor(newTensor, origin, midFormat);
        }
        if (--TensorUtils::getDescribe(tempTensor)->useCount == 0) {
            forRelease.emplace_back(tempTensor);
        }
        mTempInputCopy.emplace_back(std::make_pair(tempTensor, &slice));
    }
    if(mTempInputCopy.size() > 1) {
        mFuseRaster.first = 1;
        mFuseRaster.second = mTempInputCopy.size();
        auto& slice0 = *mTempInputCopy[0].second;
        for (int i = 1; i < mTempInputCopy.size(); ++i) {
            auto& slice = *mTempInputCopy[i].second;
            if (mTempInputCopy[i].first != mTempInputCopy[0].first) {
                mFuseRaster.first = 0;
                break;
            }
            if (slice0.src.stride[0] != slice.src.stride[0] || slice0.dst.stride[0] != slice.dst.stride[0]) {
                mFuseRaster.first = 0;
                break;
            }
            if (slice0.src.stride[1] != slice.src.stride[1] || slice0.dst.stride[1] != slice.dst.stride[1]) {
                mFuseRaster.first = 0;
                break;
            }      
            if (slice0.src.stride[2] != slice.src.stride[2] || slice0.dst.stride[2] != slice.dst.stride[2]) {
                mFuseRaster.first = 0;
                break;
            }
            if (slice0.size[0] != slice.size[0] || slice0.size[1] != slice.size[1] || slice0.size[2] != slice.size[2]) {
                mFuseRaster.first = 0;
                break;
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

    for (auto t : forRelease) {
        backend()->onReleaseBuffer(t, Backend::DYNAMIC);
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
        RasterBlit(dstPtr, srcPtr, iter.second.size, iter.second.src.stride, iter.second.dst.stride, bytes * PACK_NUMBER, runtime);
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
        backend()->onCopyBuffer(iter.first, iter.second);
    }
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