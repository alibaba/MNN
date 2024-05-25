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
    if (slice0.dst.stride[0] !=0 && slice0.src.stride[0] != 0 && slice0.src.stride[0] % slice0.dst.stride[0] != 0 && slice0.dst.stride[0] % slice0.src.stride[0] != 0) {
        return false;
    }
    return true;
}

static bool _directBlitC4(const Tensor::InsideDescribe::Region& slice0, const Tensor::InsideDescribe::Region& slice1, Tensor* tensor) {
    if(tensor->dimensions() < 2) {
        return false;
    }
    if(slice0.src.stride[1] == tensor->width() && slice0.src.stride[0] == tensor->width() * tensor->height()) {
        // area pack for fast blit only
        return false;
    }
    if(slice1.src.stride[1] == tensor->width() && slice1.src.stride[0] == tensor->width() * tensor->height()) {
        // area pack for fast blit only
        return false;
    }
    if(slice0.size[1] % PACK_NUMBER != 0) {
        return false;
    }
    if(slice1.size[1] % PACK_NUMBER != 0) {
        return false;
    }
    if(slice0.dst.offset % (slice0.size[1] * slice0.size[2]) != 0) {
        return false;
    }
    if(slice1.dst.offset % (slice1.size[1] * slice1.size[2]) != 0) {
        return false;
    }
    if(slice0.src.offset % (slice0.size[1] * slice0.size[2]) != 0) {
        return false;
    }
    if(slice1.src.offset % (slice1.size[1] * slice1.size[2]) != 0) {
        return false;
    }
    if(slice0.src.stride[2] != 1 || slice0.dst.stride[2] != 1) {
        return false;
    }
    if(slice1.src.stride[2] != 1 || slice1.dst.stride[2] != 1) {
        return false;
    }
    return _equalSizeStride(slice0, slice1); 
}

static void _turnToNewRegion(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& newRegion, const int srcStep, const int dstStep) {
    newRegion.size[0] = region.size[0];
    newRegion.size[1] = region.size[2];
    newRegion.size[2] = region.size[1];

    newRegion.src.stride[0] = region.src.stride[0];
    newRegion.src.stride[1] = region.src.stride[2] * region.size[1] * srcStep;
    newRegion.src.stride[2] = region.src.stride[1] / region.size[2];

    newRegion.dst.stride[0] = region.dst.stride[0];// * dstStep;
    newRegion.dst.stride[1] = region.dst.stride[2] * region.size[1] * dstStep;
    newRegion.dst.stride[2] = region.dst.stride[1] / region.size[2];

    newRegion.src.offset = region.src.offset / region.size[2];
    newRegion.dst.offset = region.dst.offset / region.size[2];
}

ErrorCode RasterExecution::onResize(const std::vector<Tensor *> &____inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(outputs.size() == 1);
    auto input = outputs[0];
    auto output = outputs[0];
    OpCommonUtils::rasterInputReset(____inputs, outputs[0]);
    mSingleConvert.type = 0;

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
            // MNN_PRINT("%d-%d-%d-%d-%d\n", ____inputs[i]->batch(), ____inputs[i]->height(), ____inputs[i]->width(), ____inputs[i]->channel(), outputs[0]->channel());
            // MNN_PRINT("%d-%d-%d, %d-%d-%d, %d-%d-%d, %d-%d\n\n", slice.size[0], slice.size[1], slice.size[2], slice.src.stride[0], slice.src.stride[1], slice.src.stride[2], slice.dst.stride[0], slice.dst.stride[1], slice.dst.stride[2],  slice.src.offset,  slice.dst.offset);
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
	    if(!_directBlitC4(slice0, slice, output)) {
		mFast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output, pack, false, true)) {
		mFast = false;
                break;
            }
	    }
        //MNN_PRINT("raster fast:%d regionNum:%d\n\n\n", mFast, des->regions.size());
        if (mFast) {
            for (int i=0; i< des->regions.size(); ++i) {
                int srcStep = 1;
                int dstStep = 1;
	            auto& slice = des->regions[i];
                if(slice.dst.offset / (slice.size[2] * slice.size[1]) >= 1) {
                    int batchChannel = slice.dst.offset / (slice.size[1] * slice.size[2]) + 1;
                    dstStep = dstStep > batchChannel ? dstStep : batchChannel;
                }
                if(slice.src.stride[0] != 0 && slice.dst.stride[0] / slice.src.stride[0] > 1) {
                    int tmp = slice.dst.stride[0] / slice.src.stride[0];
                    dstStep = dstStep > tmp ? dstStep : tmp;
                }
                if(slice.src.offset / (slice.size[2] * slice.size[1]) >= 1) {
                    int batchChannel = slice.src.offset / (slice.size[1] * slice.size[2]) + 1;
                    srcStep = srcStep > batchChannel ? srcStep : batchChannel;
                }
                if(slice.dst.stride[0] != 0 && slice.src.stride[0] / slice.dst.stride[0] > 1) {
                    int tmp = slice.src.stride[0] / slice.dst.stride[0];
                    srcStep = srcStep > tmp ? srcStep : tmp;
                }
		        if(____inputs[i]->channel() > slice.size[1]) {
                    int tmp = ____inputs[i]->channel() / slice.size[1];
                    srcStep = srcStep > tmp ? srcStep : tmp;
		        }
		
		        if (slice.origin == nullptr) {
                    continue;
                }
                Tensor::InsideDescribe::Region newRegion;
                // [N, C, HW] --> [N, HW, C]
                _turnToNewRegion(slice, newRegion, srcStep, dstStep);
                mFastBlit.emplace_back(std::make_pair(slice.origin, std::move(newRegion)));
                // MNN_PRINT("new step %d-%d:%d-%d-%d, %d-%d-%d, %d-%d-%d, %d-%d\n\n", srcStep, dstStep, newRegion.size[0], newRegion.size[1], newRegion.size[2], newRegion.src.stride[0], newRegion.src.stride[1], newRegion.src.stride[2], newRegion.dst.stride[0], newRegion.dst.stride[1], newRegion.dst.stride[2],  newRegion.src.offset,  newRegion.dst.offset);
	        }
            return NO_ERROR;
        }
    }

    // srcNum == 1 && srcFormat != dstFormat : Single Convert
    if (des->regions.size() == 1) {
        OpCommonUtils::turnRegion2Convert(des->regions[0], output, mSingleConvert);
        if (mSingleConvert.type > 0) {
            return NO_ERROR;
        }
    }

    std::vector<Tensor*> forRelease;
    for(int i = 0; i < des->regions.size(); i++) {
        auto& slice = des->regions[i];
        auto origin = slice.origin;
        if (TensorUtils::getDescribe(origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            mTempInputCopy.emplace_back(std::make_pair(origin, &slice));
            continue;
        }
        auto cache = static_cast<CUDABackend*>(backend())->getCache();
        auto tempTensor = cache->findCacheTensor(origin, MNN_DATA_FORMAT_NCHW);
        if (nullptr == tempTensor) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(origin, newTensor.get());
            TensorUtils::getDescribe(newTensor.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            newTensor->buffer().type = origin->getType();
            TensorUtils::setLinearLayout(newTensor.get());
            // Propagate quant info if necessary
            auto des = TensorUtils::getDescribe(newTensor.get());
            auto originDes = TensorUtils::getDescribe(origin);
            if (originDes->quantAttr != nullptr) {
                des->quantAttr.reset(new QuantAttr);
                *des->quantAttr = *originDes->quantAttr;
                des->type = static_cast<CUDABackend*>(backend())->getDataType(origin);
            }

            auto res = backend()->onAcquireBuffer(newTensor.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
            tempTensor = newTensor.get();
            TensorUtils::getDescribe(tempTensor)->useCount = TensorUtils::getDescribe(origin)->useCount;
            cache->pushCacheTensor(newTensor, origin, MNN_DATA_FORMAT_NCHW);
            mTempInput.insert(std::make_pair(origin, tempTensor));
        }
        if (--TensorUtils::getDescribe(tempTensor)->useCount == 0) {
            forRelease.emplace_back(tempTensor);
        }
        mTempInputCopy.emplace_back(std::make_pair(tempTensor, &slice));
    }

    if (MNN_DATA_FORMAT_NC4HW4 == outputDes->dimensionFormat) {
        mTempOutput.reset(new Tensor);
        TensorUtils::setupTensorInfo(output, mTempOutput.get(), MNN_DATA_FORMAT_NCHW);

        // Propagate quant info if necessary
        auto des = TensorUtils::getDescribe(mTempOutput.get());
        auto originDes = TensorUtils::getDescribe(output);
        if (originDes->quantAttr != nullptr) {
            des->quantAttr.reset(new QuantAttr);
            *des->quantAttr = *originDes->quantAttr;
            des->type = static_cast<CUDABackend*>(backend())->getDataType(output);
        }

        auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mOutputPtr = mTempOutput.get();
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
            //MNN_PRINT("%d-%d-%d\n", regionSize, temp[i], temp[regionSize+i]);
        }
        //save srcOffset/dstOffset to Device
        mOffsetTensor.reset(Tensor::createDevice<int32_t>({2*regionSize}));
        backend()->onAcquireBuffer(mOffsetTensor.get(), Backend::STATIC);
        mOffset = (void *)mOffsetTensor.get()->buffer().device;
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
    auto input = outputs[0];
    auto output = outputs[0];
    auto bytes = bn->getBytes(output);
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    if (mNeedZero) {
        auto size = static_cast<CUDABackend*>(backend())->realSize(output) * bytes;
        cudaMemset((uint8_t*)output->deviceId(), 0, size);
        checkKernelErrors;
    }
    // Use mFastBlit
    for (auto& iter : mFastBlit) {
        auto srcPtr = (uint8_t*)iter.first->deviceId() + iter.second.src.offset * bytes;
        auto dstPtr = (uint8_t*)output->deviceId() + iter.second.dst.offset * bytes;
	    RasterBlit(dstPtr, srcPtr, iter.second.size, iter.second.src.stride, iter.second.dst.stride, bytes, runtime);
        checkKernelErrors;
    }
}

ErrorCode RasterExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mFast) {
        executeFaster(inputs, outputs);
	    return NO_ERROR;
    }
    auto bn = static_cast<CUDABackend*>(backend());
    auto input = outputs[0];
    auto output = outputs[0];
    auto bytes = bn->getBytes(output);
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    // MNN_PRINT("raster format:%d -> %d, addr:%p %p bytes:%d\n", TensorUtils::getDescribe(input)->dimensionFormat, \
    //     TensorUtils::getDescribe(output)->dimensionFormat, \
    //     input->deviceId(), output->deviceId(), bytes);

    if (mSingleConvert.type > 0) {
        auto realInput = TensorUtils::getDescribe(input)->regions[0].origin;
        int srcBatch = mSingleConvert.batch, srcChannel = mSingleConvert.channel, srcArea = mSingleConvert.area;
        auto sourceFormat = TensorUtils::getDescribe(realInput)->dimensionFormat;
        PackInfo pack;
        pack.inside = srcArea;
        pack.axis = srcChannel;
        pack.unit = PACK_NUMBER;
        pack.outside = srcBatch;
        if (mSingleConvert.type == 1) {
            pack.axisStride = srcArea;
            pack.insideStride = 1;
        } else if (mSingleConvert.type == 2) {
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
            checkKernelErrors;          
        } else {
            if (output->dimensions() <= 1) {
                cudaMemcpy(dstPtr, srcPtr, bn->realSize(realInput) * bytes, cudaMemcpyDeviceToDevice);
                return NO_ERROR;
            }
            PackBuffer(dstPtr, srcPtr, &pack, bytes, runtime);
            checkKernelErrors;         
        }
        return NO_ERROR;
    }

    if (mNeedZero) {
        auto size = static_cast<CUDABackend*>(backend())->realSize(mOutputPtr) * bytes;
        cudaMemset((uint8_t*)mOutputPtr->deviceId(), 0, size);
        checkKernelErrors;
    }
    for (auto& iter : mTempInput) {
        backend()->onCopyBuffer(iter.first, iter.second);
        checkKernelErrors;
    }
    //MNN_PRINT("\n%d\n", mFuseRaster.first);
    if(mFuseRaster.first > 0) {
        MNN_ASSERT(mTempInputCopy.size() == 1);
        auto& iter  = mTempInputCopy[0];
        auto& slice = *(iter.second);
        auto srcPtr = (uint8_t*)iter.first->deviceId();
        auto dstPtr = (uint8_t*)mOutputPtr->deviceId();
        //MNN_PRINT("fuseRaster:%p-%p\n", mSrcOffset, mDstOffset);

        FuseRasterBlit(dstPtr, srcPtr, slice.size, slice.src.stride, slice.dst.stride, mFuseRaster.second, mOffset, bytes, runtime, mFuseRaster.first);
        checkKernelErrors;
    } else {
        for (auto& iter : mTempInputCopy) {
            auto srcPtr = (uint8_t*)iter.first->deviceId() + iter.second->src.offset * bytes;
            auto dstPtr = (uint8_t*)mOutputPtr->deviceId() + iter.second->dst.offset * bytes;
            RasterBlit(dstPtr, srcPtr, iter.second->size, iter.second->src.stride, iter.second->dst.stride, bytes, runtime);
            checkKernelErrors;
        }
    }

    if (nullptr != mTempOutput) {
        backend()->onCopyBuffer(mTempOutput.get(), output);
        checkKernelErrors;
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
