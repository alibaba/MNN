//
//  CPURaster.cpp
//  MNN
//
//  Created by MNN on b'2020/04/02'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPURaster.hpp"
#include "compute/CommonOptFunction.h"
#include "CPUTensorConvert.hpp"
#include "math/Vec.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Concurrency.h"
#include "compute/ConvOpt.h"
#include "CPUMatMul.hpp"
#include "CPUUnary.hpp"
#include "core/BufferAllocator.hpp"
#include "CPUResizeCache.hpp"

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

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
    // just support : [N,C,A] => [N,A,C] or [N,A,C] => [N,C,A]
    // not support :  [N, X, Y, A] => [N, A, Y, X]
    if (region.src.stride[1] != 1 && region.size[0] > 1 && region.size[1] > 1 && region.size[2] > 1) {
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

ErrorCode CPURaster::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    auto des = TensorUtils::getDescribe(input);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(input);
    mZeroPoint = 0;
    if (outputDes->quantAttr != nullptr && outputDes->type == DataType_DT_INT8) {
#ifdef MNN_USE_SSE
        mZeroPoint = (int)outputDes->quantAttr->zero + 128;
#else
        mZeroPoint = (int)outputDes->quantAttr->zero;
#endif
    }
    mTempInput.clear();
    mFastBlit.clear();
    mCacheRegions.clear();
    mTempOutput = nullptr;
    auto midFormat = MNN_DATA_FORMAT_NCHW;
    mTempInputCopy.clear();
    mOutputPtr = output->host<void>();
    mFast = false;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    // all_srcFormat == dstFormat == NC4HW4 : Fast Exe
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mFast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output, core->pack, true)) {
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
                OpCommonUtils::turnToPackRegion(slice, newRegion, output, core->pack, true);
                mFastBlit.emplace_back(std::make_pair(slice.origin->host<void>(), std::move(newRegion)));
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
        mOutputPtr = mTempOutput->host<void>();
    }
    // input is NC4HW4 add Convert
    std::vector<Tensor*> forRelease;
    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        auto origin = slice.origin;
        if (nullptr == origin || nullptr == origin->host<void>()) {
            continue;
        }
        if (slice.mask != 0) {
            mTempInputCopy.emplace_back(std::make_pair(origin->host<void>(), &slice));
            continue;
        }
        // if tensor is not NC4HW4 or has been merged, don't need deal
        if (TensorUtils::getDescribe(origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            mTempInputCopy.emplace_back(std::make_pair(origin->host<void>(), &slice));
            continue;
        }
        // if NC4HW4's C%4 == 0, change convert to transpose and fuse it
        if (origin->batch() == 1 && origin->channel() % core->pack == 0) {
            int channel = origin->channel();
            int area = 1;
            // conv3d/pool3d will has 5 dims, area = depth * width * height, otherwise area = width * height
            for (int d = 2; d < origin->dimensions(); d++) {
                area *= origin->length(d);
            }
            Tensor::InsideDescribe::Region regionTmp;
            regionTmp.src.offset = 0;
            regionTmp.src.stride[0] = area * core->pack;
            regionTmp.src.stride[1] = 1;
            regionTmp.src.stride[2] = core->pack;
            regionTmp.dst.offset = 0;
            regionTmp.dst.stride[0] = area * core->pack;
            regionTmp.dst.stride[1] = area;
            regionTmp.dst.stride[2] = 1;
            regionTmp.size[0] = channel / core->pack;
            regionTmp.size[1] = core->pack;
            regionTmp.size[2] = area;
            regionTmp.origin = slice.origin;
            bool merge = TensorUtils::fuseRegion(regionTmp, slice);
            if (merge) {
                // cache the merged tensor
                slice.mask = 1;
                mTempInputCopy.emplace_back(std::make_pair(origin->host<void>(), &slice));
                continue;
            }
        }
        auto cache = static_cast<CPUBackend*>(backend())->getCache();
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
        mTempInputCopy.emplace_back(std::make_pair(tempTensor->host<void>(), &slice));
    }
    for (auto t : forRelease) {
        backend()->onReleaseBuffer(t, Backend::DYNAMIC);
    }
    if (nullptr != mTempOutput) {
        backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }
    auto threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    if (mTempInputCopy.size() == 1 && threadNumber > 1) {
        // Split to multi region
        auto region = mTempInputCopy[0].second;
        const int thredHold = 100;//TODO: Find better way to determine it
        if (region->size[0] * region->size[1] * region->size[2] < thredHold) {
            return NO_ERROR;
        }
        auto ptr = mTempInputCopy[0].first;
        int pos = -1;
        for (int i=0; i<3; ++i) {
            if (region->size[i] > 1) {
                pos = i;
                break;
            }
        }
        if (-1 == pos) {
            // Don't need divide
            return NO_ERROR;
        }
        mTempInputCopy.clear();
        int divSize = UP_DIV(region->size[pos], threadNumber);
        mCacheRegions.resize(threadNumber);
        for (int i=0; i<threadNumber; ++i) {
            int sta = i * divSize;
            int fin = sta + divSize;
            fin = std::min(fin, region->size[pos]);
            if (fin <= sta) {
                break;
            }
            for (int v=0; v<3; ++v) {
                mCacheRegions[i].src.stride[v] = region->src.stride[v];
                mCacheRegions[i].dst.stride[v] = region->dst.stride[v];
            }
            int curSize = fin - sta;
            for (int v=0; v<pos; ++v) {
                mCacheRegions[i].size[v] = region->size[v];
            }
            mCacheRegions[i].size[pos] = curSize;
            mCacheRegions[i].src.offset = region->src.offset + sta * region->src.stride[pos];
            mCacheRegions[i].dst.offset = region->dst.offset + sta * region->dst.stride[pos];
            for (int v=pos+1; v<3; ++v) {
                mCacheRegions[i].size[v] = region->size[v];
            }
            mTempInputCopy.emplace_back(std::make_pair(ptr, mCacheRegions.data() + i));
        }
    }
    return NO_ERROR;
}
static void _transpose4Bit(int32_t* dstO, const int32_t* srcO, const Tensor::InsideDescribe::Region& region) {
    int dims[4], keepDim = -1;
    for (int i = 0; i < 3; i++) {
        if (region.src.stride[i] == 1 && region.size[i] != 1) {
            dims[1] = region.size[i];
            dims[3] = region.dst.stride[i];
        }else if (region.dst.stride[i] == 1 && region.size[i] != 1) {
            dims[0] = region.size[i];
            dims[2] = region.src.stride[i];
        } else {
            keepDim = i;
        }
    }
    for (int z=0; z<region.size[keepDim]; ++z) {
        auto srcZ = srcO + region.src.stride[keepDim] * z;
        auto dstZ = dstO + region.dst.stride[keepDim] * z;
        MNNTranspose32Bit(dstZ, srcZ, dims);
    }
}
typedef void (*BlitProc)(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds);

static void _4BitcopyWithStride(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint32_t*)srcO;
    auto dst = (uint32_t*)dstO;
    for (int i=0; i<size; ++i) {
        *dst = *src;
        src+=stride;
        dst+=ds;
    }
}

static void _2BitcopyWithStride(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint16_t*)srcO;
    auto dst = (uint16_t*)dstO;
    for (int i=0; i<size; ++i) {
        *dst = *src;
        src+=stride;
        dst+=ds;
    }
}

static void _1BitcopyWithStride(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint8_t*)srcO;
    auto dst = (uint8_t*)dstO;
    for (int i=0; i<size; ++i) {
        *dst = *src;
        src+=stride;
        dst+=ds;
    }
}
static void _4BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (float*)srcO;
    auto dst = (float*)dstO;
    for (int i=0; i<size; ++i) {
        Vec4::save(dst, Vec4::load(src));
        src+= (4 * stride);
        dst+= (4 * ds);
    }
}

static void _2BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint64_t*)srcO;
    auto dst = (uint64_t*)dstO;
    for (int i=0; i<size; ++i) {
        *dst = *src;
        src+=stride;
        dst+=ds;
    }
}

void CPURaster::executeFaster(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) const {
    auto input = inputs[0];
    auto output = outputs[0];
    auto bytes = CPUBackend::getBytes(backend(), output);
    auto core = static_cast<const CPUBackend*>(backend())->functions();
    auto threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    if (mNeedZero) {
        ::memset(output->host<void>(), mZeroPoint, static_cast<CPUBackend*>(backend())->getTensorSize(output) * bytes);
    }
    auto byteC4 = bytes * core->pack;
    auto C4proc = _4BitcopyWithStride;
    switch (byteC4) {
        case 16:
            C4proc = _4BitcopyWithStrideC4;
            break;
        case 8:
            C4proc = _2BitcopyWithStrideC4;
            break;
        case 4:
            C4proc = _4BitcopyWithStride;
            break;
        default:
            C4proc = core->MNNSelectBlitFunction(byteC4);
            break;
    }
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        for (int u=(int)tId; u<mFastBlit.size(); u+=threadNum) {
            auto& iter = mFastBlit[u];
            auto& slice = iter.second;
            //Offset use byte
            auto srcPtr = (uint8_t*)iter.first + slice.src.offset * bytes;
            auto dstPtr = (uint8_t*)mOutputPtr + slice.dst.offset * bytes;
            if (slice.src.stride[1] == slice.size[2] && slice.dst.stride[1] == slice.size[2] && slice.src.stride[2] == 1) {
                for (int z=0; z<slice.size[0]; ++z) {
                    auto srcZ = srcPtr + z * slice.src.stride[0] * byteC4;
                    auto dstZ = dstPtr + z * slice.dst.stride[0] * byteC4;
                    ::memcpy(dstZ, srcZ, slice.size[1] * slice.src.stride[1] * byteC4);
                }
                continue;
            }
            if (1 == slice.src.stride[2] && 1 == slice.dst.stride[2]) {
                for (int z=0; z<slice.size[0]; ++z) {
                    auto srcZ = srcPtr + z * slice.src.stride[0] * byteC4;
                    auto dstZ = dstPtr + z * slice.dst.stride[0] * byteC4;
                    for (int y=0; y<slice.size[1]; ++y) {
                        auto srcY = srcZ + y * slice.src.stride[1] * byteC4;
                        auto dstY = dstZ + y * slice.dst.stride[1] * byteC4;
                        ::memcpy(dstY, srcY, slice.size[2] * byteC4);
                    }
                }
                continue;
            }
            for (int z=0; z<slice.size[0]; ++z) {
                auto srcZ = srcPtr + z * slice.src.stride[0] * byteC4;
                auto dstZ = dstPtr + z * slice.dst.stride[0] * byteC4;
                for (int y=0; y<slice.size[1]; ++y) {
                    auto srcY = srcZ + y * slice.src.stride[1] * byteC4;
                    auto dstY = dstZ + y * slice.dst.stride[1] * byteC4;
                    C4proc(dstY, srcY, slice.size[2], slice.src.stride[2], slice.dst.stride[2]);
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
}

static BlitProc _selectUnitProc(int bytes) {
    auto proc = _1BitcopyWithStride;
    switch (bytes) {
        case 4:
            proc = _4BitcopyWithStride;
            break;
        case 2:
            proc = _2BitcopyWithStride;
            break;
        case 1:
            proc = _1BitcopyWithStride;
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    return proc;
}
static void _zero(const Tensor::InsideDescribe::Region& slice, int bytes, uint8_t* dstPtr) {
    for (int z=0; z<slice.size[0]; ++z) {
        auto dstZ = dstPtr + (z) * slice.dst.stride[0] * bytes;
        for (int y=0; y<slice.size[1]; ++y) {
            auto dstY = dstZ + y * slice.dst.stride[1] * bytes;
            ::memset(dstY, 0, slice.size[2] * bytes);
        }
    }
}
static void _blit(const Tensor::InsideDescribe::Region& slice, int bytes, const uint8_t* srcPtr, uint8_t* dstPtr, void(*proc)(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds)) {
    if (slice.src.stride[1] == slice.size[2] && slice.dst.stride[1] == slice.size[2] && slice.src.stride[2] == 1) {
        for (int z=0; z<slice.size[0]; ++z) {
            auto srcZ = srcPtr + z * slice.src.stride[0] * bytes;
            auto dstZ = dstPtr + z * slice.dst.stride[0] * bytes;
            ::memcpy(dstZ, srcZ, slice.size[1] * slice.src.stride[1] * bytes);
        }
        return;
    }
    if (_transpose(slice) && 4 == bytes) {
        _transpose4Bit((int32_t*)dstPtr, (const int32_t*)srcPtr, slice);
        return;
    }
    if (1 == slice.src.stride[2] && 1 == slice.dst.stride[2]) {
        for (int z=0; z<slice.size[0]; ++z) {
            auto srcZ = srcPtr + z * slice.src.stride[0] * bytes;
            auto dstZ = dstPtr + z * slice.dst.stride[0] * bytes;
            for (int y=0; y<slice.size[1]; ++y) {
                auto srcY = srcZ + y * slice.src.stride[1] * bytes;
                auto dstY = dstZ + y * slice.dst.stride[1] * bytes;
                ::memcpy(dstY, srcY, slice.size[2] * bytes);
            }
        }
        return;
    }
    for (int z=0; z<slice.size[0]; ++z) {
        auto srcZ = srcPtr + z * slice.src.stride[0] * bytes;
        auto dstZ = dstPtr + (z) * slice.dst.stride[0] * bytes;
        for (int y=0; y<slice.size[1]; ++y) {
            auto srcY = srcZ + y * slice.src.stride[1] * bytes;
            auto dstY = dstZ + y * slice.dst.stride[1] * bytes;
            proc(dstY, srcY, slice.size[2], slice.src.stride[2], slice.dst.stride[2]);
        }
    }
}
void CPURaster::tensorConvert(Tensor* input, Tensor* output, int bytes) {
    auto& subIb     = input->buffer();
    auto& subOb     = output->buffer();
    auto source = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
    if (subIb.dimensions <= 1 || source == dest) {
        ::memcpy(subOb.host, subIb.host, input->elementSize() * bytes);
        return;
    }
    auto tup = CPUTensorConverter::splitDimensions(subIb, source);
    int area = std::get<1>(tup), batch = std::get<0>(tup), channel = std::get<2>(tup);
    const int bitLength = bytes;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        CPUTensorConverter::convert(subIb.host, subOb.host, source, dest, batch, area, channel, bitLength, core, tId, threadNumber);
    };
    MNN_CONCURRENCY_END();
}


ErrorCode CPURaster::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mFast) {
        executeFaster(inputs, outputs);
        return NO_ERROR;
    }
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto input = inputs[0];
    auto output = outputs[0];
    auto bytes = CPUBackend::getBytes(backend(), output);
    auto outputEleSize = static_cast<CPUBackend*>(backend())->getTensorSize(output);
    auto threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    if (mSingleConvert > 0) {
        auto realInput = TensorUtils::getDescribe(input)->regions[0].origin;
        int srcBatch = 1, srcChannel = 1, srcArea = 1;
        getBatchChannelArea(realInput, srcBatch, srcChannel, srcArea);
        auto sourceFormat = TensorUtils::getDescribe(realInput)->dimensionFormat;
        auto destFormat = TensorUtils::getDescribe(output)->dimensionFormat;
        auto channelC4 = UP_DIV(srcChannel, core->pack);
        int batchStrideC4 = channelC4 * core->pack * srcArea * bytes;
        int batchStride = srcChannel * srcArea * bytes;
        int inputBatchStride = batchStride;
        int outputBatchStride = batchStride;
        if (MNN_DATA_FORMAT_NC4HW4 == sourceFormat) {
            if (realInput->dimensions() <= 1) {
                ::memcpy(output->host<uint8_t>(), realInput->host<uint8_t>(), realInput->elementSize() * bytes);
                return NO_ERROR;
            }
            inputBatchStride = batchStrideC4;
            if (2 == mSingleConvert) {
                destFormat = MNN_DATA_FORMAT_NHWC;
            }
        }
        if (MNN_DATA_FORMAT_NC4HW4 == destFormat) {
            if (output->dimensions() <= 1) {
                ::memcpy(output->host<uint8_t>(), realInput->host<uint8_t>(), realInput->elementSize() * bytes);
                return NO_ERROR;
            }
            outputBatchStride = batchStrideC4;
            if (2 == mSingleConvert) {
                sourceFormat = MNN_DATA_FORMAT_NHWC;
            }
        }
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(realInput->host<uint8_t>(), output->host<uint8_t>(), sourceFormat, destFormat, srcBatch, srcArea, srcChannel, bytes, core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    if (mNeedZero) {
        if (mTempOutput == nullptr) {
            ::memset(output->host<void>(), mZeroPoint, outputEleSize * bytes);
        } else {
            ::memset(mTempOutput->host<void>(), mZeroPoint, mTempOutput->elementSize() * bytes);
        }
    }
    for (auto& iter : mTempInput) {
        tensorConvert(iter.first, iter.second, bytes);
    }
    auto proc = _selectUnitProc(bytes);
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        for (int u=tId; u<mTempInputCopy.size(); u+=threadNum) {
            auto& iter = mTempInputCopy[u];
            auto& slice = *(iter.second);
            auto srcPtr = (uint8_t*)iter.first + slice.src.offset * bytes;
            auto dstPtr = (uint8_t*)mOutputPtr + slice.dst.offset * bytes;
            _blit(slice, bytes, srcPtr, dstPtr, proc);
        }
    }
    MNN_CONCURRENCY_END();
    if (nullptr != mTempOutput) {
        tensorConvert(mTempOutput.get(), output, bytes);
    }
    return NO_ERROR;
}
class CPULoop : public Execution {
public:
    struct ThreadContainer {
        std::vector<std::shared_ptr<Execution>> exe;
        std::vector<uint8_t*> stackPtr;
    };
    CPULoop(Backend* bn, const LoopParam* loop) : Execution(bn) {
        // The LoopParam is created by geometry, won't be released
        mLoop = loop;
        mStack.resize(loop->tensorNumber());
        int numberThread = mLoop->parallel() ? static_cast<CPUBackend*>(backend())->threadNumber() : 1;
        mContainer.resize(numberThread);
        for (int i=0; i<numberThread; ++i) {
            mContainer[i].stackPtr.resize(mLoop->tensorNumber());
            mContainer[i].exe.resize(mLoop->commands()->size());
        }
    }
    virtual ~ CPULoop() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        int inputIndexSize = mLoop->inputIndexes()->size();
        MNN_ASSERT(inputIndexSize == inputs.size());
        for (int i=0; i<inputIndexSize; ++i) {
            mStack[mLoop->inputIndexes()->data()[i]] = inputs[i];
        }
        int outputIndexSize = mLoop->outputIndexes()->size();
        MNN_ASSERT(outputIndexSize == outputs.size());
        for (int i=0; i<outputIndexSize; ++i) {
            mStack[mLoop->outputIndexes()->data()[i]] = outputs[i];
        }
        int numberThread = mLoop->parallel() ? static_cast<CPUBackend*>(backend())->threadNumber() : 1;
        mMaxCacheSize = 0;
        auto bytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
        for (int i=0; i<mLoop->commands()->size(); ++i) {
            auto cmd = mLoop->commands()->GetAs<RegionCommand>(i);
            auto op = cmd->op();
            if (OpType_UnaryOp == op->type()) {
                if (nullptr != op->main_as_UnaryOp()) {
                    auto view0 = cmd->view()->GetAs<View>(0);
                    auto view1 = cmd->view()->GetAs<View>(1);
                    MNN_ASSERT(view0->stride()->data()[2] == 1);
                    if (view1->stride()->data()[2] != 1) {
                        mMaxCacheSize = std::max(mMaxCacheSize, cmd->size()->data()[2] * bytes);
                    }
                }
                continue;
            }
            if (OpType_BinaryOp == op->type()) {
                auto view0 = cmd->view()->GetAs<View>(0);
                auto view1 = cmd->view()->GetAs<View>(1);
                auto view2 = cmd->view()->GetAs<View>(2);
                MNN_ASSERT(view0->stride()->data()[2] == 1);
                if (view1->stride()->data()[2] != 1 || view2->stride()->data()[2] != 1) {
                    mMaxCacheSize = std::max(mMaxCacheSize, 2 * cmd->size()->data()[2] * bytes);
                }
                continue;
            }
            if (OpType_MatMul == op->type()) {
                bool transposeC = true;
                int e = cmd->size()->data()[0];
                int l = cmd->size()->data()[1];
                int h = cmd->size()->data()[2];
                std::shared_ptr<Tensor> A, B, C, Bias;
                C.reset(Tensor::createDevice<float>({e, h}));
                if (op->main_as_MatMul()->transposeA()) {
                    A.reset(Tensor::createDevice<float>({l, e}));
                } else {
                    A.reset(Tensor::createDevice<float>({e, l}));
                }
                if (op->main_as_MatMul()->transposeB()) {
                    B.reset(Tensor::createDevice<float>({h, l}));
                } else {
                    B.reset(Tensor::createDevice<float>({l, h}));
                }
                auto view = cmd->view()->GetAs<View>(0);
                if (view->stride()->data()[0] == 1) {
                    transposeC = false;
                }
                std::vector<Tensor*> inputs, outputs;
                if (cmd->indexes()->size() > 3) {
                    Bias.reset(Tensor::createDevice<float>({h}));
                    inputs = {A.get(), B.get(), Bias.get()};
                } else {
                    inputs = {A.get(), B.get()};
                }
                outputs = {C.get()};
                auto bufferPool = static_cast<CPUBackend*>(backend())->getBufferAllocator();
                auto code = NO_ERROR;
                if (numberThread > 1) {
                    bufferPool->barrierBegin();
                }
                for (int v=0; v<numberThread; ++v) {
                    if (numberThread > 1) {
                        bufferPool->beginGroup();
                    }
                    do {
                        // If not loop parallel, parallel inside
                        bool needParallel = numberThread == 1;
                        mContainer[v].exe[i].reset(new CPUMatMul(backend(), op->main_as_MatMul()->transposeA(),  op->main_as_MatMul()->transposeB(), transposeC, needParallel));
                        if (nullptr == mContainer[v].exe[i]) {
                            code = OUT_OF_MEMORY;
                            break;
                        }
                        code = mContainer[v].exe[i]->onResize(inputs, outputs);
                    } while (false);
                    if (numberThread > 1) {
                        bufferPool->endGroup();
                    }
                    if (NO_ERROR != code) {
                        break;
                    }
                }
                if (numberThread > 1) {
                    bufferPool->barrierEnd();
                }
                if (NO_ERROR != code) {
                    return code;
                }
                continue;
            }
        }
        auto threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
        if (mMaxCacheSize > 0) {
            auto buffer = static_cast<CPUBackend*>(backend())->getBufferAllocator()->alloc(threadNumber * mMaxCacheSize);
            if (nullptr == buffer.first) {
                return OUT_OF_MEMORY;
            }
            mCacheBuffer = (uint8_t*)buffer.first + buffer.second;
            static_cast<CPUBackend*>(backend())->getBufferAllocator()->free(buffer);
        }
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor *> &originInputs, const std::vector<Tensor *> &originOutputs) override {
        auto precision = static_cast<CPUBackend*>(backend())->precisionMode();
        auto threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();

        if (1 == mLoop->commands()->size()) {
            auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
            auto op = cmd->op();
            if (OpType_UnaryOp == op->type() && nullptr == op->main()) {
                // For Gather / Single Unary
                auto index0 = cmd->iterIndexes()->data()[0];
                auto index1 = cmd->iterIndexes()->data()[1];
                int32_t iter = 0;
                int32_t* iter0 = &iter;
                int32_t* iter1 = &iter;
                int32_t iter0Stride = 0;
                int32_t iter1Stride = 0;
                if (index0 >= 0) {
                    iter0 = originInputs[index0]->host<int32_t>();
                    iter0Stride = 1;
                }
                if (index1 >= 0) {
                    iter1 = originInputs[index1]->host<int32_t>();
                    iter1Stride = 1;
                }
                Tensor::InsideDescribe::Region reg;
                auto srcView = cmd->view()->GetAs<View>(1);
                auto dstView = cmd->view()->GetAs<View>(0);
                ::memcpy(reg.size, cmd->size()->data(), 3 * sizeof(int32_t));
                ::memcpy(reg.src.stride, srcView->stride()->data(), 3 * sizeof(int32_t));
                ::memcpy(reg.dst.stride, dstView->stride()->data(), 3 * sizeof(int32_t));
                auto input = mStack[cmd->indexes()->data()[1]];
                auto inputSize = input->elementSize();
                auto output = mStack[cmd->indexes()->data()[0]];
                auto bytes = input->getType().bytes();
                if (halide_type_float == input->getType().code) {
                    bytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
                }
                auto proc = _selectUnitProc(bytes);
                auto step0 = cmd->steps()->data()[0];
                auto step1 = cmd->steps()->data()[1];
                auto loopNumber = mLoop->loopNumber();
                for (; iter<loopNumber; ++iter) {
                    auto srcIter = *(iter1 + iter1Stride * iter);
                    auto dstIter = *(iter0 + iter0Stride * iter);
                    auto srcOffset = srcIter * step1 + srcView->offset();
                    auto dstOffset = dstIter * step0 + dstView->offset();
                    if (srcOffset >= 0 && srcOffset < inputSize) {
                        _blit(reg, bytes, input->host<uint8_t>() + bytes * srcOffset, output->host<uint8_t>() + bytes * dstOffset, proc);
                    } else {
                        _zero(reg, bytes, output->host<uint8_t>() + bytes * dstOffset);
                    }
                }
                return NO_ERROR;
            }
        }
        auto bytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
        auto func = [&](int iter, int tId) {
            auto blit = _selectUnitProc(bytes);
            for (int index=0; index<mLoop->commands()->size(); ++index) {
                auto cmd = mLoop->commands()->GetAs<RegionCommand>(index);
                auto op = cmd->op();
                int size = cmd->iterIndexes()->size();
                for (int v=0; v<size; ++v) {
                    auto tensorIndex = cmd->indexes()->data()[v];
                    auto tensor = mStack[tensorIndex];
                    auto iterIndex = cmd->iterIndexes()->data()[v];
                    auto offset = iter;
                    if (iterIndex >= 0) {
                        offset = mStack[iterIndex]->host<int32_t>()[iter];
                    }
                    auto view = cmd->view()->GetAs<View>(v);
                    offset = offset * cmd->steps()->data()[v] + view->offset();
                    mContainer[tId].stackPtr[tensorIndex] = tensor->host<uint8_t>() + offset * bytes;
                    MNN_ASSERT(nullptr != tensor->host<uint8_t>());
                }
                if (OpType_UnaryOp == op->type()) {
                    auto src = (uint8_t*)mContainer[tId].stackPtr[cmd->indexes()->data()[1]];
                    auto dst = (uint8_t*)mContainer[tId].stackPtr[cmd->indexes()->data()[0]];
                    if (nullptr == op->main()) {
                        // Copy
                        Tensor::InsideDescribe::Region reg;
                        auto srcView = cmd->view()->GetAs<View>(1);
                        auto dstView = cmd->view()->GetAs<View>(0);
                        ::memcpy(reg.size, cmd->size()->data(), 3 * sizeof(int32_t));
                        ::memcpy(reg.src.stride, srcView->stride()->data(), 3 * sizeof(int32_t));
                        ::memcpy(reg.dst.stride, dstView->stride()->data(), 3 * sizeof(int32_t));
                        auto proc = _selectUnitProc(bytes);
                        auto step0 = cmd->steps()->data()[0];
                        auto step1 = cmd->steps()->data()[1];
                        auto loopNumber = mLoop->loopNumber();
                        _blit(reg, bytes, (const uint8_t*)src, (uint8_t*)dst, proc);
                        continue;
                    }
                    auto proc = static_cast<CPUBackend*>(backend())->functions()->MNNSelectUnaryFunctionForFloat(op->main_as_UnaryOp()->opType(), static_cast<CPUBackend*>(backend())->precisionMode());
                    auto lastS = cmd->size()->data()[2];
                    if (lastS == 1 || cmd->view()->GetAs<View>(1)->stride()->data()[2] == 1) {
                        for (int z=0; z<cmd->size()->data()[0]; ++z) {
                            auto srcZ = src + z * cmd->view()->GetAs<View>(1)->stride()->data()[0] * bytes;
                            auto dstZ = dst + z * cmd->view()->GetAs<View>(0)->stride()->data()[0] * bytes;
                            for (int y=0; y<cmd->size()->data()[1]; ++y) {
                                auto srcY = srcZ + y * cmd->view()->GetAs<View>(1)->stride()->data()[1] * bytes;
                                auto dstY = dstZ + y * cmd->view()->GetAs<View>(0)->stride()->data()[1] * bytes;
                                proc(dstY, srcY, lastS);
                            }
                        }
                    } else {
                        // Blit to cache
                        auto srcCache = mCacheBuffer + mMaxCacheSize * tId;
                        for (int z=0; z<cmd->size()->data()[0]; ++z) {
                            auto srcZ = src + z * cmd->view()->GetAs<View>(1)->stride()->data()[0] * bytes;
                            auto dstZ = dst + z * cmd->view()->GetAs<View>(0)->stride()->data()[0] * bytes;
                            for (int y=0; y<cmd->size()->data()[1]; ++y) {
                                auto srcY = srcZ + y * cmd->view()->GetAs<View>(1)->stride()->data()[1] * bytes;
                                auto dstY = dstZ + y * cmd->view()->GetAs<View>(0)->stride()->data()[1] * bytes;
                                blit(srcCache, srcY, lastS, cmd->view()->GetAs<View>(1)->stride()->data()[2], 1);
                                proc(dstY, srcCache, lastS);
                            }
                        }
                    }
                    continue;
                }
                if (OpType_MatMul == op->type()) {
                    const float* APtr = nullptr;
                    const float* BPtr = nullptr;
                    const float* BiasPtr = nullptr;
                    float* CPtr = nullptr;
                    auto exe = static_cast<CPUMatMul*>(mContainer[tId].exe[index].get());
                    APtr = (const float*)mContainer[tId].stackPtr[cmd->indexes()->data()[1]];
                    BPtr = (const float*)mContainer[tId].stackPtr[cmd->indexes()->data()[2]];
                    CPtr = (float*)mContainer[tId].stackPtr[cmd->indexes()->data()[0]];
                    if (size > 3) {
                        BiasPtr = (const float*)mContainer[tId].stackPtr[cmd->indexes()->data()[3]];
                    }
                    exe->execute(APtr, BPtr, CPtr, BiasPtr);
                    continue;
                }
                if (OpType_BinaryOp == op->type()) {
                    auto src0 = mContainer[tId].stackPtr[cmd->indexes()->data()[1]];
                    auto src1 = mContainer[tId].stackPtr[cmd->indexes()->data()[2]];
                    auto dst = mContainer[tId].stackPtr[cmd->indexes()->data()[0]];
                    auto proc = static_cast<CPUBackend*>(backend())->functions()->MNNSelectBinaryFunctionForFloat(op->main_as_BinaryOp()->opType());
                    auto lastS = cmd->size()->data()[2];
                    auto stride0 = cmd->view()->GetAs<View>(0)->stride()->data();
                    auto stride1 = cmd->view()->GetAs<View>(1)->stride()->data();
                    auto stride2 = cmd->view()->GetAs<View>(2)->stride()->data();
                    MNN_ASSERT(stride0[2] == 1);
                    if (cmd->size()->data()[2] == 1 || (stride1[2] == 1 && stride2[2] == 1)) {
                        for (int z=0; z<cmd->size()->data()[0]; ++z) {
                            auto src0Z = src0 + z * stride1[0] * bytes;
                            auto src1Z = src1 + z * stride2[0] * bytes;
                            auto dstZ = dst + z * stride0[0] * bytes;
                            for (int y=0; y<cmd->size()->data()[1]; ++y) {
                                auto src0Y = src0Z + y * stride1[1] * bytes;
                                auto src1Y = src1Z + y * stride2[1] * bytes;
                                auto dstY = dstZ + y * stride0[1] * bytes;
                                proc(dstY, src0Y, src1Y, cmd->size()->data()[2], -1);
                            }
                        }
                    } else {
                        auto cache0 = mCacheBuffer + mMaxCacheSize * tId;
                        auto cache1 = cache0 + cmd->size()->data()[2] * bytes;
                        for (int z=0; z<cmd->size()->data()[0]; ++z) {
                            auto src0Z = src0 + z * stride1[0] * bytes;
                            auto src1Z = src1 + z * stride2[0] * bytes;
                            auto dstZ = dst + z * stride0[0] * bytes;
                            for (int y=0; y<cmd->size()->data()[1]; ++y) {
                                auto src0Y = src0Z + y * stride1[1] * bytes;
                                auto src1Y = src1Z + y * stride2[1] * bytes;
                                auto dstY = dstZ + y * stride0[1] * bytes;
                                blit(cache0, src0Y, cmd->size()->data()[2], stride1[2], 1);
                                blit(cache1, src1Y, cmd->size()->data()[2], stride2[2], 1);
                                proc(dstY, cache0, cache1, cmd->size()->data()[2], -1);
                            }
                        }
                    }
                }
            }
        };
        if (mLoop->parallel()) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                for (int iter=tId; iter < mLoop->loopNumber(); iter+=threadNumber) {
                    func(iter, tId);
                }
            }
            MNN_CONCURRENCY_END();
        } else {
            for (int iter=0; iter < mLoop->loopNumber(); ++iter) {
                func(iter, 0);
            }
        }
        return NO_ERROR;
    }
private:
    const LoopParam* mLoop;
    std::vector<Tensor*> mStack;
    std::vector<ThreadContainer> mContainer;
    uint8_t* mCacheBuffer = nullptr;
    int mMaxCacheSize = 0;
};

class CPURasterFactory : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        if (op->type() == OpType_While) {
            if (op->main_type() != OpParameter_LoopParam) {
                return nullptr;
            }
            return new CPULoop(backend, op->main_as_LoopParam());
        }
        return new CPURaster(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPURasterFactory, OpType_Raster);
REGISTER_CPU_OP_CREATOR(CPURasterFactory, OpType_While);

}
