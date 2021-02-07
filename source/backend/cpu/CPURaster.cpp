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
using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {
static bool _canBlitFast(const Tensor::InsideDescribe::Region& region, const Tensor* dest) {
    return OpCommonUtils::canBlitFast(region, dest, 4);
}
static void _turnToC4Region(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& c4Region, const Tensor* dest) {
    return OpCommonUtils::turnToPackRegion(region, c4Region, dest, 4);
}
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
static bool _singleConvert(const Tensor::InsideDescribe::Region& region, const Tensor* dest) {
    // TODO, may be wrong
    if (region.offset != nullptr) {
        return false;
    }
    auto origin = region.origin;
    auto srcFormat = TensorUtils::getDescribe(origin)->dimensionFormat;
    auto dstFormat = TensorUtils::getDescribe(dest)->dimensionFormat;
    if (srcFormat == dstFormat) {
        return false;
    }
    if (0 != region.src.offset || 0 != region.dst.offset) {
        return false;
    }
    int dstBatch = 1, dstChannel = 1, dstArea = 1,
        srcBatch = 1, srcChannel = 1, srcArea = 1;
    getBatchChannelArea(origin, srcBatch, srcChannel, srcArea);
    getBatchChannelArea(dest, dstBatch, dstChannel, dstArea);
    if (dstBatch != srcBatch) {
        return false;
    }
    if (dstChannel != srcChannel) {
        return false;
    }
    if (dstArea != srcArea) {
        return false;
    }
    auto totalSize = dstBatch * dstChannel * dstArea;
    int srcSize = 1;
    int dstSize = 1;
    for (int i=0; i<3; ++i) {
        srcSize += (region.size[i] - 1) * region.src.stride[i];
        dstSize += (region.size[i] - 1) * region.dst.stride[i];
    }
    return srcSize == totalSize && dstSize == totalSize;
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

ErrorCode CPURaster::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    auto des = TensorUtils::getDescribe(input);
    MNN_ASSERT(des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(input);
    mTempInput.clear();
    mFastBlit.clear();
    mTempOutput = nullptr;
    auto midFormat = MNN_DATA_FORMAT_NCHW;
    mTempInputCopy.clear();
    mOutputPtr = output->host<void>();
    mFast = false;
    // all_srcFormat == dstFormat == NC4HW4 : Fast Exe
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mFast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
            if (!_canBlitFast(slice, output)) {
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
                _turnToC4Region(slice, newRegion, output);
                mFastBlit.emplace_back(std::make_pair(slice.origin->host<void>(), std::move(newRegion)));
            }
            return NO_ERROR;
        }
    }
    if (1 < static_cast<CPUBackend*>(backend())->threadNumber()) {
        mConverter.reset(new CPUTensorConverter(backend()));
    }
    mSingleConvert = false;
    // srcNum == 1 && srcFormat != dstFormat : Single Convert
    if (des->regions.size() == 1 && _singleConvert(des->regions[0], output)) {
        mSingleConvert = true;
        return NO_ERROR;
    }
    // input is NC4HW4 add Convert
    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        auto origin = slice.origin;
        if (TensorUtils::getDescribe(origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            continue;
        }
        // if NC4HW4's C%4 == 0, change convert to transpose and fuse it
        if (origin->batch() == 1 && origin->channel() % 4 == 0) {
            int channel = origin->channel();
            int area = origin->width() * origin->height();
            auto regionTmp = slice;
            regionTmp.src.offset = 0;
            regionTmp.src.stride[0] = area * 4;
            regionTmp.src.stride[1] = 1;
            regionTmp.src.stride[2] = 4;
            regionTmp.dst.offset = 0;
            regionTmp.dst.stride[0] = area * 4;
            regionTmp.dst.stride[1] = area;
            regionTmp.dst.stride[2] = 1;
            regionTmp.size[0] = channel / 4;
            regionTmp.size[1] = 4;
            regionTmp.size[2] = area;
            bool merge = TensorUtils::fuseRegion(regionTmp, slice);
            if (merge) {
                continue;
            }
        }
        if (mTempInput.find(origin)!=mTempInput.end()) {
            continue;
        }
        std::shared_ptr<Tensor> newTensor(new Tensor);
        TensorUtils::copyShape(origin, newTensor.get());
        TensorUtils::getDescribe(newTensor.get())->dimensionFormat = midFormat;
        newTensor->buffer().type = origin->getType();
        TensorUtils::setLinearLayout(newTensor.get());
        mTempInput.insert(std::make_pair(origin, newTensor));
    }
    // TODO optimize it
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
    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        if (nullptr == slice.origin) {
            continue;
        }
        auto iter = mTempInput.find(slice.origin);
        if (iter != mTempInput.end()) {
            mTempInputCopy.emplace_back(std::make_pair(iter->second->host<void>(), &slice));
            continue;
        }
        mTempInputCopy.emplace_back(std::make_pair(slice.origin->host<void>(), &slice));
        MNN_ASSERT(mTempInputCopy[i].first != nullptr);
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

static void _1BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint32_t*)srcO;
    auto dst = (uint32_t*)dstO;
    for (int i=0; i<size; ++i) {
        *dst = *src;
        src+=stride;
        dst+=ds;
    }
}
void CPURaster::executeFaster(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) const {
    auto input = inputs[0];
    auto output = outputs[0];
    auto bytes = input->getType().bytes();
    auto threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    if (mNeedZero) {
        ::memset(output->host<void>(), 0, output->size());
    }
    auto C4proc = _1BitcopyWithStrideC4;
    switch (bytes) {
        case 4:
            C4proc = _4BitcopyWithStrideC4;
            break;
        case 2:
            C4proc = _2BitcopyWithStrideC4;
            break;
        case 1:
            C4proc = _1BitcopyWithStrideC4;
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    auto byteC4 = bytes * 4;
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

ErrorCode CPURaster::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mFast) {
        executeFaster(inputs, outputs);
        return NO_ERROR;
    }
    auto input = inputs[0];
    auto output = outputs[0];
    auto bytes = input->getType().bytes();
    auto threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    if (mSingleConvert) {
        auto realInput = TensorUtils::getDescribe(input)->regions[0].origin;
        int srcBatch = 1, srcChannel = 1, srcArea = 1;
        getBatchChannelArea(realInput, srcBatch, srcChannel, srcArea);
        auto sourceFormat = TensorUtils::getDescribe(realInput)->dimensionFormat;
        auto destFormat = TensorUtils::getDescribe(output)->dimensionFormat;
        auto channelC4 = UP_DIV(srcChannel, 4);
        int batchStrideC4 = channelC4 * 4 * srcArea * bytes;
        int batchStride = srcChannel * srcArea * bytes;
        int inputBatchStride = batchStride;
        int outputBatchStride = batchStride;
        if (MNN_DATA_FORMAT_NC4HW4 == sourceFormat) {
            inputBatchStride = batchStrideC4;
        }
        if (MNN_DATA_FORMAT_NC4HW4 == destFormat) {
            outputBatchStride = batchStrideC4;
        }
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            for (int b=(int)tId; b<srcBatch; b+=(int)threadNum) {
                auto inputBatch = realInput->host<uint8_t>() + b * inputBatchStride;
                auto outputBatch = output->host<uint8_t>() + b * outputBatchStride;
                auto code = CPUTensorConverter::convert(inputBatch, outputBatch, sourceFormat, destFormat, 1, srcArea, srcChannel, bytes);
                if (NO_ERROR != code) {
                    MNN_ERROR("Error in CPURaster's convert\n");
                    break;
                }
            }
        };
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    if (mNeedZero) {
        if (mTempOutput == nullptr) {
            ::memset(output->host<void>(), 0, output->size());
        } else {
            ::memset(mTempOutput->host<void>(), 0, mTempOutput->size());
        }
    }
    for (auto& iter : mTempInput) {
        if (nullptr != mConverter) {
            mConverter->onExecute({iter.first}, {iter.second.get()});
        } else {
            CPUTensorConverter::convert(iter.first, iter.second.get());
        }
    }
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
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        for (int u=tId; u<mTempInputCopy.size(); u+=threadNum) {
            auto& iter = mTempInputCopy[u];
            auto& slice = *(iter.second);
            if (slice.offset != nullptr) {
                auto len = slice.offset->length(1);
                auto srcOffset = slice.offset->host<int>() + 0;
                auto dstOffset = slice.offset->host<int>() + len;
                for (int v = 0; v < len; ++v) {
                    auto srcPtr = (uint8_t*)iter.first + srcOffset[v] * bytes;
                    auto dstPtr = (uint8_t*)mOutputPtr + dstOffset[v] * bytes;
                    _blit(slice, bytes, srcPtr, dstPtr, proc);
                }
                continue;
            }
            auto srcPtr = (uint8_t*)iter.first + slice.src.offset * bytes;
            auto dstPtr = (uint8_t*)mOutputPtr + slice.dst.offset * bytes;
            _blit(slice, bytes, srcPtr, dstPtr, proc);
        }
    }
    MNN_CONCURRENCY_END();
    if (nullptr != mTempOutput) {
        if (nullptr != mConverter) {
            mConverter->onExecute({mTempOutput.get()}, {output});
        } else {
            CPUTensorConverter::convert(mTempOutput.get(), output);
        }
    }
    return NO_ERROR;
}

class CPURasterFactory : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPURaster(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPURasterFactory, OpType_Raster);

}
