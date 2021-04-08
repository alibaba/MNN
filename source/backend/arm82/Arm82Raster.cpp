//
//  Arm82Raster.cpp
//  MNN
//
//  Created by MNN on 2020/5/25.
//  Copyright © 2018 Alibaba. All rights reserved.
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include "Arm82Raster.hpp"
#include "math/Vec.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Concurrency.h"
#include "Arm82Backend.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

ErrorCode Arm82Raster::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    auto des = TensorUtils::getDescribe(input);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(input);
    mTempInput.clear();
    mFastBlit.clear();
    mTempOutput = nullptr;
    auto midFormat = MNN_DATA_FORMAT_NCHW;
    mTempInputCopy.clear();
    mOutputPtr = output->host<void>();
    mFast = false;
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mFast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output, 8)) {
                mFast = false;
                break;
            }
        }
        //FUNC_PRINT(mFast);
        if (mFast) {
            for (int i=0; i< des->regions.size(); ++i) {
                auto& slice = des->regions[i];
                if (slice.origin == nullptr) {
                    continue;
                }
                Tensor::InsideDescribe::Region newRegion;
                OpCommonUtils::turnToPackRegion(slice, newRegion, output, 8);
                mFastBlit.emplace_back(std::make_pair(slice.origin->host<void>(), std::move(newRegion)));
            }
//            FUNC_PRINT(1);
            return NO_ERROR;
        }
    }

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
        Vec4::save(dst + 4, Vec4::load(src + 4));
        src+= (8 * stride);
        dst+= (8 * ds);
    }
}

static void _2BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint8_t*)srcO;
    auto dst = (uint8_t*)dstO;
    for (int i=0; i<size; ++i) {
        Vec4::save((float*)dst, Vec4::load((float*)src));
        src+= 16 * stride;
        dst+= 16 * ds;
    }
}

static void _1BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint64_t*)srcO;
    auto dst = (uint64_t*)dstO;
    for (int i=0; i<size; ++i) {
        *dst = *src;
        src+=stride;
        dst+=ds;
    }
}

ErrorCode Arm82Raster::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto bytes = input->getType().bytes();
    if (input->getType().code == halide_type_float) {
        bytes = 2;
    }
    auto threadNum = static_cast<Arm82Backend*>(backend())->numberThread();
    if (mNeedZero) {
        auto size          = bytes;
        const int dimensions = input->dimensions();
        for (int i = 0; i < dimensions; i++) {
            int currentDimSize = input->length(i);
            if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
                currentDimSize = ALIGN_UP8(currentDimSize);
            }
            size *= currentDimSize;
        }
        if (mTempOutput == nullptr) {
            ::memset(output->host<void>(), 0, size);
        } else {
            ::memset(mTempOutput->host<void>(), 0, size);
        }
    }
    if (mFast) {
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
        auto byteC4 = bytes * 8;

        for (int i = 0; i < mFastBlit.size(); i++) {
            auto& iter = mFastBlit[i];
            auto& slice = iter.second;

            //Offset use byte
            auto srcPtr = (uint8_t*)iter.first + slice.src.offset * bytes;
            auto dstPtr = (uint8_t*)mOutputPtr + slice.dst.offset * bytes;
            if (slice.src.stride[1] == slice.size[2] && slice.dst.stride[1] == slice.size[2] && slice.src.stride[2] == 1) {
                int subPatch = (slice.size[1] * slice.src.stride[1] * byteC4) / threadNum; 
                int extraPatch = slice.size[1] * slice.src.stride[1] * byteC4 - subPatch * threadNum;
                MNN_CONCURRENCY_BEGIN(tId, threadNum) {
                    for (int z = 0; z < slice.size[0]; ++z) {
                        auto srcZ = srcPtr + subPatch * tId + z * slice.src.stride[0] * byteC4;
                        auto dstZ = dstPtr + subPatch * tId + z * slice.dst.stride[0] * byteC4;
                        ::memcpy(dstZ, srcZ, subPatch);
                    }
                }
                MNN_CONCURRENCY_END();
                if (extraPatch > 0) {
                    for (int z = 0; z < slice.size[0]; ++z) {
                        auto srcZ = srcPtr + subPatch * threadNum + z * slice.src.stride[0] * byteC4;
                        auto dstZ = dstPtr + subPatch * threadNum + z * slice.dst.stride[0] * byteC4;
                        ::memcpy(dstZ, srcZ, extraPatch);
                    }
                }
                continue;
            }

            if (1 == slice.src.stride[2] && 1 == slice.dst.stride[2]) {
                for (int z=0; z<slice.size[0]; ++z) {
                    auto srcZ = srcPtr + z * slice.src.stride[0] * byteC4;
                    auto dstZ = dstPtr + z * slice.dst.stride[0] * byteC4;
                    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
                        for (int y = tId; y < slice.size[1]; y += threadNum) {
                            auto srcY = srcZ + y * slice.src.stride[1] * byteC4;
                            auto dstY = dstZ + y * slice.dst.stride[1] * byteC4;
                            ::memcpy(dstY, srcY, slice.size[2] * byteC4);
                        } 
                    }
                    MNN_CONCURRENCY_END();                       
                }
                continue;
            }

            for (int z = 0; z < slice.size[0]; ++z) {
                auto srcZ = srcPtr + z * slice.src.stride[0] * byteC4;
                auto dstZ = dstPtr + z * slice.dst.stride[0] * byteC4;
                MNN_CONCURRENCY_BEGIN(tId, threadNum) {
                    for (int y = tId; y < slice.size[1]; y += threadNum) {
                        auto srcY = srcZ + y * slice.src.stride[1] * byteC4;
                        auto dstY = dstZ + y * slice.dst.stride[1] * byteC4;
                        C4proc(dstY, srcY, slice.size[2], slice.src.stride[2], slice.dst.stride[2]);
                    }
                }
                MNN_CONCURRENCY_END();
            }
        }
        return NO_ERROR;
    }

    for (auto& iter : mTempInput) {
        backend()->onCopyBuffer(iter.first, iter.second.get());
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

    for (int i = 0; i < mTempInputCopy.size(); i++) {
        auto& iter = mTempInputCopy[i];
        auto& slice = *(iter.second);
        auto srcPtr = (uint8_t*)iter.first + slice.src.offset * bytes;
        auto dstPtr = (uint8_t*)mOutputPtr + slice.dst.offset * bytes;
        if (slice.src.stride[1] == slice.size[2] && slice.dst.stride[1] == slice.size[2] && slice.src.stride[2] == 1) {
            int subPatch = (slice.size[1] * slice.src.stride[1] * bytes) / threadNum;
            int extraPatch = slice.size[1] * slice.src.stride[1] * bytes - subPatch * threadNum;
            MNN_CONCURRENCY_BEGIN(tId, threadNum) {
                for (int z = 0; z < slice.size[0]; ++z) {
                    auto srcZ = srcPtr + subPatch * tId + z * slice.src.stride[0] * bytes;
                    auto dstZ = dstPtr + subPatch * tId + z * slice.dst.stride[0] * bytes;
                    ::memcpy(dstZ, srcZ, subPatch);
                }
            }
            MNN_CONCURRENCY_END();
            if (extraPatch > 0) {
                for (int z = 0; z < slice.size[0]; ++z) {
                    auto srcZ = srcPtr + subPatch * threadNum + z * slice.src.stride[0] * bytes;
                    auto dstZ = dstPtr + subPatch * threadNum + z * slice.dst.stride[0] * bytes;
                    ::memcpy(dstZ, srcZ, extraPatch);
                }
            }
            continue;
        }
        if (1 == slice.src.stride[2] && 1 == slice.dst.stride[2]) {
            for (int z = 0; z < slice.size[0]; ++z) {
                auto srcZ = srcPtr + z * slice.src.stride[0] * bytes;
                auto dstZ = dstPtr + z * slice.dst.stride[0] * bytes;
                MNN_CONCURRENCY_BEGIN(tId, threadNum) {
                    for (int y = tId; y < slice.size[1]; y += threadNum) {
                        auto srcY = srcZ + y * slice.src.stride[1] * bytes;
                        auto dstY = dstZ + y * slice.dst.stride[1] * bytes;
                        ::memcpy(dstY, srcY, slice.size[2] * bytes);
                    }
                }
                MNN_CONCURRENCY_END();
            }
            continue;
        }
        for (int z = 0; z < slice.size[0]; ++z) {
            auto srcZ = srcPtr + z * slice.src.stride[0] * bytes;
            auto dstZ = dstPtr + (z) * slice.dst.stride[0] * bytes;
            MNN_CONCURRENCY_BEGIN(tId, threadNum) {
                for (int y = tId; y < slice.size[1]; y += threadNum) {
                    auto srcY = srcZ + y * slice.src.stride[1] * bytes;
                    auto dstY = dstZ + y * slice.dst.stride[1] * bytes;
                    proc(dstY, srcY, slice.size[2], slice.src.stride[2], slice.dst.stride[2]);
                }
            }
            MNN_CONCURRENCY_END();
        }
    }

    if (nullptr != mTempOutput) {
        backend()->onCopyBuffer(mTempOutput.get(), output);
    }
    return NO_ERROR;
}

class Arm82RasterFactory : public Arm82Backend::Arm82Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new Arm82Raster(backend);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_Raster, Arm82RasterFactory);

}
#endif
