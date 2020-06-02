//
//  Arm82Padding.cpp
//  MNN
//
//  Created by MNN on 2020/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#include "backend/arm82/Arm82Padding.hpp"
#include "backend/arm82/Arm82Backend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {

// ErrorCode memsetHelper(const Tensor *padValueTensor, Tensor *output) {
//    auto dtype     = output->getType();
//    const int size = output->elementSize();
//    if (dtype == halide_type_of<float>()) {
//        const auto padValue = padValueTensor->host<float>()[0];
//        auto ptr            = output->host<float>();
//        std::fill(ptr, ptr + size, padValue);
//    } else if (dtype == halide_type_of<int>()) {
//        const auto padValue = padValueTensor->host<int>()[0];
//        auto ptr            = output->host<int>();
//        std::fill(ptr, ptr + size, padValue);
//    } else {
//        MNN_ERROR("TODO, support other data type: %d\n", dtype.code);
//        return NOT_SUPPORT;
//    }
//    return NO_ERROR;
//}

//  refer to tflite mirrorPad
struct CacheElement {
    int start;
    int end;
};
int MirrorPadImplFp16(const Tensor *data, CacheElement *cache, Tensor *paddedData, const int *pad, int currentDim,
                  int flatIndex, int outputIndex, int offset) {
    auto dataType = data->getType();
    int bytes     = data->getType().bytes();
    // original data type is float, which means fp16 in arm82backend, divide bytes by 2
    if (dataType == halide_type_of<float>()) {
        bytes /= 2;
    }
    if (currentDim == paddedData->dimensions()) {
        if (outputIndex >= paddedData->elementSize()) {
            return outputIndex;
        }
        memcpy(paddedData->host<char>() + outputIndex * bytes, data->host<char>() + flatIndex * bytes, bytes);
        return outputIndex + 1;
    }
    const int cacheIndex = currentDim * data->elementSize() + flatIndex;
    auto &cacheEntry     = cache[cacheIndex];
    if (cacheEntry.start != -1) {
        const int size = cacheEntry.end - cacheEntry.start;
        memcpy(paddedData->host<char>() + outputIndex * bytes, paddedData->host<char>() + cacheEntry.start * bytes,
               size * bytes);
        return outputIndex + size;
    }

    cacheEntry.start     = outputIndex;
    int leftPad          = pad[2 * currentDim];
    int rightPad         = pad[2 * currentDim + 1];
    const int multiplier = data->stride(currentDim);

    for (int i = leftPad + offset - 1; i >= offset && leftPad > 0; --i, --leftPad) {
        outputIndex = MirrorPadImplFp16(data, cache, paddedData, pad, currentDim + 1, flatIndex + i * multiplier,
                                    outputIndex, offset);
    }
    const int curDimLength = data->length(currentDim);
    for (int i = 0; i < curDimLength; ++i) {
        outputIndex = MirrorPadImplFp16(data, cache, paddedData, pad, currentDim + 1, flatIndex + i * multiplier,
                                    outputIndex, offset);
    }
    for (int i = curDimLength - (1 + offset); i >= 0 && rightPad > 0; --i, --rightPad) {
        outputIndex = MirrorPadImplFp16(data, cache, paddedData, pad, currentDim + 1, flatIndex + i * multiplier,
                                    outputIndex, offset);
    }

    cacheEntry.end = outputIndex;

    return outputIndex;
}

static ErrorCode resizeImplFp16(Backend *bn, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                            Tensor *cache) {
    const int size = inputs[0]->elementSize() * inputs[0]->dimensions() * 2;
    cache->setType(DataType_DT_INT32);
    cache->buffer().dimensions = 1;
    cache->setLength(0, size);
    bool success = bn->onAcquireBuffer(cache, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    bn->onReleaseBuffer(cache, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode Arm82Padding::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mMode != PadValueMode_CONSTANT) {
        return resizeImplFp16(backend(), inputs, outputs, &mCache);
    }
    return NO_ERROR;
}

void Arm82Padding::execute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                           PadValueMode mode) {
    auto input   = inputs[0];
    auto output  = outputs[0];
    auto padding = inputs[1]->host<int32_t>();
    if (inputs.size() == 3) {
        MNN_ERROR("TODO, support 3 inputs case!");
    } else {
        ::memset(output->host<char>(), 0, output->size());
    }
    auto outputData = output->host<char>();
    auto inputData  = input->host<char>();
#define MAX_DIM 6
    MNN_ASSERT(output->dimensions() <= MAX_DIM);
    int dims[MAX_DIM];
    int oStride[MAX_DIM];
    int iStride[MAX_DIM];
    int pad[MAX_DIM];
    auto bytes    = input->getType().bytes();
    auto dataType = input->getType();
    // original data type is float, which means fp16 in arm82backend, divide bytes by 2
    if (dataType == halide_type_of<float>()) {
        bytes /= 2;
    }
    for (int i = 0; i < MAX_DIM; ++i) {
        pad[i]     = 0;
        dims[i]    = 1;
        oStride[i] = 0;
        iStride[i] = 0;
    }
    int offset = MAX_DIM - input->dimensions();
    for (int i = 0; i < input->dimensions(); ++i) {
        pad[offset + i]     = padding[2 * i];
        dims[offset + i]    = input->length(i);
        oStride[offset + i] = output->stride(i) * bytes;
        iStride[offset + i] = input->stride(i) * bytes;
    }
    for (int w = 0; w < dims[0]; ++w) {
        auto ow = outputData + (w + pad[0]) * oStride[0];
        auto sw = inputData + w * iStride[0];
#define PTR(x, y, i)                              \
    auto o##x = o##y + (x + pad[i]) * oStride[i]; \
    auto s##x = s##y + x * iStride[i];

        for (int v = 0; v < dims[1]; ++v) {
            PTR(v, w, 1);
            for (int u = 0; u < dims[2]; ++u) {
                PTR(u, v, 2);
                for (int z = 0; z < dims[3]; ++z) {
                    PTR(z, u, 3);
                    for (int y = 0; y < dims[4]; ++y) {
                        PTR(y, z, 4);
                        ::memcpy(oy + pad[5] * oStride[5], sy, iStride[4]);
                    }
                }
            }
        }
    }
#undef MAX_DIM
#undef PTR
}

ErrorCode Arm82Padding::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mMode == PadValueMode_CONSTANT) {
        execute(inputs, outputs, mMode);
    } else {
        // REFLECT or SYMMETRIC
        int offset     = mMode == PadValueMode_SYMMETRIC ? 0 : 1;
        auto cacheData = reinterpret_cast<CacheElement *>(mCache.host<char>());
        std::fill(cacheData, cacheData + mCache.elementSize() / 2, CacheElement{-1, -1});
        const int *pad  = inputs[1]->host<int32_t>();
        int outputIndex = 0;
        MirrorPadImplFp16(inputs[0], cacheData, outputs[0], pad, 0, 0, outputIndex, offset);
    }
    return NO_ERROR;
}

ErrorCode Arm82PaddingPacked::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto padding    = inputs[1];
    auto paddingPtr = padding->host<int32_t>();
    if (paddingPtr[2] != 0 || paddingPtr[3] != 0 || mMode != PadValueMode_CONSTANT) {
        mNeedConvert = true;
    }
    if (!mNeedConvert) {
        return NO_ERROR;
    }
    mTempOutput.reset(Tensor::createDevice<float>(outputs[0]->shape(), Tensor::CAFFE));
    mTempInput.reset(Tensor::createDevice<float>(inputs[0]->shape(), Tensor::CAFFE));
    bool res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
    res      = res && backend()->onAcquireBuffer(mTempInput.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    mTempInputs  = {mTempInput.get(), inputs[1]};
    mTempOutputs = {mTempOutput.get()};

    if (mMode != PadValueMode_CONSTANT) {
        resizeImplFp16(backend(), inputs, outputs, &mCache);
    }

    backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempInput.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode Arm82PaddingPacked::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (mNeedConvert) {
        auto backend = static_cast<Arm82Backend *>(Execution::backend());
        backend->onCopyBuffer(input, mTempInput.get());

        if (mMode == PadValueMode_CONSTANT) {
            Arm82Padding::execute(mTempInputs, mTempOutputs, mMode);
        } else {
            // REFLECT or SYMMETRIC
            int offset     = mMode == PadValueMode_SYMMETRIC ? 0 : 1;
            auto cacheData = reinterpret_cast<CacheElement *>(mCache.host<char>());
            std::fill(cacheData, cacheData + mCache.elementSize(), CacheElement{-1, -1});
            const int *pad  = inputs[1]->host<int32_t>();
            int outputIndex = 0;
            MirrorPadImplFp16(mTempInput.get(), cacheData, mTempOutput.get(), pad, 0, 0, outputIndex, offset);
        }
        backend->onCopyBuffer(mTempOutput.get(), output);
        return NO_ERROR;
    }
    auto iw                    = input->width();
    auto ih                    = input->height();
    auto ic                    = input->channel();
    auto ib                    = input->batch();
    const int inputBatchStride = iw * ih * UP_DIV(ic, ARMV82_CHANNEL_UNIT) * ARMV82_CHANNEL_UNIT;
    const int outputBatchStide =
        output->width() * output->height() * UP_DIV(ic, ARMV82_CHANNEL_UNIT) * ARMV82_CHANNEL_UNIT;

    auto ow               = output->width();
    auto oh               = output->height();
    auto icC8             = UP_DIV(ic, ARMV82_CHANNEL_UNIT);
    auto padding          = inputs[1]->host<int32_t>();
    int outputSizeInBytes = output->elementSize() * sizeof(FLOAT16);
    if (inputs.size() == 3) {
        return NOT_SUPPORT;
    } else {
        ::memset(output->host<FLOAT16>(), 0, outputSizeInBytes);
    }
    for (int n = 0; n < ib; ++n) {
        auto inputN  = input->host<FLOAT16>() + inputBatchStride * n;
        auto outputN = output->host<FLOAT16>() + outputBatchStide * (padding[2 * 0] + n);
        for (int c = 0; c < icC8; ++c) {
            auto inputC  = inputN + c * iw * ih * ARMV82_CHANNEL_UNIT;
            auto outputC = outputN + c * ow * oh * ARMV82_CHANNEL_UNIT;

            for (int h = 0; h < ih; ++h) {
                auto inputH  = inputC + h * iw * ARMV82_CHANNEL_UNIT;
                auto outputH = outputC + (h + padding[2 * 2]) * ow * ARMV82_CHANNEL_UNIT;

                ::memcpy(outputH + padding[2 * 3] * ARMV82_CHANNEL_UNIT, inputH,
                         iw * ARMV82_CHANNEL_UNIT * sizeof(FLOAT16));
            }
        }
    }

    return NO_ERROR;
}
class Arm82PaddingCreator : public Arm82Backend::Arm82Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto param = op->main_as_PadParam();
        auto mode  = PadValueMode_CONSTANT;
        if (param) {
            mode = param->mode();
        }
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            return new Arm82Padding(backend, mode);
        }
        if (inputs[0]->dimensions() != 4) {
            MNN_ERROR("Currently Arm82 padding only support 4 dimension for NC4HW4\n");
            return nullptr;
        }
        if (inputs[0]->buffer().type.bits != 32) {
            MNN_ERROR("Currently Arm82 padding NC4HW4 only support 32 bit padding\n");
            return nullptr;
        }
        return new Arm82PaddingPacked(backend, mode);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_Padding, Arm82PaddingCreator);
}; // namespace MNN

#endif
