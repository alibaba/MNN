//
//  ImageProcess.cpp
//  MNN
//
//  Created by MNN on 2018/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <map>
#include <MNN/ImageProcess.hpp>
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "backend/cpu/CPUTensorConvert.hpp"
#include "backend/cpu/CPUImageProcess.hpp"
#include <MNN/MNNForwardType.h>
#include "core/Backend.hpp"
#ifdef MNN_USE_SSE
#include "backend/cpu/x86_x64/AVX2Functions.hpp"
#endif

#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>
#include "core/Execution.hpp"
#include "core/Backend.hpp"
#include "MNN_generated.h"

#ifdef _MSC_VER
#include "backend/cpu/x86_x64/cpu_id.h"
#endif

#define CACHE_SIZE 256
namespace MNN {

void registerBackend();

namespace CV {
struct ImageProcess::Inside {
    Config config;
    std::unique_ptr<CPUImageProcess> execution;
};

ImageProcess::~ImageProcess() {
    delete mInside;
}

ImageProcess::ImageProcess(const Config& config) {
    mInside         = new Inside;
    mInside->config = config;
    for (int i = 0; i < 4; ++i) {
        mInside->config.mean[i]   = config.mean[i];
        mInside->config.normal[i] = config.normal[i];
    }
    registerBackend();
    auto coreFunctions =
#ifdef MNN_USE_SSE
    AVX2Functions::get();
#else
    nullptr;
#endif
    mInside->execution.reset(new CPUImageProcess(config, coreFunctions));
}

ImageProcess* ImageProcess::create(const Config& config, const Tensor* dstTensor) {
    // TODO Get dstTensor' backend
    #ifdef _MSC_VER
        auto cpuFlags = libyuv::InitCpuFlags();
        bool support = true;
        support = support && (cpuFlags & libyuv::kCpuHasSSSE3); // _mm_shuffle_epi8
        support = support && (cpuFlags & libyuv::kCpuHasSSE41); // _mm_cvtepu8_epi32
        if (!support) {
            MNN_ERROR("CPU must support SSSE3 and SSE4.1 for using ImageProcess\n");
            return nullptr;
        }
    #endif
    return new ImageProcess(config);
}

ImageProcess* ImageProcess::create(const ImageFormat sourceFormat, const ImageFormat destFormat, const float* means,
                                   const int meanCount, const float* normals, const int normalCount,
                                   const Tensor* dstTensor) {
    MNN::CV::ImageProcess::Config config;
    if (means != nullptr && meanCount > 0) {
        ::memcpy(config.mean, means, sizeof(float) * meanCount);
    }
    if (normals != nullptr && normalCount > 0) {
        ::memcpy(config.normal, normals, sizeof(float) * normalCount);
    }
    config.sourceFormat = sourceFormat;
    config.destFormat   = destFormat;
    return new ImageProcess(config);
}

void ImageProcess::setMatrix(const Matrix& matrix) {
    mTransform = matrix;
    mTransform.invert(&mTransformInvert);
    mInside->execution->setMatrix(matrix);
}

static int _getBpp(ImageFormat format) {
    switch (format) {
        case RGB:
        case BGR:
        case YCrCb:
        case YUV:
        case HSV:
        case XYZ:
            return 3;
        case RGBA:
        case BGRA:
            return 4;
        case GRAY:
            return 1;
        case BGR555:
        case BGR565:
            return 2;
        default:
            break;
    }
    return 0;
}

Tensor* ImageProcess::createImageTensor(halide_type_t type, int width, int height, int bpp, void* p) {
    return Tensor::create(std::vector<int>{1, height, width, bpp}, type, p);
}

static ImageFormat _correctImageFormat(int outputBpp, halide_type_t type, ImageFormat format) {
    if (outputBpp != 4) {
        return format;
    }
    // TODO, use same judge for uint8 -> float
    if (type.code == halide_type_float) {
        return format;
    }

    static std::map<ImageFormat, ImageFormat> imageFormatTable = {{RGB, RGBA}, {BGR, BGRA}, {GRAY, RGBA}};
    if (imageFormatTable.find(format) != imageFormatTable.end()) {
        return imageFormatTable.find(format)->second;
    }
    return format;
}

ErrorCode ImageProcess::convert(const uint8_t* source, int iw, int ih, int stride, Tensor* destOrigin) {
    auto dest = destOrigin;
    if (nullptr == dest || nullptr == source) {
        MNN_ERROR("null dest or source for image process\n");
        return INPUT_DATA_ERROR;
    }
    if (TensorUtils::getDescribe(dest)->backend == nullptr && destOrigin->buffer().host == nullptr) {
        MNN_ERROR("Invalid Tensor, the session may not be ready\n");
        return INPUT_DATA_ERROR;
    }
    std::shared_ptr<Tensor> tempTensor;
    auto ow              = dest->width();
    auto oh              = dest->height();
    auto bpp             = dest->channel();
    auto dimensionFormat = TensorUtils::getDescribe(dest)->dimensionFormat;
    auto tensorBn = TensorUtils::getDescribe(dest)->backend;
    auto bnType = MNN_FORWARD_CPU;
    if(tensorBn){
        bnType = tensorBn->type();
    }
    if (bnType != MNN_FORWARD_CPU) {
        tempTensor.reset(Tensor::create({1, bpp, oh, ow}, dest->getType(), nullptr, Tensor::CAFFE_C4),[destOrigin] (void* p) {
            auto hostTensor = (Tensor*)p;
            destOrigin->copyFromHostTensor(hostTensor);
            delete hostTensor;
        });
        dest = tempTensor.get();
    }
    else if (MNN_DATA_FORMAT_NCHW == dimensionFormat) {
        tempTensor.reset(Tensor::create(dest->shape(), dest->getType(), nullptr, Tensor::CAFFE_C4), [destOrigin](void* p) {
            auto hostTensor = (Tensor*)p;
            CPUTensorConverter::convert(hostTensor, destOrigin);
            delete hostTensor;
        });
        dest = tempTensor.get();
    }
    dimensionFormat = TensorUtils::getDescribe(dest)->dimensionFormat;
    if (dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        bpp = 4;
    }
    return convert(source, iw, ih, stride, dest->host<void>(), ow, oh, bpp, ow * bpp, dest->getType());
}

ErrorCode ImageProcess::convert(const uint8_t* source, int iw, int ih, int stride, void* dest, int ow, int oh,
                                int outputBpp, int outputStride, halide_type_t type) {
    auto ic = _getBpp(mInside->config.sourceFormat);
    auto oc = outputBpp;
    if (0 == oc) {
        oc = _getBpp(mInside->config.destFormat);
    }
    auto ins = { createImageTensor(halide_type_of<uint8_t>(), iw, ih, ic, (void*)source) };
    auto outs = { createImageTensor(type, ow, oh, oc, dest) };
    mInside->execution->setPadVal(this->mPaddingValue);
    mInside->execution->onResize(ins, outs);
    mInside->execution->onExecute(ins, outs);
    return NO_ERROR;
}
} // namespace CV
} // namespace MNN
