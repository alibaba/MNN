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
#include <MNN/MNNForwardType.h>
#include "core/Backend.hpp"

#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>
#include "core/Execution.hpp"
#include "core/Backend.hpp"
#include "MNN_generated.h"
#include "ImageProcessUtils.hpp"

#ifdef _MSC_VER
#include "backend/cpu/x86_x64/cpu_id.h"
#endif

#define CACHE_SIZE 256
namespace MNN {

void registerBackend();
namespace CV {
struct ImageProcess::Inside {
    Config config;
    // ImageProcessUtils* proc;
    std::unique_ptr<ImageProcessUtils> proc;
};
void ImageProcess::destroy(ImageProcess* pro) {
    if (nullptr != pro) {
        delete pro;
    }
}

ImageProcess::~ImageProcess() {
    delete mInside;
}
ImageProcess::ImageProcess(const Config& config) {
    mInside         = new Inside;
    mInside->config = config;
    registerBackend();
    auto coreFunctions = MNNGetCoreFunctions();
    mInside->proc.reset(new ImageProcessUtils(config, coreFunctions));
}

ImageProcess* ImageProcess::create(const Config& config, const Tensor* dstTensor) {
    // TODO Get dstTensor' backend
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

void ImageProcess::setMatrix(const CV::Matrix& matrix) {
    mInside->proc->setMatrix(matrix);
}

static int _getBpp(CV::ImageFormat format) {
    switch (format) {
        case CV::RGB:
        case CV::BGR:
        case CV::YCrCb:
        case CV::YUV:
        case CV::HSV:
        case CV::XYZ:
            return 3;
        case CV::RGBA:
        case CV::BGRA:
            return 4;
        case CV::GRAY:
            return 1;
        case CV::BGR555:
        case CV::BGR565:
            return 2;
        default:
            break;
    }
    return 0;
}

Tensor* ImageProcess::createImageTensor(halide_type_t type, int width, int height, int bpp, void* p) {
    return Tensor::create(std::vector<int>{1, height, width, bpp}, type, p);
}

static CV::ImageFormat _correctImageFormat(int outputBpp, halide_type_t type, CV::ImageFormat format) {
    if (outputBpp != 4) {
        return format;
    }
    // TODO, use same judge for uint8 -> float
    if (type.code == halide_type_float) {
        return format;
    }
    
    static std::map<CV::ImageFormat, CV::ImageFormat> imageFormatTable = {{CV::RGB, CV::RGBA}, {CV::BGR, CV::BGRA}, {CV::GRAY, CV::RGBA}};
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
    if (TensorUtils::getDescribeOrigin(dest)->getBackend() == nullptr && destOrigin->buffer().host == nullptr) {
        MNN_ERROR("Invalid Tensor, the session may not be ready\n");
        return INPUT_DATA_ERROR;
    }
    std::shared_ptr<Tensor> tempTensor;
    auto ow              = dest->width();
    auto oh              = dest->height();
    auto bpp             = dest->channel();
    auto dimensionFormat = TensorUtils::getDescribe(dest)->dimensionFormat;
    auto tensorBn = TensorUtils::getDescribeOrigin(dest)->getBackend();
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
    int ic = _getBpp(mInside->config.sourceFormat);
    mInside->proc->setPadding(mPaddingValue);
    mInside->proc->resizeFunc(ic, iw, ih, bpp, ow, oh, dest->getType(), stride);
    return mInside->proc->execFunc(source, stride, dest->host<void>());
}

ErrorCode ImageProcess::convert(const uint8_t* source, int iw, int ih, int stride, void* dest, int ow, int oh,
                                int outputBpp, int outputStride, halide_type_t type) {
    
    int ic = _getBpp(mInside->config.sourceFormat);
    int oc = outputBpp;
    if (outputBpp == 0) {
        oc = _getBpp(mInside->config.destFormat);
    }
    mInside->proc->setPadding(mPaddingValue);
    mInside->proc->resizeFunc(ic, iw, ih, oc, ow, oh, type, stride);
    return mInside->proc->execFunc(source, stride, dest);
}

void ImageProcess::setDraw() {
    if (mInside && mInside->proc) {
        mInside->proc->setDraw();
    }
}

void ImageProcess::draw(uint8_t* img, int w, int h, int c, const int* regions, int num, const uint8_t* color) {
    std::vector<int32_t> tmpReg(3 * num);
    ::memcpy(tmpReg.data(), (void*)regions, 4 * 3 * num);
    double tmpBuf[4];
    ::memcpy(tmpBuf, color, 4 * sizeof(double));
    mInside->proc->resizeFunc(c, w, h, c, w, h);
    mInside->proc->draw(img, w, h, c, tmpReg.data(), num, (uint8_t*)tmpBuf);
}
} // namespace CV
} // namespace MNN
