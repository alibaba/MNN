//
//  OpenCLBackend.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/OpenCLBackend.hpp"
#include "MNN_generated.h"

#include <core/TensorUtils.hpp>
#include <map>
#include <mutex>
#include <thread>
#include "Macro.h"

namespace MNN {
namespace OpenCL {

std::map<OpType, OpenCLBackend::Creator*>* gCreator() {
    static std::once_flag once;
    static std::map<OpType, OpenCLBackend::Creator*>* creators = nullptr;
    std::call_once(once, [&]() { creators = new std::map<OpType, OpenCLBackend::Creator*>; });
    return creators;
};

OpenCLBackend::OpenCLBackend(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power)
    : Backend(MNN_FORWARD_OPENCL) {
    mPrecision = precision;
    // Shader precision
    if (precision == BackendConfig::Precision_Low) {
        mOpenCLRuntime.reset(new OpenCLRuntime(true));
    } else {
        mOpenCLRuntime.reset(new OpenCLRuntime(false));
    }
    if(mOpenCLRuntime.get()){
        if(mOpenCLRuntime->isCreateError() == true){
            mIsCreateError = true;
        }
        // Mid memory precision
        cl_channel_type dataType = CL_HALF_FLOAT;
        if (precision == BackendConfig::Precision_High) {
            dataType = CL_FLOAT;
        }
        mImagePool.reset(new ImagePool(mOpenCLRuntime->context(), dataType));
        mStaticImagePool.reset(new ImagePool(mOpenCLRuntime->context(), dataType));
        mBufferPool.reset(new BufferPool(mOpenCLRuntime->context(), CL_MEM_READ_WRITE));
    }
    
}

OpenCLBackend::~OpenCLBackend() {
#ifdef LOG_VERBOSE
    MNN_PRINT("enter OpenCLBackend::~OpenCLBackend \n");
#endif
}

OpenCLRuntime* OpenCLBackend::getOpenCLRuntime() {
    return mOpenCLRuntime.get();
}

bool OpenCLBackend::onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start OpenCLBackend::onAcquireBuffer !\n");
#endif
    auto tensorShape = OpenCL::tensorShapeFormat(nativeTensor);

    int N = tensorShape.at(0);
    int H = tensorShape.at(1);
    int W = tensorShape.at(2);
    int C = tensorShape.at(3);

    size_t imageWidth  = (size_t)UP_DIV(C, 4) * W;
    size_t imageHeight = (size_t)N * H;

    const std::vector<size_t> requestShape{imageWidth, imageHeight};
#ifdef LOG_VERBOSE
    MNN_PRINT("OpenCLBackend::onAcquireBuffer: [%d, %d, %d, %d], [%d, %d]\n", N, H, W, C, (int)imageWidth,
              (int)imageHeight);
#endif

    if (storageType == DYNAMIC_SEPERATE) {
        auto image                               = mImagePool->alloc(imageWidth, imageHeight, true);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
        return true;
    }
    if (storageType == DYNAMIC) {
        auto image                               = mImagePool->alloc(imageWidth, imageHeight);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
        return true;
    }
    MNN_ASSERT(storageType == STATIC);
    auto image                               = mStaticImagePool->alloc(imageWidth, imageHeight);
    ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
    return true;
}

bool OpenCLBackend::onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) {
    if (storageType == DYNAMIC_SEPERATE) {
        return true;
    }
    auto image = (cl::Image*)nativeTensor->deviceId();
    if (storageType == DYNAMIC) {
        mImagePool->recycle(image);
        return true;
    }
    if (storageType == STATIC) {
        mStaticImagePool->recycle(image, true);
    }
    return true;
}
bool OpenCLBackend::onAllocateBuffer() {
    return true;
}

bool OpenCLBackend::onClearBuffer() {
    mImagePool->clear();
    mBufferPool->clear();
    return true;
}

Execution* OpenCLBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start OpenCLBackend::onCreate \n");
#endif
    auto creators = gCreator();
    auto iter      = creators->find(op->type());
    if (iter == creators->end()) {
        MNN_PRINT("Don't support type %d, %s\n", op->type(), op->name()->c_str());
        return NULL;
    }

    auto maxImageSize = mOpenCLRuntime->getMaxImage2DSize();
    bool valid        = true;
    for (auto t : inputs) {
        int imageHeight = t->batch() * t->height();
        int imageWidth  = t->width() * UP_DIV(t->channel(), 4);
        if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
            (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1))) {
            valid = false;
            break;
        }
    }
    for (auto t : outputs) {
        int imageHeight = t->batch() * t->height();
        int imageWidth  = t->width() * UP_DIV(t->channel(), 4);
        if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
            (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1))) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        return NULL;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        MNN_PRINT("The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
        return NULL;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("End OpenCLBackend::onCreate \n");
#endif
    return exe;
}

void OpenCLBackend::onExecuteBegin() const {
}

void OpenCLBackend::onExecuteEnd() const {
}

bool OpenCLBackend::onWaitFinish() {
    int rc = mOpenCLRuntime.get()->commandQueue().finish();
    return rc == 0;
}

bool OpenCLBackend::isCreateError() const {
    return mIsCreateError;
}

void OpenCLBackend::_allocHostBuffer(int length) const {
    MNN_ASSERT(length > 0);
    if (nullptr != mHostBuffer.second && length < mHostBuffer.first) {
        return;
    }
    mHostBuffer.first = length;
    mHostBuffer.second.reset(
        new cl::Buffer(mOpenCLRuntime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, length));
}

void OpenCLBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start onCopyBuffer !\n");
#endif

    std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(srcTensor);

#ifdef LOG_VERBOSE
    MNN_PRINT("buffer shape : %d, %d, %d, %d \n", bufferShape.at(0), bufferShape.at(1), bufferShape.at(2),
              bufferShape.at(3));
#endif
    MNN::Tensor interBuffer(0, Tensor::TENSORFLOW);
    interBuffer.buffer().dimensions = bufferShape.size();
    for (int i = 0; i < bufferShape.size(); i++) {
        interBuffer.buffer().dim[i].extent = bufferShape.at(i);
    }

    if (!srcTensor->deviceId()) {
#ifdef LOG_VERBOSE
        MNN_PRINT("Host -> OpenCL !\n");
#endif
        auto needSize = srcTensor->size();
        _allocHostBuffer(needSize);
        interBuffer.buffer().device = (uint64_t)mHostBuffer.second.get();
        auto hostPtr                = srcTensor->host<float>();
        cl_int error                = CL_SUCCESS;
        auto bufferPtr = mOpenCLRuntime->commandQueue().enqueueMapBuffer(*mHostBuffer.second, CL_TRUE, CL_MAP_WRITE, 0,
                                                                         needSize, nullptr, nullptr, &error);
        if (error != CL_SUCCESS) {
            MNN_ERROR("Error to map buffer in copy buffer, error=%d\n", error);
            return;
        }
        if(bufferPtr != nullptr){
            ::memcpy(bufferPtr, hostPtr, needSize);
        }
        mOpenCLRuntime->commandQueue().enqueueUnmapMemObject(*mHostBuffer.second, bufferPtr);
        // Host -> OpenCL
        MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
        if (MNN_DATA_FORMAT_NHWC == data_format) {
            OpenCL::convertNHWCBufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                             *const_cast<cl::Kernel*>(&mNHWCBufferToImageFloat), mOpenCLRuntime.get());
            return;
        }
        if (MNN_DATA_FORMAT_NCHW == data_format) {
            OpenCL::convertNCHWBufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                             *const_cast<cl::Kernel*>(&mNCHWBufferToImageFloat), mOpenCLRuntime.get());
            return;
        }
        if (MNN_DATA_FORMAT_NC4HW4 == data_format) {
            OpenCL::convertNC4HW4BufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                               *const_cast<cl::Kernel*>(&mNC4HW4BufferToImageFloat),
                                               mOpenCLRuntime.get());
            return;
        }
        MNN_ASSERT(false);
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("OpenCL -> Host !\n");
#endif
    // OpenCL -> Host

    auto needSize = dstTensor->size();
    _allocHostBuffer(needSize);
    interBuffer.buffer().device = (uint64_t)mHostBuffer.second.get();

    MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    switch (data_format) {
        case MNN_DATA_FORMAT_NHWC:
            OpenCL::convertImageToNHWCBuffer(srcTensor, &interBuffer,
                                             *const_cast<cl::Kernel*>(&mImageToNHWCBufferFloat), mOpenCLRuntime.get());
            break;
        case MNN_DATA_FORMAT_NCHW:
            OpenCL::convertImageToNCHWBuffer(srcTensor, &interBuffer,
                                             *const_cast<cl::Kernel*>(&mImageToNCHWBufferFloat), mOpenCLRuntime.get());
            break;
        case MNN_DATA_FORMAT_NC4HW4:
            OpenCL::convertImageToNC4HW4Buffer(
                srcTensor, &interBuffer, *const_cast<cl::Kernel*>(&mImageToNC4HW4BufferFloat), mOpenCLRuntime.get());
            break;
        default:
            break;
    }
    auto hostPtr = dstTensor->host<float>();
    cl_int error                = CL_SUCCESS;
    auto bufferPtr =
        mOpenCLRuntime->commandQueue().enqueueMapBuffer(*mHostBuffer.second, true, CL_MAP_READ, 0, needSize, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        MNN_ERROR("Error to map buffer in copy buffer, error=%d\n", error);
        return;
    }
    if(bufferPtr != nullptr && hostPtr != nullptr){
        ::memcpy(hostPtr, bufferPtr, needSize);
    }
    mOpenCLRuntime->commandQueue().enqueueUnmapMemObject(*mHostBuffer.second, bufferPtr);

#ifdef LOG_VERBOSE
    MNN_PRINT("end onCopyBuffer !\n");
#endif
}

bool OpenCLBackend::addCreator(OpType t, Creator* c) {
    auto map = gCreator();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

class CLBackendCreator : public BackendCreator {
public:
    virtual Backend* onCreate(const Backend::Info& info) const override {
#ifdef MNN_USE_OPENCL_WRAPPER
        OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance();
        if (nullptr == OpenCLSymbolsOperator::getOpenclSymbolsPtr()) {
            MNN_PRINT("OpenCL init error , callback ... \n");
            return nullptr;
        }
        if (true == OpenCLSymbolsOperator::getOpenclSymbolsPtr()->isError()) {
            MNN_PRINT("parsing symbols error !!! \n");
            return nullptr;
        }
#endif
        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != info.user) {
            precision = info.user->precision;
            power     = info.user->power;
        }
        auto backend = new OpenCLBackend(precision, power);
        if(backend != nullptr){
            if(!backend->isCreateError()){
                return backend;
            }else{
                delete backend;
            }
        }
        return nullptr;    
    }
};

static const auto __opencl_global_initializer = []() {
    MNNInsertExtraBackendCreator(MNN_FORWARD_OPENCL, new CLBackendCreator, true);
    return true;
}();
} // namespace OpenCL
} // namespace MNN
