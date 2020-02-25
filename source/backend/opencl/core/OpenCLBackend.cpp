//
//  OpenCLBackend.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/OpenCLBackend.hpp"
#include "MNN_generated.h"

#include "core/TensorUtils.hpp"
#include "core/SizeComputer.hpp"
#include <map>
#include <mutex>
#include <thread>
#include "core/Macro.h"

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
        mBufferPoolInt8.reset(new BufferPoolInt8(mOpenCLRuntime->context(), CL_MEM_READ_WRITE));
        std::set<std::string> buildOptions;
        mNC4HW4BufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nc4hw4_buffer_to_image", buildOptions);
        mNCHWBufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nchw_buffer_to_image", buildOptions);
        mNHWCBufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nhwc_buffer_to_image", buildOptions);
        mImageToNC4HW4BufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nc4hw4_buffer", buildOptions);
        mImageToNHWCBufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        mImageToNCHWBufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nchw_buffer", buildOptions);
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

    //int8
    if(nativeTensor->getType().code == halide_type_int && nativeTensor->getType().bits == 8){

        unsigned int size = nativeTensor->size();
#ifdef LOG_VERBOSE
    MNN_PRINT("enter int8 alloc ! size : %d \n", size);
#endif
        if (storageType == DYNAMIC_SEPERATE || storageType == STATIC) {
            auto buffer                               = mBufferPoolInt8->alloc(size, true);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
            return true;
        }
        if (storageType == DYNAMIC) {
            auto buffer                               = mBufferPoolInt8->alloc(size);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
            return true;
        }
        return false;
    }
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
    if(nativeTensor->getType().code == halide_type_int && nativeTensor->getType().bits == 8){

        return true;
    }
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
    mBufferPoolInt8->clear();
    return true;
}
std::pair<float, bool> OpenCLBackend::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    auto creators = gCreator();
    auto iter      = creators->find(op->type());
    if (iter == creators->end()) {
        return std::make_pair(0.0f, false);
    }
    const float defaultScheduleTime = 0.05f;
    auto flops = SizeComputer::computeFlops(op, inputs, outputs);

    auto computeFlops = mOpenCLRuntime->flops();
    return std::make_pair(defaultScheduleTime + flops / 1024.0f / computeFlops * 1000.0f, true);
}
Execution* OpenCLBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start OpenCLBackend::onCreate \n");
#endif
    auto creators = gCreator();
    auto iter      = creators->find(op->type());
    if (iter == creators->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type %s, %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }

    auto maxImageSize = mOpenCLRuntime->getMaxImage2DSize();
    bool valid        = true;
    for (auto t : inputs) {
        int imageHeight = t->batch() * t->height();
        int imageWidth  = t->width() * UP_DIV(t->channel(), 4);
        if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
            valid = false;
            break;
        }
    }
    for (auto t : outputs) {
        int imageHeight = t->batch() * t->height();
        int imageWidth  = t->width() * UP_DIV(t->channel(), 4);
        if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
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
    if (nullptr != mHostBuffer.second && length <= mHostBuffer.first) {
        return;
    }
    mHostBuffer.first = length;
    mHostBuffer.second.reset(
        new cl::Buffer(mOpenCLRuntime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, length));
}

void OpenCLBackend::copyFromDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const{
        auto needSize = dstTensor->size();
        auto hostPtr = dstTensor->host<float>();
        cl_int error                = CL_SUCCESS;
        auto DeviceBuffer = (cl::Buffer*)srcTensor->deviceId();
        mOpenCLRuntime->commandQueue().enqueueReadBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, hostPtr);
}

void OpenCLBackend::copyToDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const{
        auto needSize = srcTensor->size();
        auto hostPtr                = srcTensor->host<int8_t>();
        cl_int error                = CL_SUCCESS;
        auto DeviceBuffer = (cl::Buffer*)dstTensor->deviceId();
        mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, hostPtr);
}

void OpenCLBackend::copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(srcTensor);
    MNN::Tensor interBuffer(0, Tensor::TENSORFLOW);
    interBuffer.buffer().dimensions = bufferShape.size();
    for (int i = 0; i < bufferShape.size(); i++) {
        interBuffer.buffer().dim[i].extent = bufferShape.at(i);
    }
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

    mOpenCLRuntime->commandQueue().enqueueReadBuffer(*mHostBuffer.second, CL_TRUE, 0, needSize, hostPtr);
}
void OpenCLBackend::copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(srcTensor);
    MNN::Tensor interBuffer(0, Tensor::TENSORFLOW);
    interBuffer.buffer().dimensions = bufferShape.size();
    for (int i = 0; i < bufferShape.size(); i++) {
        interBuffer.buffer().dim[i].extent = bufferShape.at(i);
    }
    auto needSize = srcTensor->size();
    _allocHostBuffer(needSize);
    interBuffer.buffer().device = (uint64_t)mHostBuffer.second.get();
    auto hostPtr                = srcTensor->host<float>();
    cl_int error                = CL_SUCCESS;
    mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*mHostBuffer.second, CL_TRUE, 0, needSize, hostPtr);
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

void OpenCLBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start onCopyBuffer !\n");
#endif
    //int8
    if(srcTensor->getType().code == halide_type_int && srcTensor->getType().bits == 8){
        if (srcTensor->deviceId() == 0 && dstTensor->deviceId() != 0) {
            copyToDeviceInt8(srcTensor, dstTensor);
        }else if(srcTensor->deviceId() != 0 && dstTensor->deviceId() == 0){
            copyFromDeviceInt8(srcTensor, dstTensor);
        }else{
            MNN_PRINT("onCopyBuffer int8 error !!! \n");
        }
    }else{
        if (srcTensor->deviceId() == 0 && dstTensor->deviceId() != 0) {
            copyToDevice(srcTensor, dstTensor);
        }else if(srcTensor->deviceId() != 0 && dstTensor->deviceId() == 0){
            copyFromDevice(srcTensor, dstTensor);
        }else{
            MNN_PRINT("onCopyBuffer float error !!! \n");
        }
    }

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
