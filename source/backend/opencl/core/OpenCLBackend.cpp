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
#include "shape/SizeComputer.hpp"
#include <map>
#include <mutex>
#include <thread>
#include "core/Macro.h"

namespace MNN {
namespace OpenCL {

CLRuntime::CLRuntime(const Backend::Info& info){
    mInfo = info;
    
    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
    if (nullptr != mInfo.user) {
        precision = mInfo.user->precision;
        power     = mInfo.user->power;
    }

    mPrecision = precision;
    // Shader precision
    if (precision == BackendConfig::Precision_Low) {
        mOpenCLRuntime.reset(new OpenCLRuntime(true));
    } else {
        mOpenCLRuntime.reset(new OpenCLRuntime(false));
    }
    
    mCLRuntimeError = mOpenCLRuntime->isCreateError();
    if(!mCLRuntimeError){
        mImagePool.reset(new ImagePool(mOpenCLRuntime->context()));
        mStaticImagePool.reset(new ImagePool(mOpenCLRuntime->context()));
        mBufferPool.reset(new BufferPool(mOpenCLRuntime->context(), CL_MEM_READ_WRITE));
        mBufferPoolInt8.reset(new BufferPoolInt8(mOpenCLRuntime->context(), CL_MEM_READ_WRITE));
    }
}

CLRuntime::~CLRuntime() {
    mOpenCLRuntime = nullptr;
    mImagePool = nullptr;
    mStaticImagePool = nullptr;
    mBufferPool = nullptr;
    mBufferPoolInt8 = nullptr;
}

bool CLRuntime::onSetCache(const void* buffer, size_t size) {
    mOpenCLRuntime->setCache(std::make_pair(buffer, size));
    return true;
}

std::pair<const void*, size_t> CLRuntime::onGetCache() {
    return mOpenCLRuntime->makeCache();
}

Backend* CLRuntime::onCreate() const {
    return new OpenCLBackend(this);
}

void CLRuntime::onGabageCollect(int level) {
    //nothing now
}

bool CLRuntime::isCLRuntimeError() {
    return mCLRuntimeError;
}

std::map<OpType, OpenCLBackend::Creator*>* gCreator() {
    static std::once_flag once;
    static std::map<OpType, OpenCLBackend::Creator*>* creators = nullptr;
    std::call_once(once, [&]() { creators = new std::map<OpType, OpenCLBackend::Creator*>; });
    return creators;
};

OpenCLBackend::OpenCLBackend(const CLRuntime *runtime)
    : Backend(MNN_FORWARD_OPENCL) {
        
    mCLRuntime = runtime;
    mOpenCLRuntime = mCLRuntime->mOpenCLRuntime;
    mImagePool = mCLRuntime->mImagePool;
    mStaticImagePool = mCLRuntime->mStaticImagePool;
    mBufferPool = mCLRuntime->mBufferPool;
    mBufferPoolInt8 = mCLRuntime->mBufferPoolInt8;
    mPrecision = mCLRuntime->mPrecision;

    if(mOpenCLRuntime.get()){
        if(mOpenCLRuntime->isCreateError() == true){
            mIsCreateError = true;
        }
        std::set<std::string> buildOptions;
        //when input or output need buffer2image transformation, open macro BUFFER_IMAGE_IO_TRANS
        //because cpu input and output are fp32
        buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
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
    
    cl_channel_type dataType = CL_HALF_FLOAT;
    //when user want high precision or the device not support fp16, use float datatype
    if (mPrecision == BackendConfig::Precision_High) {
        dataType = CL_FLOAT;
    }
    //Currently! int datatype will be converted to float
    /*
    if(nativeTensor->getType().code == halide_type_int) {
        dataType = CL_SIGNED_INT32;
        if(nativeTensor->getType().bits == 8) {
            //dataType = CL_SIGNED_INT8;
        }
    } else if(nativeTensor->getType().code == halide_type_uint) {
        dataType = CL_UNSIGNED_INT32;
        if(nativeTensor->getType().bits == 8) {
            //dataType = CL_UNSIGNED_INT8;
        }
    }
     */


    if (storageType == DYNAMIC_SEPERATE) {
        auto image                               = mImagePool->alloc(imageWidth, imageHeight, dataType, true);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
        return true;
    }
    if (storageType == DYNAMIC) {
        auto image                               = mImagePool->alloc(imageWidth, imageHeight, dataType);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
        return true;
    }
    MNN_ASSERT(storageType == STATIC);
    auto image                               = mStaticImagePool->alloc(imageWidth, imageHeight, dataType);
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
#ifndef MNN_BUILD_MINI
    auto flops = SizeComputer::computeFlops(op, inputs, outputs);
#else
    auto flops = 0.0f;
#endif
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
#if 0
    bool res = false;
#define PERMIT(t) if (op->type() == t) res = true
    PERMIT(OpType_Convolution);
    PERMIT(OpType_Deconvolution);
    PERMIT(OpType_Pooling);
    PERMIT(OpType_ReLU);
    //PERMIT(OpType_Softmax);
    PERMIT(OpType_UnaryOp);
    //PERMIT(OpType_SoftmaxGrad);
    PERMIT(OpType_Conv2DBackPropFilter);
#undef PERMIT
    if (!res) {
        return nullptr;
    }
#endif
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
        auto tensorShape = OpenCL::tensorShapeFormat(t);
        int imageHeight = tensorShape[0] * tensorShape[1];
        int imageWidth  = tensorShape[2] * UP_DIV(tensorShape[3], 4);
        if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
            valid = false;
            break;
        }
        
        //input in raster not used, origin instead
        auto des = TensorUtils::getDescribe(t)->regions;
        for(auto region : des)
        {
            auto tensor = region.origin;
            auto tensorShape = OpenCL::tensorShapeFormat(tensor);
            int originHeight = tensorShape[0] * tensorShape[1];
            int originWidth  = tensorShape[2] * UP_DIV(tensorShape[3], 4);
            if (originHeight > maxImageSize.at(0) || originWidth > maxImageSize.at(1)) {
                valid = false;
                break;
            }
        }
    }
    for (auto t : outputs) {
        auto tensorShape = OpenCL::tensorShapeFormat(t);
        int imageHeight = tensorShape[0] * tensorShape[1];
        int imageWidth  = tensorShape[2] * UP_DIV(tensorShape[3], 4);
        if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
            valid = false;
            break;
        }
    }
    
    if (!valid) {
        for (auto t : inputs) {
            auto tensorShape = OpenCL::tensorShapeFormat(t);
            MNN_PRINT("input n:%d, h:%d, w:%d, c:%d\n", tensorShape[0], tensorShape[1], tensorShape[2], tensorShape[3]);
        }
        for (auto t : outputs) {
            auto tensorShape = OpenCL::tensorShapeFormat(t);
            MNN_PRINT("output n:%d, h:%d, w:%d, c:%d\n", tensorShape[0], tensorShape[1], tensorShape[2], tensorShape[3]);
        }
        MNN_PRINT("beyond cl_image creat size! fallback to cpu backend\n");
        return NULL;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        if (nullptr != op->name()) {
            MNN_PRINT("The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("The Creator Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("End OpenCLBackend::onCreate \n");
#endif
    return exe;
}

void OpenCLBackend::onResizeBegin() {
    mOpenCLRuntime->setCommandQueueProfileEnable();
}

void OpenCLBackend::onResizeEnd() {
#ifndef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->setCommandQueueProfileDisable();
#endif
}

void OpenCLBackend::onExecuteBegin() const {
    mOpenCLRuntime->mQueueCount = 0;
    mOpenCLRuntime->mKernelTime = 0;
}

void OpenCLBackend::onExecuteEnd() const {
    mOpenCLRuntime->mQueueCount = 0;
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
    std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(dstTensor);


    auto needSize = dstTensor->size();
    auto hostPtr = dstTensor->host<int8_t>();
    auto DeviceBuffer = (cl::Buffer*)srcTensor->deviceId();
    cl_int error                = CL_SUCCESS;

#ifndef MNN_OCL_QUANT_DUMP
    error = mOpenCLRuntime->commandQueue().enqueueReadBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, hostPtr);
    MNN_ASSERT(error == 0);
#else//for dump test
    int8_t* tmpPtr = (int8_t *)malloc(needSize);
    error = mOpenCLRuntime->commandQueue().enqueueReadBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, tmpPtr);
    MNN_ASSERT(error == 0);
    int C_4 = (bufferShape[3]+3)/4;
    for(int n=0; n<bufferShape[0]; n++) {
        for(int c=0; c<bufferShape[3]; c++) {
            for(int h=0; h<bufferShape[1]; h++) {
                for(int w=0; w<bufferShape[2]; w++) {
                   hostPtr[n*bufferShape[3]*bufferShape[1]*bufferShape[2] + c*bufferShape[1]*bufferShape[2] + h*bufferShape[2] + w] =
                    tmpPtr[n*C_4*bufferShape[1]*bufferShape[2]*4 + (c/4)*bufferShape[1]*bufferShape[2]*4 + h*bufferShape[2]*4 + w*4 + c%4];
                }
            }
        }
    }
    if(tmpPtr != nullptr) {
        free(tmpPtr);
        tmpPtr = nullptr;
    }
#endif
   
#ifdef ENABLE_OPENCL_TIME_PROFILER
    MNN_PRINT("total kernel time:%d us\n", (int)mOpenCLRuntime->mKernelTime);
#endif
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
    
    
    void* hostPtr;
    void* tmpPtr;
    if(dstTensor->getType().code == halide_type_int) {
        if(dstTensor->getType().bits == 8){
            needSize *= 4;
            hostPtr = malloc(needSize);
        } else if(dstTensor->getType().bits == 32){
            hostPtr = malloc(needSize);
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", dstTensor->getType().bits);
            MNN_ASSERT(false);
        }
    } else if(dstTensor->getType().code == halide_type_uint){
        if(dstTensor->getType().bits == 8){
            needSize *= 4;
            hostPtr = malloc(needSize);
        } else if(dstTensor->getType().bits == 32){
            hostPtr = malloc(needSize);
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", dstTensor->getType().bits);
            MNN_ASSERT(false);
        }
    } else {
        hostPtr = dstTensor->host<float>();
    }
    
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

    cl_int error                = CL_SUCCESS;

    #ifdef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->commandQueue().finish();
    {
        AUTOTIME;
        mOpenCLRuntime->commandQueue().enqueueReadBuffer(*mHostBuffer.second, CL_TRUE, 0, needSize, hostPtr);
    }
    #else
    mOpenCLRuntime->commandQueue().enqueueReadBuffer(*mHostBuffer.second, CL_TRUE, 0, needSize, hostPtr);
    #endif
    
    if(dstTensor->getType().code == halide_type_int) {
        if(dstTensor->getType().bits == 8){
            tmpPtr = dstTensor->host<int8_t>();
            for(int i=0; i<needSize/4; i++) {
                ((int8_t*)tmpPtr)[i] = (int8_t)((float*)hostPtr)[i];
            }
        } else if(dstTensor->getType().bits == 32){
            tmpPtr = dstTensor->host<int32_t>();
            for(int i=0; i<needSize/4; i++) {
                ((int32_t*)tmpPtr)[i] = (int32_t)((float*)hostPtr)[i];
            }
        }
        if(hostPtr != nullptr) {
            free(hostPtr);
            hostPtr = nullptr;
        }
    } else if(dstTensor->getType().code == halide_type_uint){
        if(dstTensor->getType().bits == 8){
            tmpPtr = dstTensor->host<uint8_t>();
            for(int i=0; i<needSize/4; i++) {
                ((uint8_t*)tmpPtr)[i] = (uint8_t)((float*)hostPtr)[i];
            }
        } else if(dstTensor->getType().bits == 32){
            tmpPtr = dstTensor->host<uint32_t>();
            for(int i=0; i<needSize/4; i++) {
                ((uint32_t*)tmpPtr)[i] = (uint32_t)((float*)hostPtr)[i];
            }
        }
        if(hostPtr != nullptr) {
            free(hostPtr);
            hostPtr = nullptr;
        }
    }
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    MNN_PRINT("total kernel time:%d us\n", (int)mOpenCLRuntime->mKernelTime);
#endif
}
void OpenCLBackend::copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(srcTensor);
    MNN::Tensor interBuffer(0, Tensor::TENSORFLOW);
    interBuffer.buffer().dimensions = bufferShape.size();
    for (int i = 0; i < bufferShape.size(); i++) {
        interBuffer.buffer().dim[i].extent = bufferShape.at(i);
    }
    
    auto needSize = srcTensor->size();

    void* hostPtr;
    void* tmpPtr;
    if(srcTensor->getType().code == halide_type_int) {
        //Copy maybe slow, TODO
        if(srcTensor->getType().bits == 8){
            tmpPtr = srcTensor->host<int8_t>();
            needSize *= 4;
            hostPtr = malloc(needSize);
            for(int i=0; i<needSize/4; i++) {
                ((float*)hostPtr)[i] = (float)((int8_t*)tmpPtr)[i];
            }
        } else if(srcTensor->getType().bits == 32){
            tmpPtr = srcTensor->host<int32_t>();
            hostPtr = malloc(needSize);
            for(int i=0; i<needSize/4; i++) {
                ((float*)hostPtr)[i] = (float)((int32_t*)tmpPtr)[i];
            }
        }

    } else if(srcTensor->getType().code == halide_type_uint){
        //Copy maybe slow, TODO
        if(srcTensor->getType().bits == 8){
            tmpPtr = srcTensor->host<uint8_t>();
            needSize *= 4;
            hostPtr = malloc(needSize);
            for(int i=0; i<needSize/4; i++) {
                ((float*)hostPtr)[i] = (float)((uint8_t*)tmpPtr)[i];
            }
        } else if(srcTensor->getType().bits == 32){
            tmpPtr = srcTensor->host<uint32_t>();
            hostPtr = malloc(needSize);
            for(int i=0; i<needSize/4; i++) {
                ((float*)hostPtr)[i] = (float)((uint32_t*)tmpPtr)[i];
            }
        }
    } else {
        hostPtr                = srcTensor->host<float>();
    }
    
    _allocHostBuffer(needSize);
    interBuffer.buffer().device = (uint64_t)mHostBuffer.second.get();
    
    cl_int error                = CL_SUCCESS;
    
    #ifdef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->commandQueue().finish();
    {
        AUTOTIME;
        mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*mHostBuffer.second, CL_TRUE, 0, srcTensor->elementSize()*sizeof(float), hostPtr);
    }
    #else
    mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*mHostBuffer.second, CL_TRUE, 0, srcTensor->elementSize()*sizeof(float), hostPtr);
    #endif
    // Host -> OpenCL
    MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    if (MNN_DATA_FORMAT_NHWC == data_format) {
        OpenCL::convertNHWCBufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                         *const_cast<cl::Kernel*>(&mNHWCBufferToImageFloat), mOpenCLRuntime.get());
    } else if (MNN_DATA_FORMAT_NCHW == data_format) {
        OpenCL::convertNCHWBufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                         *const_cast<cl::Kernel*>(&mNCHWBufferToImageFloat), mOpenCLRuntime.get());
    } else if (MNN_DATA_FORMAT_NC4HW4 == data_format) {
        OpenCL::convertNC4HW4BufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                           *const_cast<cl::Kernel*>(&mNC4HW4BufferToImageFloat),
                                           mOpenCLRuntime.get());
    } else {
        MNN_PRINT("data format not support\n");
        MNN_ASSERT(false);
    }
    
    if(srcTensor->getType().code == halide_type_uint || srcTensor->getType().code == halide_type_int){
        mOpenCLRuntime.get()->commandQueue().finish();
        if(nullptr != hostPtr){
            free(hostPtr);
            hostPtr = nullptr;
        }
    }
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

// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Runtime Register
// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class CLRuntimeCreator : public RuntimeCreator {
    virtual Runtime* onCreate(const Backend::Info& info) const {
    #ifdef MNN_USE_LIB_WRAPPER
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
        auto rt = new CLRuntime(info);
        if(rt->isCLRuntimeError() == true) {
            delete rt;
            return nullptr;
        }
        return rt;
    }
    virtual bool onValid(Backend::Info& info) const {
        return true;
    }
};

static bool gResistor = []() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_OPENCL, new CLRuntimeCreator, true);
    return false;
}();

} // namespace OpenCL
} // namespace MNN
