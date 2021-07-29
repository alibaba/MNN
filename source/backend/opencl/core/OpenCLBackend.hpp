//
//  OpenCLBackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLBackend_hpp
#define OpenCLBackend_hpp

#include "core/Backend.hpp"
#include "MNN_generated.h"

#include <list>
#include <vector>
#include "backend/opencl/core/BufferPool.hpp"
#include "backend/opencl/core/ImageBufferConvertor.hpp"
#include "backend/opencl/core/BufferConvertor.hpp"
#include "backend/opencl/core/ImagePool.hpp"
#include "core/Macro.h"
#include "backend/opencl/core/ImageBufferConvertor.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "half.hpp"

#ifdef ENABLE_OPENCL_TIME_PROFILER
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#endif

namespace MNN {
namespace OpenCL {

class CLRuntime : public Runtime {
public:
    CLRuntime(const Backend::Info& info);
    virtual ~CLRuntime();
    
    virtual Backend* onCreate(const BackendConfig* config) const override;
    virtual void onGabageCollect(int level) override;
    virtual std::pair<const void*, size_t> onGetCache() override;
    virtual bool onSetCache(const void* buffer, size_t size) override;
    bool isCLRuntimeError();
    
private:
    Backend::Info mInfo;
    std::shared_ptr<OpenCLRuntime> mOpenCLRuntime;
    
    BackendConfig::PrecisionMode mPrecision;
    bool mCLRuntimeError = false;

    friend class OpenCLBackend;
    
};
 

class OpenCLBackend final : public Backend {
public:
    OpenCLBackend(const CLRuntime *runtime);
    ~OpenCLBackend();

    OpenCLRuntime *getOpenCLRuntime();
    virtual bool onAcquireBuffer(const Tensor *nativeTensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor *nativeTensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;

    virtual void onResizeBegin() override;
    virtual void onResizeEnd() override;
    
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;


    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;

    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &output, const MNN::Op *op, Backend *backend) const = 0;
    };

    static bool addCreator(std::pair<OpType, GpuMemObject> t, Creator *c);

    BufferPool *getBufferPool() const {
        return mBufferPool.get();
    }
 
    BackendConfig::PrecisionMode getPrecision() const {
        return mPrecision;
    }
    
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                             const MNN::Op* op) override;

    bool isCreateError() const;
    virtual void* onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor) override;
    virtual bool onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr) override;
    
private:
    void copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyBetweenDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyFromDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyToDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const;

    void _allocHostBuffer(int length) const;
    cl::Kernel mImageToNCHWBufferFloat;
    cl::Kernel mImageToNC4HW4BufferFloat;
    cl::Kernel mImageToNHWCBufferFloat;
    cl::Kernel mNC4HW4BufferToImageFloat;
    cl::Kernel mNCHWBufferToImageFloat;
    cl::Kernel mNHWCBufferToImageFloat;
    cl::Kernel mNHWCBufferToImageInt8;
    
    cl::Kernel mNC4HW4BufferToNCHWBufferOut;
    cl::Kernel mNC4HW4BufferToNHWCBufferOut;
    cl::Kernel mNC4HW4BufferToNC4HW4BufferOut;
    cl::Kernel mNC4HW4BufferToNC4HW4BufferInp;
    cl::Kernel mNCHWBufferToNC4HW4BufferInp;
    cl::Kernel mNHWCBufferToNC4HW4BufferInp;
    cl::Kernel mNC4HW4BufferToNC4HW4Buffer;

    const CLRuntime* mCLRuntime;
    
    std::shared_ptr<ImagePool> mImagePool;
    std::shared_ptr<ImagePool> mStaticImagePool;
    std::shared_ptr<BufferPool> mBufferPool;
    std::shared_ptr<BufferPool> mStaticBufferPool;
    
    std::shared_ptr<OpenCLRuntime> mOpenCLRuntime;
    
    mutable std::pair<int, std::shared_ptr<cl::Buffer>> mHostBuffer;
    BackendConfig::PrecisionMode mPrecision;
    bool mIsCreateError{false};
    
private:
    
    void convertToDevice(const Tensor* srcTensor, const Tensor* dstTensor, MNN_DATA_FORMAT data_format, bool svmFlag = false) const;
    void convertFromDevice(const Tensor* srcTensor, const Tensor* dstTensor, MNN_DATA_FORMAT data_format, bool svmFlag = false) const;

    void* svmPtr = nullptr;
    std::pair<int, void *> mMapMem;
    bool mUseSvm = false;
    void* allocMapTensorMemory(int length, bool svmFlag = false, cl_device_svm_capabilities svm_cap_ = 0);

};

template <class T>
class OpenCLCreatorRegister {
public:
    OpenCLCreatorRegister(OpType type, GpuMemObject memObj) {
        T *t = new T;
        OpenCLBackend::addCreator(std::make_pair(type, memObj), t);
    }
    ~OpenCLCreatorRegister() = default;
};

template <typename T>
class TypedCreator : public OpenCLBackend::Creator {
public:
    virtual ~TypedCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                Backend *backend) const override {
        return new T(inputs, op, backend);
    }
};

} // namespace OpenCL
} // namespace MNN
#endif  /* OpenCLBackend_hpp */
