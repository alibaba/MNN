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
#include <MNN/ErrorCode.hpp>

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
#define MNN_USER_SET_DEVICE
#include "MNN/MNNSharedContext.h"

#ifdef ENABLE_OPENCL_TIME_PROFILER
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#endif

namespace MNN {
namespace OpenCL {
struct TuneInfo;
struct RecordUpdateInfo{
    std::vector<cl_array_arg_qcom> update_kernel_args;
    std::vector<cl_workgroup_qcom> update_global_size;
    std::vector<cl_workgroup_qcom> update_local_size;
};
struct RecordInfo{
    cl_recording_qcom record;
    std::vector<RecordUpdateInfo*> updateInfo;
};
class CLRuntime : public Runtime {
public:
    CLRuntime(const Backend::Info& info, int platformSize, int platformId, int deviceId = 0, void *contextPtr = nullptr, void *glshared = nullptr);
    virtual ~CLRuntime();

    virtual Backend* onCreate(const BackendConfig* config) const override;
    virtual void onGabageCollect(int level) override;
    virtual float onGetMemoryInMB() override;
    virtual std::pair<const void*, size_t> onGetCache() override;
    virtual bool onSetCache(const void* buffer, size_t size) override;
    bool isCLRuntimeError();
    int onGetRuntimeStatus(RuntimeStatus statusEnum) const override;
    virtual bool onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           const MNN::Op* op, OpInfo& dstInfo) const override;
    virtual void onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const MNN::Op* op) override;
    void convertToDevice(const Tensor* srcTensor, const Tensor* dstTensor, MNN_DATA_FORMAT data_format, bool svmFlag = false, int memtype = MNN_FORWARD_CPU) const;
    void convertFromDevice(const Tensor* srcTensor, const Tensor* dstTensor, MNN_DATA_FORMAT data_format, bool svmFlag = false, int memtype = MNN_FORWARD_CPU) const;
    void copyBetweenDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;

private:
    Backend::Info mInfo;
    std::shared_ptr<OpenCLRuntime> mOpenCLRuntime;
    std::shared_ptr<ImagePool> mImagePool;
    std::shared_ptr<BufferPool> mBufferPool;
    BackendConfig::PrecisionMode mPrecision;
    BackendConfig::MemoryMode mMemory;
    bool mCLRuntimeError = false;

    friend class OpenCLBackend;
    TuneInfo* mTunedInfo;
};


class OpenCLBackend : public Backend {
public:
    OpenCLBackend(std::shared_ptr<ImagePool>imgPool, std::shared_ptr<BufferPool> bufPool, const CLRuntime *runtime);
    ~OpenCLBackend();

    OpenCLRuntime *getOpenCLRuntime();
    virtual Backend::MemObj* onAcquire(const Tensor *nativeTensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;

    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;

    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) override;

    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;

    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &output, const MNN::Op *op, Backend *backend) const = 0;
    };

    static bool addCreator(std::pair<OpType, GpuMemObject> t, Creator *c);

    BufferPool *getBufferPool() const {
        return mBufferPool;
    }
    virtual bool onSelectDynamicAllocator(int index, int maxIndex) override;

    BackendConfig::PrecisionMode getPrecision() const {
        return mPrecision;
    }

    BackendConfig::MemoryMode getMemory() const {
        return mMemory;
    }
    
    float getBytes(const Tensor* tensor);
    DataType getDataType(const Tensor* tensor);

    cl_channel_type fpType();
    int fpBytes();
    
    void clearRecord() const;
    void enqeueRecord() const;
    void releaseRecord();
    bool isUseRecordQueue(){
        return mUseRecordQueue;
    }
    bool isDevideOpRecord(){
        return mDevideOpRecord;
    }
    void addRecord(cl_recording_qcom &record, std::vector<RecordUpdateInfo *>updateInfo);
    void recordKernel2d(const std::shared_ptr<KernelWrap> &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws, RecordUpdateInfo *updateInfo = nullptr);
    void recordKernel3d(const std::shared_ptr<KernelWrap> &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws, RecordUpdateInfo *updateInfo = nullptr);
    void startRecord(cl_recording_qcom &recording);
    void endRecord(cl_recording_qcom &recording, bool flag = false);

    bool isCreateError() const;
    virtual void* onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor) override;
    virtual bool onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr) override;

private:
    void copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyFromDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyToDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyBetweenDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;

    void _allocHostBuffer(int length, const Tensor* srcTensor) const;

    const CLRuntime* mCLRuntime;

    std::shared_ptr<ImagePool> mImagePoolSecond;
    std::shared_ptr<BufferPool> mBufferPoolSecond;

    ImagePool* mImagePool;
    BufferPool* mBufferPool;

    std::shared_ptr<ImagePool> mImagePoolFirst;
    std::shared_ptr<BufferPool> mBufferPoolFirst;
    std::shared_ptr<ImagePool> mStaticImagePool;
    std::shared_ptr<BufferPool> mStaticBufferPool;
    
    std::shared_ptr<OpenCLRuntime> mOpenCLRuntime;

    mutable std::pair<int, std::shared_ptr<cl::Buffer>> mHostBuffer;
    mutable cl::Buffer *mDeviceBuffer = nullptr;
    mutable std::shared_ptr<cl::Image> mDeviceTexture;
    BackendConfig::PrecisionMode mPrecision;
    BackendConfig::MemoryMode mMemory;
    bool mIsCreateError{false};
    mutable std::vector<RecordInfo> mRecordings;
    bool mUseRecordQueue = false;
    bool mDevideOpRecord = false;
    uint32_t mRecordNums = 0;
    uint32_t mUseRecordableQueueSize;
private:

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

#ifdef MNN_OPENCL_SEP_BUILD
#define REGISTER_OPENCL_OP_CREATOR(name, opType, memObj)  \
    OpenCLCreatorRegister<name> ___OpenCL##name##__##opType##__##memObj##__(opType, memObj)
#else
#define REGISTER_OPENCL_OP_CREATOR(name, opType, memObj)                   \
    void ___OpenCL##name##__##opType##__##memObj##__() {                   \
        static name _temp;                                                 \
        OpenCLBackend::addCreator(std::make_pair(opType, memObj), &_temp); \
    }
#endif

#ifdef MNN_OPENCL_SEP_BUILD
#define REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(name, opType, memObj)  \
    OpenCLCreatorRegister<name> ___OpenCL##name##__##opType##__##memObj##__(opType, memObj)
#else
#define REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(name, opType, memObj)                   \
    void ___OpenCL##name##__##opType##__##memObj##__() {                   \
        static name _temp;                                                 \
        OpenCLBackend::addCreator(std::make_pair(opType, memObj), &_temp); \
    }
#endif


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
