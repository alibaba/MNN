//
//  OpenCLBackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLBackend_hpp
#define OpenCLBackend_hpp

#include "Backend.hpp"
#include "MNN_generated.h"

#include <list>
#include <vector>
#include "BufferPool.hpp"
#include "ImageBufferConvertor.hpp"
#include "ImagePool.hpp"
#include "Macro.h"
#include "core/ImageBufferConvertor.hpp"
#include "core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class OpenCLBackend final : public Backend {
public:
    OpenCLBackend(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power);
    ~OpenCLBackend();

    OpenCLRuntime *getOpenCLRuntime();
    virtual bool onAcquireBuffer(const Tensor *nativeTensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor *nativeTensor, StorageType storageType) override;
    virtual bool onAllocateBuffer() override;
    virtual bool onClearBuffer() override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;

    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    virtual bool onWaitFinish() override;

    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;

    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &output, const MNN::Op *op, Backend *backend) const = 0;
    };

    static bool addCreator(OpType t, Creator *c);

    BufferPool *getBufferPool() const {
        return mBufferPool.get();
    }
    BackendConfig::PrecisionMode getPrecision() const {
        return mPrecision;
    }

    bool isCreateError() const;

private:
    void _allocHostBuffer(int length) const;
    cl::Kernel mImageToNCHWBufferFloat;
    cl::Kernel mImageToNC4HW4BufferFloat;
    cl::Kernel mImageToNHWCBufferFloat;
    cl::Kernel mNC4HW4BufferToImageFloat;
    cl::Kernel mNCHWBufferToImageFloat;
    cl::Kernel mNHWCBufferToImageFloat;
    std::shared_ptr<ImagePool> mImagePool;
    std::shared_ptr<ImagePool> mStaticImagePool;
    std::shared_ptr<BufferPool> mBufferPool;
    std::shared_ptr<OpenCLRuntime> mOpenCLRuntime;

    mutable std::pair<int, std::shared_ptr<cl::Buffer>> mHostBuffer;
    BackendConfig::PrecisionMode mPrecision;
    bool mIsCreateError{false};
};

template <class T>
class OpenCLCreatorRegister {
public:
    OpenCLCreatorRegister(OpType type) {
        T *t = new T;
        OpenCLBackend::addCreator(type, t);
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
