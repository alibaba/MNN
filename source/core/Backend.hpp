//
//  Backend.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Backend_hpp
#define Backend_hpp

#include <stdio.h>
#include <map>
#include <memory>
#include <vector>
#include "ErrorCode.hpp"
#include "MNNForwardType.h"
#include "NonCopyable.hpp"
#include "Tensor.hpp"

namespace MNN {

struct Op;
struct GpuLibrary;
class Execution;

/** abstract backend */
class Backend : public NonCopyable {
public:
    /** info used to create backend */
    struct Info {
        /** forward type. */
        MNNForwardType type = MNN_FORWARD_CPU;
        /** for CPU only. number of threads. */
        int numThread = 4;
        /** user data. */
        BackendConfig* user = NULL;
        enum Mode {
            // The Op will be run in execution->onExecute
            DIRECT = 0,

            // The Op will be recorded. Run in onExecuteBegin and Wait in onExecuteEnd
            INDIRECT = 1
        };
        Mode mode = DIRECT;
    };

    /** backend buffer storage type */
    enum StorageType {
        /**
         use NOT reusable memory.
         - allocates memory when `onAcquireBuffer` is called.
         - releases memory when `onReleaseBuffer` is called or when the backend is deleted.
         - do NOTHING when `onClearBuffer` is called.
         */
        STATIC,
        /**
         use reusable memory.
         - allocates or reuses memory when `onAcquireBuffer` is called. prefers reusing.
         - collects memory for reuse when `onReleaseBuffer` is called.
         - releases memory when `onClearBuffer` is called or when the backend is deleted.
         */
        DYNAMIC,
        /**
         use NOT reusable memory.
         - allocates memory when `onAcquireBuffer` is called.
         - do NOTHING when `onReleaseBuffer` is called.
         - releases memory when `onClearBuffer` is called or when the backend is deleted.
         */
        DYNAMIC_SEPERATE
    };

public:
    /**
     * @brief initializer.
     * @param type  forward type.
     */
    Backend(MNNForwardType type) : mType(type) {
        // nothing to do
    }

    /**
     * @brief deinitializer.
     */
    virtual ~Backend() = default;

public:
    /**
     * @brief measure the cost for op with input and output tensors.
     * @param inputs    input tensors.
     * @param outputs   output tensors.
     * @param op        given op.
     * @return std::make_pair(timeDelayInMs, support);
     */
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op) {
        return std::make_pair(0.0f, false);
    }

    /**
     * @brief create execution for op with input and output tensors.
     * @param inputs    input tensors.
     * @param outputs   output tensors.
     * @param op        given op.
     * @return created execution if op is supported, nullptr otherwise.
     */
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) = 0;

    /**
     * @brief callback before resize ops.
     */
    virtual void onResizeBegin() {
        // nothing to do
    }
    /**
     * @brief callback after resize ops.
     */
    virtual void onResizeEnd() {
        // nothing to do
    }

    /**
     * @brief callback before executing ops.
     */
    virtual void onExecuteBegin() const = 0;
    /**
     * @brief callback after executing ops.
     */
    virtual void onExecuteEnd() const = 0;
    /**
     * @brief wait for all async execution to be finished.
     * @return success or not.
     */
    virtual bool onWaitFinish() {
        return true;
    }
    /**
     * @brief load GPU library resource.
     * @param library   loading load GPU library.
     * @return success or not.
     */
    virtual bool onLoadLibrary(const GpuLibrary* library) {
        return false;
    }

public:
    /**
     * @brief allocate buffer of tensor for given storage type.
     * @param tensor        buffer provider.
     * @param storageType   buffer storage type.
     * @return success or not.
     */
    virtual bool onAcquireBuffer(const Tensor* tensor, StorageType storageType) = 0;

    /**
     * @brief release buffer of tensor for given storage type.
     * @param tensor        buffer provider.
     * @param storageType   buffer storage type.
     * @return success or not.
     */
    virtual bool onReleaseBuffer(const Tensor* tensor, StorageType storageType) = 0;

    /**
     * @brief callback after all buffers needed by backend ops were allocated.
     * @return success or not. (result not used currently)
     */
    virtual bool onAllocateBuffer() {
        return true;
    }

    /**
     * @brief clear all dynamic buffers.
     * @return success or not.
     */
    virtual bool onClearBuffer() = 0;

    /**
     * @brief copy buffer from tensor to tensor.
     * @param srcTensor source buffer provider.
     * @param dstTensor dest buffer provider.
     */
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const = 0;

public:
    /**
     * @brief get forward type.
     * @return forward type.
     */
    inline MNNForwardType type() const {
        return mType;
    }

private:
    const MNNForwardType mType;
};

/** abstract backend register */
class BackendCreator {
public:
    /**
     @brief initializer.
     */
    virtual ~BackendCreator() = default;

    /**
     @brief create backend with given info.
     @param info    info to create backend.
     @return created backend
     */
    virtual Backend* onCreate(const Backend::Info& info) const = 0;


    /**
     @brief Turn info to supported.
     @param info    info to valid.
     @return success or not
     */
    virtual bool onValid(Backend::Info& info) const {
        info.mode = Backend::Info::DIRECT;
        return true;
    }
protected:
    /**
     @brief deinitializer.
     */
    BackendCreator() = default;
};

/**
 * @brief get registered backend creator for given forward type.
 * @param type  given forward type.
 * @return backend creator pointer if registered, nullptr otherwise.
 */
MNN_PUBLIC const BackendCreator* MNNGetExtraBackendCreator(MNNForwardType type);

/**
 * @brief register backend creator for given forward type.
 * @param type given forward type.
 * @param creator registering backend creator.
 * @return true if backend creator for given forward type was not registered before, false otherwise.
 */
MNN_PUBLIC bool MNNInsertExtraBackendCreator(MNNForwardType type, const BackendCreator* creator,
                                             bool needCheck = false);

} // namespace MNN

#endif /* Backend_hpp */
