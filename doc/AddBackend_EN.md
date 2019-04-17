[中文版本](AddBackend_CN.md)

# Custom Backend

Backends of MNN are abstraction of computing devices. MNN currently supports Backend for CPU, Vulkan, OpenCL, Metal, etc. **Add new backend only if the computing device is not supported**. To add Op, see [Add Op document](AddOp_EN.md).

## Declaration

All new backends need to inherit the `Backend` class and implement all pure virtual functions.

``` c++
class XPUBackend final : public Backend {
  XPUBackend(MNNForwardType type, MemoryMode mode);
  virtual ~XPUBackend();

  virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
  virtual void onExecuteBegin() const override;
  virtual void onExecuteEnd() const override;

  virtual bool onAcquireBuffer(const Tensor* tensor, StorageType storageType) override;
  virtual bool onReleaseBuffer(const Tensor* tensor, StorageType storageType) override;
  virtual bool onClearBuffer() override;
  virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
}
```

## Construction and Deconstruction

When backend is constructed, you can specify an additional memory option. In a memory-constrained environment, you should avoid unnecessary memory usage. In the constructor, you can complete the necessary initialization of the access to the computing device, such as preloading shaders running on GPU.

``` c++
/** backend memory mode */
enum MemoryMode {
    /** use memory without limit. */
    NORMAL = 0,
    /** use memory thriftily. */
    LIMIT = 1
};

/**
 * @brief initializer.
 * @param type  forward type.
 * @param mode  memory mode.
 */
Backend(MNNForwardType type, MemoryMode mode = NORMAL);
```

## Execution Create

Backend creates an exection instance with method `onCreate`:

``` c++
virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
```

An execution instance should be created and returned in this method. To separate type of op, providing a registration interface is more recommended than using the switch-case pattern.:

``` c++
class XPUBackend final : public Backend {
    // ...

    class Creator {
    public:
        /**
         * @brief create execution for given input, op on metal backend.
         * @param inputs    given input tensors.
         * @param op        given op.
         * @param backend   metal backend.
         * @return created execution if supported, NULL otherwise.
         */
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                    Backend *backend) const = 0;
    };

    /**
     * @brief register creator for given op type.
     * @param type      given op type.
     * @param creator   registering creator.
     */
    static void addCreator(OpType type, Creator *creator);

    // ...
};

template <class T>
class XPUCreatorRegister {
public:
    /**
     * @brief initializer. register T creator for given op type.
     * @param type  given op type.
     */
    XPUCreatorRegister(OpType type) {
        T *test = new T;
        XPUBackend::addCreator(type, test);
    }
};
```

In this way, you can append the Op type by registration in each Op Execution file:

``` c++
class XPUPoolingCreator : public XPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new XPUPooling(backend, op->main_as_Pool());
    }
};
static XPUCreatorRegister<XPUPoolingCreator> __reg(OpType_Pooling);
```

## Memory Management

Backend allocates memory for tensor via `onAcquireBuffer` and frees memory via `onReleaseBuffer`. There are three storage modes in memory: `STATIC` memory is not reused, generally used for op constant storage; `DYNAMIC` memory can be reused, generally used for variable storage; `DYNAMIC_SEPERATE` memory can be reused between pipelines, generally used for pipeline Constant storage. _There is a better way to allocate/release memory –- record memory usage changes only in `onAcquireBuffer` and `onReleaseBuffer`, allocate/release memory until `onAllocateBuffer` is called, so that all memory needed could be merged into continuous memory and the allocate/release could be completed in one call._

``` c++
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
     - collects memory for reuse when `onReleaseBuffer` is called
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
```

After all memory is allocated, Backend will receive the `onAllocateBuffer` callback:

``` c++
/**
 * @brief callback after all buffers needed by backend ops were allocated.
 * @return success or not. (result not used currently)
 */
virtual bool onAllocateBuffer() {
    return true;
}
```

Backend needs to release all the memory of `DYNAMIC` and `DYNAMIC_SEPERATE` storage mode when `onClearBuffer` called:

``` c++
/**
 * @brief clear all dynamic buffers.
 * @return success or not.
 */
virtual bool onClearBuffer() = 0;
```

In addition, backend is also responsible for copying the tensor data:
``` c++
/**
 * @brief copy buffer from tensor to tensor.
 * @param srcTensor source buffer provider.
 * @param dstTensor dest buffer provider.
 */
virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const = 0;
```

**The copy process may be inside Backend or between Backend and CPU Backend. **
**The copy process needs to handle the layout conversion between Tensor. With the same layout, you can directly copy the data; different layouts, such as `NHWC` and `NC4HW4`, need to do special conversion generally. **

## Pipeline Callback

Backend receives callbacks in each cycle of the pipeline execution, `onResizeBegin` and `onResizeEnd` are called before and after adjusting memory allocation (`onResize` of op will be called here); `onExecuteBegin` and `onExecuteEnd` are called before and after op execution (`onExecute` of op will be called here); `onWaitFinish` is relatively special and is called by the user, the asynchronously executed pipeline needs to wait synchronously for completion.

``` c++
/**
 * @brief callback before resize ops.
 */
virtual void onResizeBegin();
/**
 * @brief callback after resize ops.
 */
virtual void onResizeEnd();

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
virtual bool onWaitFinish();
```

## Regist Backend

Finally, define Backend Creator and call `MNNInsertExtraBackendCreator` to complete the registration of Backend:

``` c++
class XPUBackendCreator : public BackendCreator {
    virtual Backend *onCreate(const Backend::Info &info) const {
        return new MetalBackend;
    }
};
static bool __reg = MNNInsertExtraBackendCreator(MNN_FORWARD_METAL, new XPUBackendCreator);
```
