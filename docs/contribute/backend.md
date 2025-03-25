# 自定义后端
Runtime-Backend是MNN对计算设备的抽象。MNN当前已经支持CPU、Vulkan、OpenCL、Metal、CUDA等Backend，**只在计算设备暂未支持时新增Backend**，新增Op，请参阅[新增Op文档](op)。



## 声明
所有新增Backend都需继承`Backend`类，并实现所有纯虚函数。
```cpp
class XPUBackend final : public Backend {
  XPUBackend(MNNForwardType type, MemoryMode mode);
  virtual ~XPUBackend();
  virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
  virtual void onExecuteBegin() const override;
  virtual void onExecuteEnd() const override;
  virtual void onResizeBegin() override;
  virtual ErrorCode onResizeEnd() override;

  virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
  virtual bool onClearBuffer() override;
  virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
}
```

## 构造与销毁
Backend构造时，可以额外指定内存环境，在内存受限环境中，应避免非必要的内存使用。可以在构造函数中，完成对计算设备访问的必要初始化，如GPU下预加载shader等。
```cpp
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

## Execution创建
Backend需要通过`onCreate`为op创建出exection实例：
```cpp
virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
```

可以在方法内根据op类型创建，但更建议提供注册接口：
```cpp
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

这样，Op Execution中，就可以通过注册追加Op类型：
```cpp
class XPUPoolingCreator : public XPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new XPUPooling(backend, op->main_as_Pool());
    }
};
static XPUCreatorRegister<XPUPoolingCreator> __reg(OpType_Pooling);
```

## 内存管理
Backend通过`onAcquire`创建`MemObj`内存对象，定义其析构函数以便为tensor释放内存。内存有三种存储模式：`STATIC`内存不复用，一般用于op常量存储；`DYNAMIC`内存可复用，一般用于变量存储；`DYNAMIC_SEPERATE`内存在pipeline间可复用，一般用于pipeline常量存储。

```cpp
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
     * @return MemObj for release, if failed, return nullptr.
     */
    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) = 0;
```

Backend在调用`onClearBuffer`时，需要释放所有`DYNAMIC`和`DYNAMIC_SEPERATE`存储模式的内存：
```cpp
/**
 * @brief clear all dynamic buffers.
 * @return success or not.
 */
virtual bool onClearBuffer() = 0;
```

此外，backend还需要负责tensor数据的拷贝：
```cpp
/**
 * @brief copy buffer from tensor to tensor.
 * @param srcTensor source buffer provider.
 * @param dstTensor dest buffer provider.
 */
virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const = 0;
```

**拷贝可能在backend内部，也可能在backend与CPU backend之间。**
**拷贝需要处理Tensor间的布局转换，相同布局时，可以直接拷贝数据；不同布局，如**`**NHWC**`**和**`**NC4HW4**`**，则一般需要做特殊转换。**

## Pipeline回调
Backend在pipeline执行的各个周期都会收到回调，`onResizeBegin`和`onResizeEnd`在调整内存分配前后调用（op的`onResize`会在此间调用）；`onExecuteBegin`和`onExecuteEnd`在op执行前后调用（op的`onExecute`会在此间调用）；`onWaitFinish`相对特殊，由用户主动调用，异步执行的pipeline需要同步等待完成。
```cpp
/**
 * @brief callback before resize ops.
 */
virtual void onResizeBegin();
/**
 * @brief callback after resize ops.
 */
virtual ErrorCode onResizeEnd();
/**
 * @brief callback before executing ops.
 */
virtual void onExecuteBegin() const = 0;
/**
 * @brief callback after executing ops.
 */
virtual void onExecuteEnd() const = 0;

```

## Runtime（运行时）
对于使用同一种后端，且存在先后顺序，不会同时运行的模型，MNN提供机制使其共享部分计算资源，比如线程池，内存池等等。
这部分计算资源使用Runtime存储。而Backend则由Runtime创建

### CompileType

Runtime 可以通过指定 CompileType ，决定 MNN 是否跳过几何计算步骤：

```
enum CompilerType {
    // 部分执行几何计算，分解形变算子，但不分解 BatchMatMul / Gather 等算子
    Compiler_Geometry = 0,

    // 完全跳过几何计算步骤，直接使用原始算子
    Compiler_Origin = 1,

    // 完全执行几何计算，仅此模式下，可以在算子不支持时自动回退到CPU计算
    Compiler_Loop = 2,
};
```

### 实现Runtime
Runtime主要实现如下接口：

```
    virtual Backend* onCreate(const BackendConfig* config = nullptr, Backend* origin = nullptr) const = 0;

    /**
     @brief reset runtime
     */
    virtual void onReset(int numberThread, const BackendConfig* config, bool full) {
        // Do nothing
    }

    /**
     @brief clear unuseful resource
     @param level clear level: 0 - 100, bigger mean clear more, smaller mean cache more
     */
    virtual void onGabageCollect(int level) = 0;

```

- onCreate ：创建 Backend
- onReset ：重设默认配置
- onGabageCollect ：清理资源以节省内存


### 注册Runtime
注册方法中调用`MNNInsertExtraRuntimeCreator`就可以完成Runtime的注册，这里的注册方法需要在Backend.cpp中声明并调用：
```cpp
class XPURuntimeCreator : public RuntimeCreator {
    virtual Runtime* onCreate(const Backend::Info &info) const {
        return new XPURuntime;
    }
};
void registerXPURuntimeCreator() {
    MNNInsertExtraBackendCreator(MNN_FORWARD_XPU, new XPURuntimeCreator);
};
```

使用cmake编译时，完成代码修改后，也需要相应修改CMakeLists.txt。
