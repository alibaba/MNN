//
//  QNNBackend.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNBACKEND_HPP
#define MNN_QNNBACKEND_HPP

// Qnn API Interface
#include "QnnInterface.h"
#include "HTP/QnnHtpGraph.h"

#include "core/Backend.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "QNNUtils.hpp"
#include "QNNWrapper.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "QNNPerf.hpp"
#include <memory>
#ifdef ENABLE_QNN_CONVERT_MODE
#include "QNNConvertorInterface.hpp"
#include "QNNConvertor.hpp"
#endif

#define REGISTER_QNN_OP_CREATOR(name, opType)       \
    void ___##name##__##opType##__() {              \
        QnnBackend::addCreator(opType, new name);   \
    }

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QnnRuntime;

class QnnBackend : public Backend {
public:
    QnnBackend(const QnnRuntime* runtime);
    virtual ~QnnBackend();
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;
    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

private:
    void startProfile() const;
    void inputIO(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void outputIO(const Tensor* srcTensor, const Tensor* dstTensor) const;

public:
    class Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addCreator(OpType t, Creator* c);

private:
    void createContextAndGraph();
    void finalizeGraph();
    void executeGraph() const;
    void freeContextAndGraph();

public:
    void addNodeToGraph(Qnn_OpConfigVersion_t version, const char* nodeName, const char* packageName, const char* nodeType, std::vector<Qnn_Param_t> & params, std::vector<Qnn_Tensor_t> & inputs, std::vector<Qnn_Tensor_t> & outputs);
    void addStaticTensorToGraph(Qnn_Tensor_t * staticTensor);
    void addStageTensorToGraph(Qnn_Tensor_t * stageTensor);
    int getTensorIdx(const Tensor * tensor) const;
    Qnn_Tensor_t * getNativeTensor(const Tensor * tensor);
    std::shared_ptr<QNNTensorWrapper> getTensorWrapper(const Tensor * tensor);
    bool useCache() const;
    bool getUseFP16() const;
    void buildOutputDequant();
    void pushReleaseFunc(std::function<void()> func){
        mReleaseFunc.push_back(func);
    }

private:
    void clean();

private:
    const QnnRuntime * mRuntime;

    std::unique_ptr<QNNPerf> mPerf;

    bool mUseFP16;
    const BackendConfig::PowerMode mPower;

    // Qnn Profile
    Qnn_ProfileHandle_t mQnnProfileHandle = nullptr;
    // Qnn Signal
    Qnn_SignalHandle_t mQnnSignalHandle = nullptr;
    // Qnn Context
    Qnn_ContextHandle_t mQnnContextHandle = nullptr;
    const QnnContext_Config_t** mQnnContextConfig = nullptr;
    // Qnn Graph
    Qnn_GraphHandle_t mQnnGraphHandle = nullptr;
    QnnHtpGraph_CustomConfig_t mQnnHtpGraphCustomConfig{};
    QnnGraph_Config_t mQnnGraphConfig{};
    const std::string mQnnGraphName = "MNN_QNN_GRAPH";

    // Tensor related
    // add <mutable> due to <getTensorIdx> has to be const
    // <getTensorIdx> has to be const due to <onCopyBuffer> has to be const
    mutable int mTensorCounter = 0;
    mutable std::vector<std::shared_ptr<QNNTensorWrapper>> mQNNTensorWrappers;
    mutable std::map<const Tensor::InsideDescribe::NativeInsideDescribe *, int> mTensorMap;
    mutable std::map<const Tensor::InsideDescribe::NativeInsideDescribe *, std::pair<const Tensor*, std::shared_ptr<Tensor>>> mDeQuantOutputTensorMap;
    std::vector<int> mInputTensorIndexes;
    std::vector<int> mOutputTensorIndexes;
    std::vector<std::function<void()>> mReleaseFunc;
};


class QnnRuntime : public Runtime {
private:
    QnnRuntime(const Backend::Info& info, QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_LogHandle_t qnnLogHandle, Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle);

public:
    // Release all resources.
    ~QnnRuntime();
    // Create QnnBackend.
    Backend* onCreate(const BackendConfig* config = nullptr, Backend* origin = nullptr) const override;
    // Create QnnRuntime. Return nullptr if it fails.
    static QnnRuntime* create(const Backend::Info& info);

    void onGabageCollect(int level) override;
    virtual CompilerType onGetCompilerType() const override;
    // If buffer is not nullptr, try copy cache, else delete cache
    virtual bool onSetCache(const void* buffer, size_t size) override;
    
    virtual std::pair<const void*, size_t> onGetCache() override;
    virtual bool onSetCachePath(const char* path, int mode) override;

private:
    void freeContext() const;
    void allocContext() const;
    static bool registerCustomOpPackage(QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_BackendHandle_t backendHandle, const std::string & path, const std::string & interfaceProvider, const std::string & target);

private:
    bool mUseCache = false;

    // Backend config
    Backend::Info mInfo;
    BackendConfig::PowerMode mPower;
    BackendConfig::MemoryMode mMemory;
    BackendConfig::PrecisionMode mPrecision;
    // Qnn related
    QNN_INTERFACE_VER_TYPE mQnnInterface{};
    Qnn_LogHandle_t mQnnLogHandle = nullptr;
    Qnn_BackendHandle_t mQnnBackendHandle = nullptr;
    Qnn_DeviceHandle_t mQnnDeviceHandle = nullptr;
    // Qnn Context
    mutable Qnn_ContextHandle_t mQnnContextHandle = nullptr;
    const QnnContext_Config_t** mQnnContextConfig = nullptr;
    mutable std::vector<int8_t> mBinaryBuffer;
friend class QnnBackend;
};


#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNBACKEND_HPP
