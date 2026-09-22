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
#ifdef MNN_QNN_DSP_RUNTIME
#include "DSP/QnnDspGraph.h"
#endif

#include "core/Backend.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "QNNUtils.hpp"
#include "QNNWrapper.hpp"
#include "QnnTensorConvert.hpp"
#include "QNNPerf.hpp"
#ifdef MNN_QNN_DSP_RUNTIME
#include "QNNDspPerf.hpp"
#endif
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#ifdef ENABLE_QNN_CONVERT_MODE
#include "QNNConvertorInterface.hpp"
#include "QNNConvertor.hpp"
#endif

#define REGISTER_QNN_OP_CREATOR(name, opType)       \
    void ___##name##__##opType##__() {              \
        QnnBackend::addCreator(opType, new name);   \
    }

namespace MNN {
struct QnnBackendOptions;
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNTensorDumper;
class QnnRuntime;
struct QnnContext;

class QnnBackend : public Backend {
public:
    QnnBackend(const QnnRuntime* runtime);
    virtual ~QnnBackend();
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
    ErrorCode runGraphOnce() const;
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
    void ensureContextAndGraph();
    ErrorCode finalizeGraph();
    void executeGraph() const;
    void freeContextAndGraph();
    bool checkQnnCall(Qnn_ErrorHandle_t result, const char* call) const;

public:
    void addNodeToGraph(Qnn_OpConfigVersion_t version, const char* nodeName, const char* packageName, const char* nodeType, std::vector<Qnn_Param_t> & params, std::vector<Qnn_Tensor_t> & inputs, std::vector<Qnn_Tensor_t> & outputs);
    void addTensor(Qnn_Tensor_t * tensor);
    Qnn_Tensor_t* getMaskTensor(int maxKVSize);
    Qnn_Tensor_t* addExtraInput(Tensor* tensor);
    Qnn_Tensor_t* addExtraOutput(Tensor* tensor);
    int getTensorIdx(const Tensor * tensor) const;
    Qnn_Tensor_t * getNativeTensor(const Tensor * tensor);
    std::shared_ptr<QNNTensorWrapper> getTensorWrapper(const Tensor * tensor);
    bool useCache() const;
    bool getUseFP16() const;
    bool isTensorDumpEnabled() const;
    bool isDspBackend() const;
    bool isExplicitQnnSession() const;
    bool isDedicatedQnnSession() const;
    bool requiresQuantizedGraph() const;
    bool canDumpTensor(Qnn_DataType_t dataType,
                       const std::string& name) const;
    bool prepareDebugTensor(
        const std::shared_ptr<QNNTensorWrapper>& tensor,
        Tensor::DimensionType dimType = gQnnTensorDimType);
    bool registerDebugTensor(
        const std::shared_ptr<QNNTensorWrapper>& tensor);
    const std::string& v66LayerNormOpPackageName() const;
    void buildOutputDequant();
    void buildInputCast(const Tensor *tensor);
    void buildOutputCast();
    void pushReleaseFunc(std::function<void()> func){
        mReleaseFunc.push_back(func);
    }
    virtual const Runtime* getRuntime() override;

private:
    void clean();
private:
    const QnnRuntime * mRuntime;
    mutable ErrorCode mExecutionStatus = NO_ERROR;
    mutable bool mGraphExecuted = false;
    mutable bool mResetStatusOnNextExecute = true;
    std::shared_ptr<BufferAllocator> mOfflineHostAllocator;

    std::unique_ptr<QNNPerf> mPerf;
#ifdef MNN_QNN_DSP_RUNTIME
    std::unique_ptr<QNNDspPerf> mDspPerf;
#endif
    std::unique_ptr<QNNTensorDumper> mTensorDumper;
    bool mDumpIntermediateOutputs = false;
    bool mUseHtpBackend = true;
    bool mUseFP16 = false;
    bool mUseDirectInt8NhwcIo = false;
    bool mRequireInt8Graph = false;
    bool mGraphFinalized = false;
    // The online context/graph is created lazily on the first scheduled op.
    // Models running through the offline plugin path never reach it, so no
    // online QnnContext is created and the HTP prepare library stays unloaded.
    bool mOnlineGraphReady = false;
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
#ifdef MNN_QNN_DSP_RUNTIME
    QnnDspGraph_CustomConfig_t mQnnDspEncodingCustomConfig =
        QNN_DSP_GRAPH_CUSTOM_CONFIG_INIT;
    QnnGraph_Config_t mQnnDspEncodingGraphConfig{};
    QnnDspGraph_CustomConfig_t mQnnDspPriorityCustomConfig =
        QNN_DSP_GRAPH_CUSTOM_CONFIG_INIT;
    QnnGraph_Config_t mQnnDspPriorityGraphConfig{};
#endif
    const std::string mQnnGraphName = "MNN_QNN_GRAPH";

    // Tensor related
    // add <mutable> due to <getTensorIdx> has to be const
    // <getTensorIdx> has to be const due to <onCopyBuffer> has to be const
    mutable int mTensorCounter = 0;
    mutable std::vector<std::shared_ptr<QNNTensorWrapper>> mQNNTensorWrappers;
    mutable std::map<const Tensor::InsideDescribe::NativeInsideDescribe *, int> mTensorMap;
    mutable std::map<const Tensor::InsideDescribe::NativeInsideDescribe *, std::pair<const Tensor*, std::shared_ptr<Tensor>>> mInputCastTensorMap;
    mutable std::map<const Tensor::InsideDescribe::NativeInsideDescribe *, std::pair<const Tensor*, std::shared_ptr<Tensor>>> mOutputCastTensorMap;
    mutable std::map<const Tensor::InsideDescribe::NativeInsideDescribe *, std::pair<const Tensor*, std::shared_ptr<Tensor>>> mDeQuantOutputTensorMap;
    mutable std::map<const Tensor::InsideDescribe::NativeInsideDescribe *, std::shared_ptr<QNNTensorWrapper>> mInputNhwcTensorMap;
    mutable std::map<const Tensor::InsideDescribe::NativeInsideDescribe *, std::shared_ptr<QNNTensorWrapper>> mOutputNhwcTensorMap;
    std::vector<std::shared_ptr<QNNParamTensorWrapper>> mIoParamTensorWrappers;
    std::vector<int> mInputTensorIndexes;
    std::vector<int> mOutputTensorIndexes;
    std::vector<std::shared_ptr<QNNTensorWrapper>> mDebugTensorWrappers;
    // QNN graph inputs and outputs are allocated once during resize and keep
    // stable native handles for the lifetime of the finalized graph. Cache
    // their contiguous descriptors instead of rebuilding two vectors on
    // every graphExecute call.
    mutable std::vector<Qnn_Tensor_t> mExecuteInputs;
    mutable std::vector<Qnn_Tensor_t> mExecuteOutputs;
    std::vector<std::function<void()>> mReleaseFunc;
    std::shared_ptr<QNNTensorWrapper> mMaskTensor;
    std::vector<std::shared_ptr<QNNTensorWrapper>> mExtraInputs;
    std::vector<std::shared_ptr<QNNTensorWrapper>> mExtraOutputs;
    bool mCpuFallbackDetected = false;
    mutable int mLastQnnError = QNN_SUCCESS;

};


class QnnRuntime : public Runtime {
private:
    QnnRuntime(const Backend::Info& info, QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_LogHandle_t qnnLogHandle,
               Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle,
               const std::string& v66LayerNormOpPackageName, QnnContext* selectedContext,
               std::shared_ptr<void> backendContextOwner);

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
    Qnn_ErrorHandle_t allocContext() const;
    static bool registerCustomOpPackage(QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_BackendHandle_t backendHandle, const std::string & path, const std::string & interfaceProvider, const std::string & target);

private:
    bool mUseCache = false;

    // Backend config
    Backend::Info mInfo;
    std::shared_ptr<QnnBackendOptions> mQnnOptions;
    bool mQnnOfflineContextModel = false;
    BackendConfig::PowerMode mPower;
    BackendConfig::MemoryMode mMemory;
    BackendConfig::PrecisionMode mPrecision;
    QnnBackendKind mBackendKind = QnnBackendKind::None;
    bool mRequireInt8Graph = false;
    bool mUseDirectInt8NhwcIo = false;
    bool mDumpIntermediateOutputs = false;
    std::string mV66LayerNormOpPackageName;
    // Qnn related
    QNN_INTERFACE_VER_TYPE mQnnInterface{};
    Qnn_LogHandle_t mQnnLogHandle = nullptr;
    Qnn_BackendHandle_t mQnnBackendHandle = nullptr;
    Qnn_DeviceHandle_t mQnnDeviceHandle = nullptr;
    QnnContext* mSelectedContext = nullptr;
    std::shared_ptr<void> mBackendContextOwner;
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
