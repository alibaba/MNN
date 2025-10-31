#ifndef NeuropilotBackend_hpp
#define NeuropilotBackend_hpp 
#ifdef MNN_NEUROPILOT_CONVERT_MODE
#include "core/Backend.hpp"
#include "converter/ConvertTflite.hpp"
#include <memory>
#include <map>
namespace MNN {
class NeuropilotRuntime : public Runtime {
public:
    ~NeuropilotRuntime() {
        // Do nothing
    }
    NeuropilotRuntime(const Backend::Info& info) {
        // Do nothing
    }

    virtual Backend* onCreate(const BackendConfig* config = nullptr, Backend* origin = nullptr) const override;
    virtual CompilerType onGetCompilerType() const override {
        return Compiler_Origin;
    }
    virtual bool onSetCachePath(const char* path, int mode) override {
        pCachePath = path;
        return true;
    }
    virtual void onGabageCollect(int level) override {
        // Do nothing
    }

    std::string pCachePath;
private:

};
class NeuropilotBackend : public Backend {
public:
    NeuropilotBackend(const NeuropilotRuntime* runtime) : Backend(MNN_CONVERT_NEUROPILOT) {
        mRuntime = runtime;
    }
    virtual ~NeuropilotBackend() {
        // Do nothing
    }
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;
    virtual void onExecuteBegin() const override {
        // Do nothing
    }
    virtual void onExecuteEnd() const override {
        // Do nothing
    }
    virtual bool onClearBuffer() override {
        return true;
    }
    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override {
        // Do nothing
        return;
    }

    class Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addCreator(OpType t, Creator* c);
    virtual const Runtime* getRuntime() override {
        return mRuntime;
    }
    friend class ConvertExecution;
    struct ExecuteInfo {
        const MNN::Op* op;
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
    };
    void prepareTensorQuantInfo(const Tensor* tensor, std::unique_ptr<tflite::QuantizationParametersT>&& param);
    void setPackTensor(const Tensor* tensor, int packBits = 4);
    void setTensorName(const Tensor* tensor, std::string name);
    void insertExtraInput(Tensor* tensor);
    void insertExtraOutput(Tensor* tensor);
    Tensor* getStateMask(int maxLength);
    Tensor* getConstTensor(std::string name, std::shared_ptr<Tensor> ref);
private:
    // TensorFlow Lite 转换相关的辅助函数
    std::unique_ptr<tflite::ModelT> createTensorFlowLiteModel();
    void saveTensorFlowLiteModel(std::unique_ptr<tflite::ModelT>& model, const std::string& filePath);
    std::map<const Tensor*, int> mTensorIndexMap;
    int _createTensorFromMNNTensor(const Tensor* tensor, tflite::SubGraphT* dstGraph, std::vector<std::unique_ptr<tflite::BufferT>>& dstBuffers);

private:
    std::shared_ptr<Tensor> mStateMask;
    std::map<std::string, std::shared_ptr<Tensor>> mSharedConst;
    std::map<const Tensor*, int> mExtraInputs;
    std::map<const Tensor*, int> mExtraOutputs;
    std::map<int, const Tensor*> mDequantTensor;
    std::map<const Tensor*, int> mPackInfo;
    std::map<const Tensor*, std::string> mUserTensorName;
    const NeuropilotRuntime* mRuntime = nullptr;
    std::vector<ExecuteInfo> mInfos;
    std::map<const Tensor*, std::unique_ptr<tflite::QuantizationParametersT>> mQuantInfo;
    std::map<int, int> mIOIndexMap; // First: MNN's index, Second: Tflite's index
    
};

};
#endif


#endif
