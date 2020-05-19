//
//  Executor.hpp
//  MNN
//
//  Created by MNN on 2019/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef Executor_hpp
#define Executor_hpp
#include <MNN/ErrorCode.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/Tensor.hpp>
#include <vector>
#include <mutex>
#include <set>
#include <MNN/MNNForwardType.h>
namespace MNN {
class Backend;
class Execution;
namespace Express {
class MNN_PUBLIC Executor {
public:
    class ComputeCache {
    public:
        void setShapeDirty(int offset, Variable::Info* info);
        void setContentDirty();
        void setContentReady();
        void syncInput(int offset, const Variable::Info* info);
        void syncOutput(int offset, Variable::Info* info);

        struct TensorContent {
            std::shared_ptr<Tensor> tensor;
            int refCount = 0;
            void reset();
        };
        struct Unit;
        virtual ~ ComputeCache() {}
        ComputeCache() {}
        virtual ErrorCode compute() = 0;
        virtual ErrorCode resize() = 0;
    protected:
        // Get the index tensor with the need of needBackend
        // If the Tensor don't belong to the backend, need use needBackend to alloc it and return
        virtual Tensor* getTensor(int index, bool host) = 0;
        void _setShapeDirty();
        friend class Executor;
        bool mContentDirty = true;
        bool mShapeDirty = true;
    };
    struct Requirement {
        std::vector<bool> contentNeedContent;
        std::vector<bool> shapeNeedContent;
        std::vector<bool> supportError;
    };
    ~Executor();
    Requirement getRequirement(Expr* expr) const;
    ErrorCode computeInfo(Expr* expr);
    void makeCache(const std::vector<EXPRP>& expr, bool forceCPU = false);
    ErrorCode runCache(std::shared_ptr<ComputeCache> cache);
    void setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread);
    enum GCFlag {
        FULL,
        PART
    };
    void gc(GCFlag flag = FULL);
    static std::shared_ptr<Executor> getGlobalExecutor();
    void resetProfile();
    void dumpProfile();
    void addOpCostTime(int op, float costTime);
    class Profiler;
private:
    void _createSingle(EXPRP expr);
    void _create(const std::vector<EXPRP>& outputs, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::vector<ComputeCache::TensorContent>&& tensors, bool forceCPU);

    void _addToCache(const std::vector<std::shared_ptr<ComputeCache>>& caches);
    void _resetCache();
    void _visit(EXPRP expr, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::vector<ComputeCache::TensorContent>& tensors);

    Executor(std::shared_ptr<Backend> backend);
    std::shared_ptr<Backend> mBackend;
    std::shared_ptr<Backend> mBackupBackend;
    std::mutex mMutex;
    std::vector<std::shared_ptr<Tensor>> mStack;
    std::vector<Tensor*> mStackInputs;
    std::vector<Tensor*> mStackOutputs;
    std::shared_ptr<Profiler> mProfiler;
};
} // namespace Express
} // namespace MNN
#endif
