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
        void setShapeDirty();
        void setContentDirty();
        
        ErrorCode compute();
        ErrorCode resize();
        Tensor* output(EXPRP outputExpr, int index, bool host = true) const;
        void dup(EXPRP src, EXPRP dst);
        void recycle(Expr* expr);
        struct TensorContent {
            std::shared_ptr<Tensor> tensor;
            int refCount = 0;
            void reset();
        };
        struct Unit {
            std::vector<Tensor*> inputs;
            std::vector<bool> inputFromCache;
            std::vector<Tensor*> outputs;
            const Expr* origin;
            std::shared_ptr<Execution> exe;
        };
        static void create(const std::vector<EXPRP>& outputs, std::map<EXPRP, ComputeCache::Unit>& units, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::vector<ComputeCache::TensorContent>&& tensors, std::shared_ptr<Backend> bn, std::shared_ptr<Backend> backendBn);

        ~ ComputeCache();
        void addLink(std::shared_ptr<ComputeCache> cache);
        bool valid() const {
            return !mOutputTensors.empty();
        }
    private:
        ComputeCache(){};
        std::set<std::shared_ptr<ComputeCache>> mInputs;
        // First is Host Tensor, Second is Device Tensor
        std::map<Expr*, std::vector<std::pair<Tensor*, Tensor*>>> mOutputTensors;
        std::vector<TensorContent> mTensors;
        std::vector<Unit> mUnits;
        std::vector<std::weak_ptr<ComputeCache>> mLinks;
        bool mContentDirty = true;
        bool mShapeDirty = true;
        std::shared_ptr<Backend> mBackend;
        std::shared_ptr<Backend> mBackupBackend;
    };
    struct Requirement {
        std::vector<bool> contentNeedContent;
        std::vector<bool> shapeNeedContent;
        std::vector<bool> supportError;
    };
    ~Executor();
    Requirement getRequirement(Expr* expr) const;
    ErrorCode computeInfo(Expr* expr);
    void makeCache(std::vector<EXPRP> expr);
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
    void _addToCache(const std::vector<std::shared_ptr<ComputeCache>>& caches);
    void _resetCache();
    void _visit(EXPRP expr, std::map<EXPRP, ComputeCache::Unit>& units, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::vector<ComputeCache::TensorContent>& tensors);

    Executor(std::shared_ptr<Backend> backend);
    std::shared_ptr<Backend> mBackend;
    std::shared_ptr<Backend> mBackupBackend;
    std::mutex mMutex;
    std::vector<std::shared_ptr<Tensor>> mStack;
    std::vector<Tensor*> mInputs;
    std::vector<Tensor*> mOutputs;
    std::shared_ptr<Profiler> mProfiler;
};
} // namespace Express
} // namespace MNN
#endif
