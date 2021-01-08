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
#include <MNN/Interpreter.hpp>
#include <vector>
#include <mutex>
#include <set>
#include <MNN/MNNForwardType.h>
namespace MNN {
class Backend;
class Execution;
class Runtime;
struct Op;
namespace Express {
class MNN_PUBLIC Executor {
public:
    class ComputeCache;
    struct Unit;
    static void setShapeDirty(ComputeCache* cache);
    static void setContentDirty(ComputeCache* cache);
    static Tensor* getOutput(ComputeCache* cache, int offset);
    static void* mapOutput(ComputeCache* cache, int offset, Tensor* dest);
    struct Requirement {
        std::vector<bool> contentNeedContent;
        std::vector<bool> shapeNeedContent;
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

    static std::shared_ptr<Executor> newExecutor(MNNForwardType type,
                                                 const BackendConfig& config,
                                                 int numberThread);
    void resetProfile();
    void dumpProfile();
    void addOpCostTime(int op, float costTime);
    void addOpCostTime(const std::string& type, float costTime);
    void addOpFlops(const std::string& type, float flops);
    class Profiler;
    static RuntimeInfo getRuntime();
private:
    void _makeCache(const std::vector<EXPRP>& outputs, bool forceCPU);
    void _create(const std::vector<EXPRP>& outputs, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::set<std::shared_ptr<Expr::Inside>>&& inputNode, bool forceCPU);

    void _visit(EXPRP expr, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::set<std::shared_ptr<Expr::Inside>>& inputNode);

    Executor(std::shared_ptr<Runtime> backend, MNNForwardType type);
    std::pair<std::shared_ptr<Runtime>, MNNForwardType> mRuntime;
    std::pair<std::shared_ptr<Runtime>, MNNForwardType> mBackupRuntime;
    std::mutex mMutex;
    std::shared_ptr<Profiler> mProfiler;
};
} // namespace Express
} // namespace MNN
#endif
