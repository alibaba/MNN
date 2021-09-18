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
    
    struct Cache;
    class RuntimeManager {
    public:
        RuntimeManager(std::vector<ScheduleConfig> &configs);
        ~RuntimeManager() {};
        
        /**
         * @param configs: schedule configs.
         * @param cacheName: full path for cache file. Note: should choose location for reading and writing.
         */
        static RuntimeManager* createRuntimeManager(std::vector<ScheduleConfig> &configs);

        /**
         * @brief set cache file. when file not exist -- create it, when file exist -- load it.
         * When should use : When choose GPU backend or use AUTO backend.
         * Calling Position: calling after createRuntimeManager.
         */
        void setCache(std::string cacheName);
        
        /**
         * @brief update cache file
         * When should use   : Together with setCache API. calling for first inference and when input shape is changed.
         * Calling Position  : calling after inference done.
         */
        void updateCache();
        std::vector<bool> isBackendSupport(const std::vector<MNNForwardType> type);
        RuntimeInfo getRuntimeInfo() {
            return mRuntime;
        }
    private:
        RuntimeInfo mRuntime;
        std::shared_ptr<Runtime> mInfo;
        std::shared_ptr<Cache> mCache;
        
    };

    
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
