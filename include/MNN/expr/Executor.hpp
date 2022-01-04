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
struct RuntimeAttr;
class MNN_PUBLIC Executor {
public:
    class ComputeCache;
    struct Unit;
    struct DebugTools;
    /**Internal Usage Begin*/
    static void setShapeDirty(ComputeCache* cache);
    static void setContentDirty(ComputeCache* cache);
    static Tensor* getOutput(ComputeCache* cache, int offset);
    static std::pair<std::shared_ptr<Backend>, std::shared_ptr<Backend>> getBackends(ComputeCache* cache);
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
    /**Internal Usage End*/

    void setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread);
    int getCurrentRuntimeStatus(RuntimeStatus statusEnum);
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
    /**Internal Usage Begin*/
    void addOpCostTime(int op, float costTime);
    void addOpCostTime(const std::string& type, float costTime);
    void addOpFlops(const std::string& type, float flops);
    class Profiler;
    /**Internal Usage End*/
    static RuntimeInfo getRuntime();
    void setCallBack(TensorCallBackWithInfo&& before, TensorCallBackWithInfo&& after);
    const DebugTools* getDebugTools() const {
        return mDebug.get();
    }
    struct Cache;
    class RuntimeManager {
    public:
        ~RuntimeManager();
        /**
         * @param configs: schedule configs.
         * @param cacheName: full path for cache file. Note: should choose location for reading and writing.
         */
        static RuntimeManager* createRuntimeManager(const ScheduleConfig& config);

        /**
         * Deceperate, the same as createRuntimeManager(configs[0])
         * @param configs: schedule configs.
         * @param cacheName: full path for cache file. Note: should choose location for reading and writing.
         */
        static RuntimeManager* createRuntimeManager(std::vector<ScheduleConfig>& configs);

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
        friend class Executor;
        void setMode(Interpreter::SessionMode mode);
        void setHint(Interpreter::HintMode mode, int value);
        bool getInfo(Interpreter::SessionInfoCode code, void* ptr);
        const RuntimeAttr* getInside() const {
            return mInside;
        }
    private:
        RuntimeManager();
        RuntimeInfo mRuntime;
        std::shared_ptr<Runtime> mInfo;
        std::shared_ptr<Cache> mCache;
        RuntimeAttr* mInside;
    };
private:
    void _makeCache(const std::vector<EXPRP>& outputs, bool forceCPU);
    void _create(const std::vector<EXPRP>& outputs, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::set<std::shared_ptr<Expr::Inside>>&& inputNode, bool forceCPU);

    void _visit(EXPRP expr, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::set<std::shared_ptr<Expr::Inside>>& inputNode);
    std::map<std::pair<MNNForwardType, int>, std::shared_ptr<Runtime>> mRuntimes;

    Executor(std::shared_ptr<Runtime> backend, MNNForwardType type, int numberThread);
    std::mutex mMutex;
    std::shared_ptr<Profiler> mProfiler;
    std::shared_ptr<DebugTools> mDebug;

    std::pair<MNNForwardType, int> mFirstType;
};
} // namespace Express
} // namespace MNN
#endif
