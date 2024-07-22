//
//  Executor.hpp
//  MNN
//
//  Created by MNN on 2019/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MNN_Executor_hpp
#define MNN_Executor_hpp
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
struct ExecutorAttr;
class MNN_PUBLIC Executor {
public:
    class ComputeCache;
    struct DebugTools;
    /**Internal Usage Begin*/
    struct Requirement {
        std::vector<bool> contentNeedContent;
        std::vector<bool> shapeNeedContent;
    };
    ~Executor();
    Requirement getRequirement(Expr* expr) const;
    ErrorCode computeInfo(Expr* expr);
    void makeCache(const std::vector<EXPRP>& expr, bool forceCPU = false);
    /**Internal Usage End*/

    bool lazyEval = true;
    enum LazyMode {
        LAZY_FULL = 0,
        // Don't compute content until user needed.
        LAZY_CONTENT = 1 << 0,
        
        // Expr can only compute once, it can reduce the create cost of expr
        LAZY_COMPUTE_ONCE = 1 << 1,
    };
    uint32_t getLazyMode() const {
        return mLazyMode;
    }
    void setLazyComputeMode(uint32_t mode);
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

    struct SubGraph;
    bool registerSubGraph(const std::string& submoduleName, VARPS outputs, VARPS inputs);
    std::shared_ptr<SubGraph> findSubGraph(const std::string& submoduleName);
    static RuntimeInfo getRuntime();
    void setCallBack(TensorCallBackWithInfo&& before, TensorCallBackWithInfo&& after);
    const DebugTools* getDebugTools() const {
        return mDebug.get();
    }
    ExecutorAttr* getAttr() const;
    class MNN_PUBLIC RuntimeManager {
    public:
        ~RuntimeManager();
        /**
         * @param configs : schedule configs.
         * @param cacheName : full path for cache file. Note: should choose location for reading and writing.
         */
        static RuntimeManager* createRuntimeManager(const ScheduleConfig& config);
        
        /**
         * @param rtmgr : the rtmgr to destroy
         */
        static void destroy(RuntimeManager* rtmgr);

        /**
         * Deceperate, the same as createRuntimeManager(configs[0])
         * @param configs : schedule configs.
         * @param cacheName : full path for cache file. Note: should choose location for reading and writing.
         */
        static RuntimeManager* createRuntimeManager(std::vector<ScheduleConfig>& configs);

        /**
         * @brief set cache file. when file not exist -- create it, when file exist -- load it.
         * When should use : When choose GPU backend or use AUTO backend.
         * Calling Position: calling after createRuntimeManager.
         */
        void setCache(std::string cacheName);
        
        /**
         * @brief set external file.
         */
        void setExternalFile(std::string fileName);

        /**
         * @brief update cache file
         * When should use   : Together with setCache API. calling for first inference and when input shape is changed.
         * Calling Position  : calling after inference done.
         */
        void updateCache();
        std::vector<bool> isBackendSupport(const std::vector<MNNForwardType> type);
        friend class Executor;
        void setMode(Interpreter::SessionMode mode);
        void setHint(Interpreter::HintMode mode, int value);
        bool getInfo(Interpreter::SessionInfoCode code, void* ptr);
        BackendConfig* getBnConfig();
        const RuntimeAttr* getInside() const {
            return mInside;
        }
    private:
        std::mutex mLock;
        RuntimeAttr* mInside;
        friend class StaticModule;
        RuntimeManager();
    };
    static bool getComputeInfo(EXPRP expr, Interpreter::SessionInfoCode code, void* ptr);
private:
    void _refreshRuntime();
    Executor(std::shared_ptr<Runtime> backend, MNNForwardType type, int numberThread);
    void _makeCache(const std::vector<EXPRP>& outputs, bool forceCPU);

    // TODO: Remove mRuntimes, only use mRuntimeInfo
    std::map<MNNForwardType, std::shared_ptr<Runtime>> mRuntimes;
    RuntimeInfo mRuntimeInfo;
    std::shared_ptr<DebugTools> mDebug;
    std::map<std::string, std::shared_ptr<SubGraph>> mSubGraph;
    uint32_t mLazyMode = 0;
    std::shared_ptr<ExecutorAttr> mAttr;
    std::mutex mMutex;
};
} // namespace Express
} // namespace MNN
#endif
