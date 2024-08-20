//
//  Interpreter.hpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Interpreter_hpp
#define MNN_Interpreter_hpp

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <MNN/ErrorCode.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

namespace MNN {

/** session schedule config */
struct ScheduleConfig {
    /** which tensor should be kept */
    std::vector<std::string> saveTensors;
    /** forward type */
    MNNForwardType type = MNN_FORWARD_CPU;
    /** CPU:number of threads in parallel , Or GPU: mode setting*/
    union {
        int numThread = 4;
        int mode;
    };

    /** subpath to run */
    struct Path {
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;

        enum Mode {
            /**
             * Op Mode
             * - inputs means the source op, can NOT be empty.
             * - outputs means the sink op, can be empty.
             * The path will start from source op, then flow when encounter the sink op.
             * The sink op will not be compute in this path.
             */
            Op = 0,

            /**
             * Tensor Mode
             * - inputs means the inputs tensors, can NOT be empty.
             * - outputs means the outputs tensors, can NOT be empty.
             * It will find the pipeline that compute outputs from inputs.
             */
            Tensor = 1
        };

        /** running mode */
        Mode mode = Op;
    };
    Path path;

    /** backup backend used to create execution when desinated backend do NOT support any op */
    MNNForwardType backupType = MNN_FORWARD_CPU;

    /** extra backend config */
    BackendConfig* backendConfig = nullptr;
};

class Session;
struct Content;
class Tensor;
class Backend;
class Runtime;

class MNN_PUBLIC OperatorInfo {
    struct Info;

public:
    /** Operator's name*/
    const std::string& name() const;

    /** Operator's type*/
    const std::string& type() const;

    /** Operator's flops, in M*/
    float flops() const;

protected:
    OperatorInfo();
    ~OperatorInfo();
    Info* mContent;
};

typedef std::function<bool(const std::vector<Tensor*>&, const std::string& /*opName*/)> TensorCallBack;
typedef std::function<bool(const std::vector<Tensor*>&, const OperatorInfo*)> TensorCallBackWithInfo;
typedef std::pair< std::map<MNNForwardType, std::shared_ptr<Runtime>>,  std::shared_ptr<Runtime>> RuntimeInfo;

/**
 * @brief get mnn version info.
 * @return mnn version string.
 */
MNN_PUBLIC const char* getVersion();

/** net data holder. multiple sessions could share same net. */
class MNN_PUBLIC Interpreter {
public:
    /**
     * @brief create net from file.
     * @param file  given file.
     * @return created net if success, NULL otherwise.
     */
    static Interpreter* createFromFile(const char* file);
    /**
     * @brief create net from buffer.
     * @param buffer    given data buffer.
     * @param size      size of data buffer.
     * @return created net if success, NULL otherwise.
     */
    static Interpreter* createFromBuffer(const void* buffer, size_t size);
    ~Interpreter();
    
    /**
     * @brief destroy Interpreter
     * @param model    given Interpreter to release.
     */
    static void destroy(Interpreter* net);

    enum SessionMode {
        /** About CallBack, Default Session_Debug*/
        /** runSessionWithCallBack is allowed and can get internal op info*/
        Session_Debug = 0,
        /** runSessionWithCallBack is not valid and can't get any info of op in session*/
        Session_Release = 1,

        /** About input tenosr, Default Session_Input_Inside*/
        /** The input tensor is alloced by session, input data after session resized*/
        Session_Input_Inside = 2,
        /** The input tensor is alloced by user, set input data before session resize*/
        Session_Input_User = 3,

        /** The output tensor depends on session, and can't be separate used*/
        Session_Output_Inside = 4,
        /** The output tensor can be separated from session*/
        Session_Output_User = 5,

        /** Try Resize Session when create Session or not, default direct: */
        Session_Resize_Direct = 6,
        Session_Resize_Defer = 7,

        /** Determine the Execution's forward type is determine by user or auto determine */
        Session_Backend_Fix = 8, // Use the backend user set, when not support use default backend
        Session_Backend_Auto = 9, // Auto Determine the Op type by MNN

        /** Determine static memory whether recyle in resizeSession or just cache the memory */
        Session_Memory_Collect = 10, // Recycle static memory when session resize in case memory explosion 
        Session_Memory_Cache = 11, // Cache the static memory for next forward usage

        /** Determine whether use codegen function */
        Session_Codegen_Disable = 12, // Disable codegen in case extra build codegen cost
        Session_Codegen_Enable = 13, // Enable codegen
        
        /** Dynamic Reisze Optimization */
        Session_Resize_Check = 14, // Open Trace for resize
        Session_Resize_Fix = 15, // Apply Resize Optimization
    };
    /**
     * @brief The API shoud be called before create session.
     * @param mode      session mode
     */
    void setSessionMode(SessionMode mode);

    /**
     * @brief The API shoud be called before create session.
     * If the cache exist, try to load cache from file.
     * After createSession, try to save cache to file.
     * @param cacheFile      cache file name
     * @param keySize        depercerate, for future use.
     */
    void setCacheFile(const char* cacheFile, size_t keySize = 128);

    /**
     * @brief The API shoud be called before create session.
     * @param file      external data file name
     * @param keySize        depercerate, for future use.
     */
    void setExternalFile(const char* file, size_t flag = 128);
    /**
     * @brief The API shoud be called after last resize session.
     * If resize session generate new cache info, try to rewrite cache file.
     * If resize session do not generate any new cache info, just do nothing.
     * @param session    given session
     * @param flag   Protected param, not used now 
     */

    ErrorCode updateCacheFile(Session *session, int flag = 0);

    enum HintMode {
        // Max Op number for async tuning
        MAX_TUNING_NUMBER = 0,
        // Strictly check model file or not, default 1. if set 0, will not check model file valid/invalid
        STRICT_CHECK_MODEL = 1,
        MEM_ALLOCATOR_TYPE = 2,
        // Winograd unit candidates count, default 3. if set 0, will use less unit candidates for less memory at the expense of performance.
        WINOGRAD_MEMORY_LEVEL = 3,

        // Geometry Compute option, default is 0xFFFF
        GEOMETRY_COMPUTE_MASK = 4,

        // 0: Close dynamic quant; 1: per batch quant; 2: per tensor quant
        DYNAMIC_QUANT_OPTIONS = 5,

        // For Mobile CPU with big-litter core, set decrease rate to let MNN divide task differential by CPU's performance
        // 0-100, 50 means litter core has 50% capacity of large core
        // Default is 50
        CPU_LITTLECORE_DECREASE_RATE = 6,

        // 0: Do not quantize kvcache, just store float
        // 1: Only quantize key cache, use int8 asymmetric quantization 
        // 2: Only quantize value cache, use fp8 quantization
        // 3: quantize both key and value cache as described above
        KVCACHE_QUANT_OPTIONS = 7,
    };

    enum GeometryComputeMask {
        // Support Region Fuse
        GEOMETRCOMPUTEMASK_FUSEREGION = 1 << 0,

        // Support Region Fuse to input with multi-region, eg: pad + concat
        GEOMETRCOMPUTEMASK_FUSEREGION_MULTI = 1 << 1,

        // Use loop instead of raster + compute if possible
        GEOMETRCOMPUTEMASK_USELOOP = 1 << 2,
        
        // Support Geometry Cache, if shape changed, will try recompute, and then run compute if failed
        GEOMETRCOMPUTEMASK_OPENCACHE = 1 << 3,
        
        // Full option open mask, for example, if want to close useloop, can set mask as (GEOMETRCOMPUTEMASK_ALL - GEOMETRCOMPUTEMASK_USELOOP)
        GEOMETRCOMPUTEMASK_ALL = 0xFFFF,
    };

    /**
     * @brief The API shoud be called before create session.
     * @param mode      Hint type
     * @param value     Hint value
     */
    void setSessionHint(HintMode mode, int value);
public:
    /**
     * @brief create runtimeInfo separately with schedule config.
     * @param configs session schedule configs.
     */
    static RuntimeInfo createRuntime(const std::vector<ScheduleConfig>& configs);

    /**
     * @brief create session with schedule config. created session will be managed in net.
     * @param config session schedule config.
     * @return created session if success, NULL otherwise.
     */
    Session* createSession(const ScheduleConfig& config);

    /**
     * @brief create session with schedule config and user-specified runtime.
     * @param config session schedule config, runtime runtimeInfo used by the created session.
     * @return created session if success, NULL otherwise.
     */
    Session* createSession(const ScheduleConfig& config, const RuntimeInfo& runtime);

    /**
     * @brief create multi-path session with schedule configs. created session will be managed in net.
     * @param configs session schedule configs.
     * @return created session if success, NULL otherwise.
     */
    Session* createMultiPathSession(const std::vector<ScheduleConfig>& configs);

    /**
     * @brief create multi-path session with schedule configs and user-specified runtime.
              created session will be managed in net.
     * @param configs session schedule configs.
     * @return created session if success, NULL otherwise.
     */
    Session* createMultiPathSession(const std::vector<ScheduleConfig>& configs, const RuntimeInfo& runtime);

    /**
     * @brief release session.
     * @param session   given session.
     * @return true if given session is held by net and is freed.
     */
    bool releaseSession(Session* session);

    /**
     * @brief call this function to get tensors ready. output tensor buffer (host or deviceId) should be retrieved
     *        after resize of any input tensor.
     * @param session given session.
     */
    void resizeSession(Session* session);

    /**
     * @brief call this function to get tensors ready. output tensor buffer (host or deviceId) should be retrieved
     *        after resize of any input tensor.
     * @param session given session.
     * @param needRelloc, 1 means need realloc.
     */
    void resizeSession(Session* session, int needRelloc);

    
    /**
     * @brief call this function if don't need resize or create session any more, it will save a few memory that equal
     * to the size of model buffer
     */
    void releaseModel();

    /**
     * @brief Get the model buffer for user to save
     * @return std::make_pair(modelBuffer, modelSize).
     * @example:
     * std::ofstream output("trainResult.alinn")
     * auto buffer = net->getModelBuffer();
     * output.write((const char*)buffer.first, buffer.second);
     */
    std::pair<const void*, size_t> getModelBuffer() const;

    /**
     * @brief Get the model's version info.
     * @return const char* of model's version info like "2.0.0";
     * If model is not loaded or model no version info, return "version info not found".
     */
    const char* getModelVersion() const;

    /**
     * @brief update Session's Tensor to model's Const Op
     * @param session   given session.
     * @return result of running.
     */
    ErrorCode updateSessionToModel(Session* session);

    /**
     * @brief run session.
     * @param session   given session.
     * @return result of running.
     */
    ErrorCode runSession(Session* session) const;

    /*
     * @brief run session.
     * @param session   given session.
     * @param before    callback before each op. return true to run the op; return false to skip the op.
     * @param after     callback after each op. return true to continue running; return false to interrupt the session.
     * @param sync      synchronously wait for finish of execution or not.
     * @return result of running.
     */
    ErrorCode runSessionWithCallBack(const Session* session, const TensorCallBack& before, const TensorCallBack& end,
                                     bool sync = false) const;

    /*
     * @brief run session.
     * @param session   given session.
     * @param before    callback before each op. return true to run the op; return false to skip the op.
     * @param after     callback after each op. return true to continue running; return false to interrupt the session.
     * @param sync      synchronously wait for finish of execution or not.
     * @return result of running.
     */
    ErrorCode runSessionWithCallBackInfo(const Session* session, const TensorCallBackWithInfo& before,
                                         const TensorCallBackWithInfo& end, bool sync = false) const;

    /**
     * @brief get input tensor for given name.
     * @param session   given session.
     * @param name      given name. if NULL, return first input.
     * @return tensor if found, NULL otherwise.
     */
    Tensor* getSessionInput(const Session* session, const char* name);
    /**
     * @brief get output tensor for given name.
     * @param session   given session.
     * @param name      given name. if NULL, return first output.
     * @return tensor if found, NULL otherwise.
     */
    Tensor* getSessionOutput(const Session* session, const char* name);

    enum SessionInfoCode {
        /** memory session used in MB, float* */
        MEMORY = 0,

        /** float operation needed in session in M, float* */
        FLOPS = 1,

        /** Backends in session in M, int*, length >= 1 + number of configs when create session */
        BACKENDS = 2,

        /** Resize Info, int*, 0: ready to execute, 1: need malloc, 2: need resize */
        RESIZE_STATUS = 3,
        
        /** Mode / NumberThread, int* */
        THREAD_NUMBER = 4,

        ALL
    };

    /**
     * @brief get session info
     * @param session   given session.
     * @param code      given info code.
     * @param ptr     given info ptr, see SessionInfoCode for detail
     * @return true if support the code, false otherwise.
     */
    bool getSessionInfo(const Session* session, SessionInfoCode code, void* ptr);

    /**
     * @brief get all output tensors.
     * @param session   given session.
     * @return all output tensors mapped with name.
     */
    const std::map<std::string, Tensor*>& getSessionOutputAll(const Session* session) const;
    /**
     * @brief get all input tensors.
     * @param session   given session.
     * @return all input tensors mapped with name.
     */
    const std::map<std::string, Tensor*>& getSessionInputAll(const Session* session) const;

public:
    /**
     * @brief resize given tensor.
     * @param tensor    given tensor.
     * @param dims      new dims. at most 6 dims.
     */
    void resizeTensor(Tensor* tensor, const std::vector<int>& dims);

    /**
     * @brief resize given tensor by nchw.
     * @param batch  / N.
     * @param channel   / C.
     * @param height / H.
     * @param width / W
     */
    void resizeTensor(Tensor* tensor, int batch, int channel, int height, int width);

    /**
     * @brief get backend used to create given tensor.
     * @param session   given session.
     * @param tensor    given tensor.
     * @return backend used to create given tensor, may be NULL.
     */
    const Backend* getBackend(const Session* session, const Tensor* tensor) const;

    /**
     * @brief get business code (model identifier).
     * @return business code.
     */
    const char* bizCode() const;

    /**
     * @brief get model UUID
     * @return Model UUID.
     */
    const char* uuid() const;

private:
    static Interpreter* createFromBufferInternal(Content* net, bool enforceAuth);

    Content* mNet = nullptr;
    Interpreter(Content* net);

    Interpreter(const Interpreter&)  = delete;
    Interpreter(const Interpreter&&) = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter& operator=(const Interpreter&&) = delete;
    void waitSessionFinish(const Session* session) const;
#ifdef MNN_INTERNAL_ENABLED
    void logForRunSession(const Session* session, float time, const char* api) const;
#endif
};
} // namespace MNN

#endif /* Interpreter_hpp */
