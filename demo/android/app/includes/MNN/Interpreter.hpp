//
//  Interpreter.hpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Interpreter_hpp
#define Interpreter_hpp

#include <functional>
#include <map>
#include <string>
#include <MNN/ErrorCode.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

namespace MNN {

/** session schedule config */
struct ScheduleConfig {
    /** which tensor should be kept */
    std::vector<std::string> saveTensors;
    /** forward type */
    MNNForwardType type = MNN_FORWARD_CPU;
    /** number of threads in parallel */
    int numThread = 4;

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
             * Tensor Mode (NOT supported yet)
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

public:
    /**
     * @brief create session with schedule config. created session will be managed in net.
     * @param config session schedule config.
     * @return created session if success, NULL otherwise.
     */
    Session* createSession(const ScheduleConfig& config);

    /**
     * @brief create multi-path session with schedule configs. created session will be managed in net.
     * @param configs session schedule configs.
     * @return created session if success, NULL otherwise.
     */
    Session* createMultiPathSession(const std::vector<ScheduleConfig>& configs);

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
     * @brief call this function if don't need resize or create session any more, it will save a few memory that equal
     * to the size of model buffer
     */
    void releaseModel();

    /**
     * @brief Get the model buffer for user to save
     * @return std::make_pair(modleBuffer, modelSize).
     * @example:
     * std::ofstream output("trainResult.alinn")
     * auto buffer = net->getModelBuffer();
     * output.write((const char*)buffer.first, buffer.second);
     */
    std::pair<const void*, size_t> getModelBuffer() const;

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

    /**
     * @brief get all input tensors.
     * @param session   given session.
     * @return all input tensors mapped with name.
     */
    const std::map<std::string, Tensor*>& getSessionOutputAll(const Session* session) const;
    /**
     * @brief get all output tensors.
     * @param session   given session.
     * @return all output tensors mapped with name.
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

private:
    static Interpreter* createFromBufferInternal(Content* net);

    Content* mNet = nullptr;
    Interpreter(Content* net);

    Interpreter(const Interpreter&)  = delete;
    Interpreter(const Interpreter&&) = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter& operator=(const Interpreter&&) = delete;
};
} // namespace MNN

#endif /* Interpreter_hpp */
