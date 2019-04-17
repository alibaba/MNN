//
//  Execution.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Execution_hpp
#define Execution_hpp

#include <memory>
#include <string>
#include "ErrorCode.hpp"
#include "MNNForwardType.h"
#include "NonCopyable.hpp"
#include "Tensor.hpp"

namespace MNN {
class Backend;
struct Op;

/** abstract execution */
class Execution : public NonCopyable {
public:
    /**
     * @brief initializer.
     * @param backend   backend that exection will running on.
     */
    Execution() = delete;
    Execution(Backend *backend) : mBackEnd(backend) {
        // nothing to do
    }
    /**
     * @brief deinitializer.
     */
    virtual ~Execution() = default;

    /**
     * @brief response shape change of input or output tensors.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return resize result
     */
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        return NO_ERROR;
    }

    /**
     * @brief if the session will not be resized any more,
        it will call it for execution to release cache used for resize
     * @return release result
     */
    virtual ErrorCode onReleaseCache() {
        return NO_ERROR;
    }

    /**
     * @brief perform execution.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return execution result
     */
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) = 0;

public:
    /**
     * @brief designed for plugin system. not ready yet.
     */
    class Creator : public NonCopyable {
    public:
        /**
         * @brief deinitializer.
         */
        virtual ~Creator() = default;
        /**
         * @brief create execution for given op on given backend.
         * @param backend   given backend.
         * @param op        given op.
         * @return execution.
         */
        virtual Execution *onCreate(Backend *backend, const Op *op) const = 0;
    };

    // Search for extra creator, if not found, return nullptr
    MNN_PUBLIC static const Creator *searchExtraCreator(const std::string &key, MNNForwardType type);

    /**
     * @brief register creator for given key and backend type.
     * @param creator registering creator.
     * @param key given key.
     * @param type given backend type.
     * @return false if registered creator for same key and type exists, true otherwise.
     */
    MNN_PUBLIC static bool insertExtraCreator(std::shared_ptr<Creator> creator, const std::string &key,
                                              MNNForwardType type);

    /**
     * @brief unregister creator for given key and backend type.
     * @param key given key.
     * @param type given backend type.
     * @return true if registered creator for given key and type exists, false otherwise.
     */
    MNN_PUBLIC static bool removeExtraCreator(const std::string &key, MNNForwardType type);

public:
    /**
     * @brief check if execution is valid.
     * @return valid or not.
     */
    inline bool valid() const {
        return mValid;
    }
    /**
     * @brief get backend.
     * @return backend.
     */
    Backend *backend() const {
        return mBackEnd;
    }

protected:
    bool mValid = true;

private:
    Backend *mBackEnd;
};

} // namespace MNN

#endif /* Execution_hpp */
