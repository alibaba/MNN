//
//  GLBackend.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLBACKEND_H
#define GLBACKEND_H

#include <list>
#include <map>
#include <memory>
#include "Backend.hpp"
#include "GLContext.hpp"
#include "GLProgram.hpp"
#include "GLSSBOBuffer.hpp"
#include "GLTexture.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace OpenGL {
class GLBackend : public Backend {
public:
    GLBackend(MNNForwardType type);
    virtual ~GLBackend();

    void upload(GLuint textureId, const float* inputData, int d1, int d2, int d3, bool align = false) const;
    void download(GLuint textureId, float* outputData, int d1, int d2, int d3, bool align = false) const;

    std::shared_ptr<GLProgram> getProgram(const std::string& key, const char* content);
    std::shared_ptr<GLProgram> getProgram(const std::string& key, const char* content,
                                          const std::vector<std::string>& prefix);

    /*For Buffer alloc and release*/
    virtual bool onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) override;

    // If STATIC, delete the buffer. If dynamic don't free the buffer, just set it to reused
    virtual bool onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) override;

    // Clear All Dynamic Buffer
    virtual bool onClearBuffer() override;

    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    virtual void onExecuteBegin() const override;

    virtual void onExecuteEnd() const override;

    /// get execution
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;

    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &output, const MNN::Op *op, Backend *backend) const = 0;
    };
    static void addCreator(OpType t, Creator *c);

private:
    struct Runtime {
        GLContext::nativeContext* mContext;
        std::shared_ptr<GLProgram> mUploadProgram;
        std::shared_ptr<GLProgram> mDownloadProgram;
        std::shared_ptr<GLProgram> mUploadCopyProgram;
        std::shared_ptr<GLProgram> mDownloadCopyProgram;

        std::map<std::string, std::shared_ptr<GLProgram>> mProgramCache;

        std::list<std::shared_ptr<GLTexture>> mBlocks;
        std::list<std::pair<const Tensor*, GLuint>> mFreeTextures;

        mutable std::shared_ptr<GLSSBOBuffer> mTempBuffer;
    };
    Runtime* mRuntime;
    GLContext::nativeContext* mContext;
};

template <class T>
class GLCreatorRegister {
public:
    GLCreatorRegister(OpType type) {
        T *t = new T;
        GLBackend::addCreator(type, t);
    }
    ~GLCreatorRegister() = default;
};

template <typename T>
class TypedCreator : public GLBackend::Creator {
public:
    virtual ~TypedCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                Backend *backend) const override {
        return new T(inputs, op, backend);
    }
};

} // namespace OpenGL
} // namespace MNN
#endif
