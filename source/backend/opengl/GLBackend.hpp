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
#include "core/Backend.hpp"
#include "backend/opengl/GLContext.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLSSBOBuffer.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
#include "backend/opengl/GLUtils.hpp"
#include "core/TensorUtils.hpp"
#include "backend/opengl/GLHead.hpp"

namespace MNN {
namespace OpenGL {
    /** Each backend belong to a runtime*/
class GLRuntime : public Runtime {
public:
    GLRuntime(const Backend::Info& info) : mInfo(info) {
        // Do nothing
    }
    virtual ~GLRuntime() = default;
    /**
     @brief create backend
     @return created backend
     */
    virtual Backend* onCreate(const BackendConfig* config) const override;

    /**
     @brief clear unuseful resource
     @param level clear level: 0 - 100, bigger mean clear more, smaller mean cache more
     */
    virtual void onGabageCollect(int level) override {
        // Do nothing
    }
    virtual CompilerType onGetCompilerType() const override;
    /**
     @brief Measure the memory it used in MB
     */
    virtual float onGetMemoryInMB() override {
        return 0.0f;
    }

    int onGetRuntimeStatus(RuntimeStatus statusEnum) const override;

private:
    Backend::Info mInfo;
};

class GLBackend : public Backend {
public:
    GLBackend(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power);
    virtual ~GLBackend();

    void upload(GLuint textureId, const float* inputData, int d1, int d2, int d3, bool align = false) const;
    void download(GLuint textureId, float* outputData, int d1, int d2, int d3, bool align = false) const;

    void copyImageToNhwcBuffer(GLuint textureId, float *outputData, int width, int height, int channel) const;
    void copyNhwcBufferToImage(GLuint textureId, const float *inputData, int width, int height, int channel) const;

    std::shared_ptr<GLProgram> getProgram(const std::string& key, const char* content);
    std::shared_ptr<GLProgram> getProgram(const std::string& key, const char* content,
                                          const std::vector<std::string>& prefix);

    enum GPUType { ADRENO = 0, MALI = 1, OTHER = 2 };

    inline GPUType gpuType() const {
        return mGpuType;
    }

    inline int glVersion() const {
        return mVersion;
    }

    void wait() const;

    void compute(int dim1, int dim2, int dim3, bool needWait = false) const;

    /*For Buffer alloc and release*/
    virtual Backend::MemObj* onAcquire(const Tensor* nativeTensor, StorageType storageType) override;

    // Clear All Dynamic Buffer
    virtual bool onClearBuffer() override;

    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    virtual void onExecuteBegin() const override;

    virtual void onExecuteEnd() const override;

    virtual ErrorCode onResizeEnd() override { return NO_ERROR; }

    /// get execution
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;

    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &output, const MNN::Op *op, Backend *backend) const = 0;
    };
    static bool addCreator(OpType t, Creator *c);
    bool isCreateError() const;
    bool isSupportHalf() const;
    static bool getOpenGLExtensions(const std::string& extStr);
    GLenum getTextrueFormat() const;
    std::string getImageFormat() const;
    std::shared_ptr<GLProgram> getTreatedProgramWithPrefix(const char *content,
                                                              const std::vector<std::string> &prefix);
    std::shared_ptr<GLProgram> getTreatedProgram(const char *content);
    struct Runtime {
        std::shared_ptr<GLProgram> mNchw2ImageProgram;
        std::shared_ptr<GLProgram> mImage2NchwProgram;
        std::shared_ptr<GLProgram> mNc4hw42ImageProgram;
        std::shared_ptr<GLProgram> mImage2Nc4hw4Program;

        std::shared_ptr<GLProgram> mNhwc2ImageProgram;
        std::shared_ptr<GLProgram> mImage2NhwcProgram;

        std::map<std::string, std::shared_ptr<GLProgram>> mProgramCache;

        std::list<std::shared_ptr<GLTexture>> mBlocks;
        std::list<std::pair<const Tensor*, GLuint>> mFreeTextures;
        mutable std::shared_ptr<GLSSBOBuffer> mTempBuffer;
    };
private:
    Runtime* mRuntime = nullptr;
    static std::unique_ptr<GLContext> mContext;
    GPUType mGpuType = OTHER;
    int mVersion = 0;
    int mLocalSize[3];
    bool mIsCreateError{false};
    bool mIsSupportHalf{false};
    GLenum mTextrueFormat{GL_RGBA32F};
    std::string mImageFormat{"rgba32f"};
};

inline std::vector<int> tensorShapeFormat(const Tensor *input) {
    int iN = std::max(1, input->batch());
    int iC = std::max(1, input->channel());
    int iH = std::max(1, input->height());
    int iW = std::max(1, input->width());

    if (input->dimensions() == 3) {
        iN = 1;
        iH = input->buffer().dim[0].extent;
        iW = input->buffer().dim[1].extent;
        iC = input->buffer().dim[2].extent;
    }

    if (input->dimensions() == 2) {
        iN = input->buffer().dim[0].extent;
        iH = 1;
        iW = 1;
        iC = input->buffer().dim[1].extent;
    }
    if (input->dimensions() == 1) {
        iN = 1;
        iH = 1;
        iW = 1;
        iC = input->buffer().dim[0].extent;
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("dim %d : [%d, %d, %d, %d] \n",input->dimensions(), iN, iH, iW, iC);
#endif
    std::vector<int> shape_vec{iN, iH, iW, iC};

    return shape_vec;
}

template <class T>
class GLCreatorRegister {
public:
    GLCreatorRegister(OpType type) {
        GLBackend::addCreator(type, new T);
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
