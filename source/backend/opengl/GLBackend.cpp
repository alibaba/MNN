//
//  GLBackend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <sstream>
#include "AllShader.h"
#include "GLConvolution.h"
#include "GLConvolutionDepthwise.h"
#include "GLSSBOBuffer.h"
#include "GLTexture.h"
#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"
#include "GLBackend.h"
#include "GLConcat.h"
#include "GLEltwise.h"
#include "GLPool.h"
#include "Macro.h"
#include "OpenGLWorker.h"
#include "TensorUtils.hpp"
namespace MNN {
class GLThreadExecution : public Execution {
public:
    GLThreadExecution(Execution *real, Backend *bn) : Execution(bn), mRealExecution(real) {
        MNN_ASSERT(nullptr != real);
        MNN_ASSERT(nullptr != bn);
    }
    ~GLThreadExecution() {
        std::shared_ptr<GLWork> work(new GLFunctionWork([this]() { delete mRealExecution; }));
        auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
        sema->wait();
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        ErrorCode code = NO_ERROR;
        std::shared_ptr<GLWork> work(new GLFunctionWork(
            [this, &code, &inputs, &outputs]() { code = mRealExecution->onResize(inputs, outputs); }));
        auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
        sema->wait();
        mExeWork.reset(new GLFunctionWork([this, &inputs, &outputs]() { mRealExecution->onExecute(inputs, outputs); }));
        return code;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        OpenGLWorker::getInstance()->queueWork(mExeWork, false);
        return NO_ERROR;
    }

private:
    Execution *mRealExecution;
    std::shared_ptr<GLWork> mExeWork;
};

static std::shared_ptr<GLProgram> getTreatedProgramWithPrefix(const char *content,
                                                              const std::vector<std::string> &prefix) {
    std::ostringstream tc;
    tc << GLProgram::getHead();
    for (auto &s : prefix) {
        tc << s << "\n";
    }
    tc << content;
    return std::shared_ptr<GLProgram>(new GLProgram(tc.str()));
}
static std::shared_ptr<GLProgram> getTreatedProgram(const char *content) {
    std::ostringstream tc;
    tc << GLProgram::getHead() << content;
    return std::shared_ptr<GLProgram>(new GLProgram(tc.str()));
}

GLBackend::GLBackend(MNNForwardType type) : Backend(type) {
    // MNN_PRINT("%s, %d", __func__, __LINE__);
    Runtime *runtime = nullptr;
    std::shared_ptr<GLWork> work(new GLFunctionWork([this, &runtime]() {
        runtime                       = new Runtime;
        runtime->mDownloadProgram     = getTreatedProgram(glsl_download_glsl);
        runtime->mUploadProgram       = getTreatedProgram(glsl_upload_glsl);
        runtime->mUploadCopyProgram   = getTreatedProgram(glsl_buffer2Image_glsl);
        runtime->mDownloadCopyProgram = getTreatedProgram(glsl_image2Buffer_glsl);
    }));
    auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
    sema->wait();
    mRuntime = runtime;
}

GLBackend::~GLBackend() {
    std::shared_ptr<GLWork> work(new GLFunctionWork([this]() { delete mRuntime; }));
    auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
    sema->wait();
}

void GLBackend::download(GLuint textureId, float *outputData, int d1, int d2, int d3, bool align) const {
    auto sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    OPENGL_CHECK_ERROR;
    glWaitSync(sync, 0, GL_TIMEOUT_IGNORED);
    OPENGL_CHECK_ERROR;
    glDeleteSync(sync);
    OPENGL_CHECK_ERROR;
    auto depthQuad = UP_DIV(d3, 4);
    auto size      = depthQuad * 4 * d1 * d2 * sizeof(float);
    if (NULL == mRuntime->mTempBuffer.get() || mRuntime->mTempBuffer->size() < size) {
        mRuntime->mTempBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(size));
    }
    auto &buffer = mRuntime->mTempBuffer;
    if (align) {
        mRuntime->mDownloadCopyProgram->use();
    } else {
        mRuntime->mDownloadProgram->use();
    }
    glBindImageTexture(0, textureId, 0, GL_TRUE, 0, GL_READ_ONLY, TEXTURE_FORMAT);
    OPENGL_CHECK_ERROR;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(2, d1);
    glUniform1i(3, d2);
    OPENGL_CHECK_ERROR;

    glDispatchCompute(UP_DIV(d1, 8), UP_DIV(d2, 8), depthQuad);
    OPENGL_CHECK_ERROR;

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    OPENGL_CHECK_ERROR;

    auto gpuoutput = buffer->map(GL_MAP_READ_BIT);
    if (align) {
        ::memcpy(outputData, gpuoutput, size);
    } else {
        ::memcpy(outputData, gpuoutput, d1 * d2 * d3 * sizeof(float));
    }
    buffer->unmap();
}

void GLBackend::upload(GLuint textureId, const float *inputData, int d1, int d2, int d3, bool align) const {
    auto depthQuad = UP_DIV(d3, 4);
    auto size      = depthQuad * 4 * d1 * d2 * sizeof(float);
    if (NULL == mRuntime->mTempBuffer.get() || mRuntime->mTempBuffer->size() < size) {
        mRuntime->mTempBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(size));
    }
    auto &buffer = mRuntime->mTempBuffer;

    auto gpuoutput = buffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if (align) {
        ::memcpy(gpuoutput, inputData, size);
    } else {
        ::memcpy(gpuoutput, inputData, d1 * d2 * d3 * sizeof(float));
    }
    buffer->unmap();
    if (align) {
        mRuntime->mUploadCopyProgram->use();
    } else {
        mRuntime->mUploadProgram->use();
    }
    glBindImageTexture(0, textureId, 0, GL_TRUE, 0, GL_WRITE_ONLY, TEXTURE_FORMAT);
    OPENGL_CHECK_ERROR;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(2, d1);
    glUniform1i(3, d2);
    OPENGL_CHECK_ERROR;

    glDispatchCompute(UP_DIV(d1, 8), UP_DIV(d2, 8), depthQuad);
    OPENGL_CHECK_ERROR;
#ifdef MNN_GPU_FORCE_FINISH
    glFinish();
#endif
}

Execution *GLBackend::onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const MNN::Op *op) {
    // MNN_PRINT("Create op for %s\n", op->name()->c_str());
    Execution *exe = nullptr;
    std::shared_ptr<GLWork> work(new GLFunctionWork([this, &inputs, op, &exe]() {
        switch (op->type()) {
            case OpType_Convolution:
                exe = new GLConvolution(op, this);
                break;
            case OpType_Pooling:
                exe = new GLPool(op->main_as_Pool(), this);
                break;
            case OpType_Eltwise:
                exe = new GLEltwise(op->main_as_Eltwise()->type(), inputs.size(), this);
                break;
            case OpType_Concat: {
                bool valid = true;
                if (op->main_as_Axis()->axis() == 1) {
                    for (int i = 1; i < inputs.size(); ++i) {
                        if (inputs[i]->channel() % 4 != 0) {
                            valid = false;
                            break;
                        }
                    }
                }
                if (valid) {
                    exe = new GLConcat(op->main_as_Axis()->axis(), this);
                }
            } break;
            case OpType_ConvolutionDepthwise:
                exe = new GLConvolutionDepthwise(op, this);
                break;
            case OpType_BinaryOp: {
                auto binary = op->main_as_BinaryOp();
                if (binary->T() == DataType_DT_FLOAT) {
                    switch (binary->opType()) {
                        case BinaryOpOperation_ADD:
                            exe = new GLEltwise(EltwiseType_SUM, inputs.size(), this);
                            break;
                        case BinaryOpOperation_MUL:
                            exe = new GLEltwise(EltwiseType_PROD, inputs.size(), this);
                            break;
                        case BinaryOpOperation_MAX_TEMP:
                            exe = new GLEltwise(EltwiseType_MAXIMUM, inputs.size(), this);
                            break;
                        default:
                            break;
                    }
                }
                break;
            }
            default:
                break;
        }
    }));
    auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
    sema->wait();
    if (nullptr == exe) {
        MNN_PRINT("%s use cpu \n", op->name()->c_str());
        return nullptr;
    }

    return new GLThreadExecution(exe, this);
}

void GLBackend::onExecuteEnd() const {
    // MNN_PRINT("Finish\n");
    // glFinish();
}

void GLBackend::onExecuteBegin() const {
}

void GLBackend::onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const {
    // MNN_PRINT("src: %p, %lld, dst: %p, %lld\n", srcTensor->buffer().host, srcTensor->buffer().device,
    // dstTensor->buffer().host, dstTensor->buffer().device);
    std::shared_ptr<GLWork> work(new GLFunctionWork([this, srcTensor, dstTensor]() {
        if (NULL == srcTensor->buffer().host && srcTensor->buffer().device > 0) {
            // GPU to CPU
            MNN_ASSERT(NULL != dstTensor->buffer().host);
            MNN_ASSERT(srcTensor->buffer().device > 0);
            download((GLuint)srcTensor->buffer().device, (float *)dstTensor->buffer().host, dstTensor->width(),
                     dstTensor->height(), dstTensor->channel(),
                     TensorUtils::getDescribe(dstTensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
            return;
        }
        if (NULL == dstTensor->buffer().host && dstTensor->buffer().device > 0) {
            upload((GLuint)dstTensor->buffer().device, srcTensor->host<float>(), srcTensor->width(),
                   srcTensor->height(), srcTensor->channel(),
                   TensorUtils::getDescribe(srcTensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
            return;
        }
        MNN_ASSERT(false);
    }));
    auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
    sema->wait();
}

bool GLBackend::onClearBuffer() {
    std::shared_ptr<GLWork> work(new GLFunctionWork([this]() {
        mRuntime->mBlocks.clear();
        mRuntime->mFreeTextures.clear();
    }));
    auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
    sema->wait();
    return true;
}

bool GLBackend::onReleaseBuffer(const Tensor *nativeTensor, Backend::StorageType storageType) {
    std::shared_ptr<GLWork> work(new GLFunctionWork([this, nativeTensor, storageType]() {
        mRuntime->mFreeTextures.push_back(std::make_pair(nativeTensor, nativeTensor->buffer().device));
    }));
    auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
    sema->wait();
    return true;
}

bool GLBackend::onAcquireBuffer(const Tensor *nativeTensor, Backend::StorageType storageType) {
    std::shared_ptr<GLWork> work(new GLFunctionWork([this, nativeTensor, storageType]() {
        auto tensor = (Tensor *)nativeTensor;
        for (auto iter = mRuntime->mFreeTextures.begin(); iter != mRuntime->mFreeTextures.end(); ++iter) {
            auto preiousTensor = iter->first;
            if (preiousTensor->width() >= nativeTensor->width() && preiousTensor->height() >= nativeTensor->height() &&
                UP_DIV(preiousTensor->channel(), 4) >= UP_DIV(nativeTensor->channel(), 4)) {
                mRuntime->mFreeTextures.erase(iter);
                tensor->buffer().device = iter->second;
                return;
            }
        }

        std::shared_ptr<GLTexture> newTexture(
            new GLTexture(nativeTensor->width(), nativeTensor->height(), nativeTensor->channel()));
        tensor->buffer().device = newTexture->id();

        mRuntime->mBlocks.push_back(std::move(newTexture));
    }));
    auto sema = OpenGLWorker::getInstance()->queueWork(work, true);
    sema->wait();
    return true;
}

std::shared_ptr<GLProgram> GLBackend::getProgram(const std::string &key, const char *content,
                                                 const std::vector<std::string> &prefix) {
    if (key.empty()) {
        return getTreatedProgramWithPrefix(content, prefix);
    }
    // Generate New Key
    std::ostringstream newKey;
    for (auto s : prefix) {
        newKey << s;
    }
    newKey << key;
    auto newKeyStr = newKey.str();

    auto iter = mRuntime->mProgramCache.find(newKeyStr);
    if (iter != mRuntime->mProgramCache.end()) {
        return iter->second;
    }
    auto program = getTreatedProgramWithPrefix(content, prefix);
    mRuntime->mProgramCache.insert(std::make_pair(newKeyStr, program));

    return program;
}

std::shared_ptr<GLProgram> GLBackend::getProgram(const std::string &key, const char *content) {
    if (key.empty()) {
        return getTreatedProgram(content);
    }
    auto iter = mRuntime->mProgramCache.find(key);
    if (iter != mRuntime->mProgramCache.end()) {
        return iter->second;
    }
    auto program = getTreatedProgram(content);
    mRuntime->mProgramCache.insert(std::make_pair(key, program));

    return program;
}

class GLBackendCreator : public BackendCreator {
public:
    virtual Backend *onCreate(const Backend::Info &info) const override {
        return new GLBackend(MNN_FORWARD_OPENGL);
    }
};

class GLBackendRegistor {
public:
    GLBackendRegistor() {
        MNNInsertExtraBackendCreator(MNN_FORWARD_OPENGL, new GLBackendCreator);
    }

    ~GLBackendRegistor() {
    }
};

static GLBackendRegistor gRegistor;
} // namespace MNN
