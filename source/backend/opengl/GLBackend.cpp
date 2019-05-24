//
//  GLBackend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <sstream>
#include "AllShader.hpp"
#include "GLConvolution.hpp"
#include "GLConvolutionDepthwise.hpp"
#include "GLSSBOBuffer.hpp"
#include "GLTexture.hpp"
#include "AutoTime.hpp"
#include "GLBackend.hpp"
#include "GLConcat.hpp"
#include "GLEltwise.hpp"
#include "GLPool.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
namespace MNN {
namespace OpenGL {

std::map<OpType, GLBackend::Creator*>& gCreator() {
    static std::once_flag once;
    static std::map<OpType, GLBackend::Creator*>* creators;
    std::call_once(once, []() { creators = new std::map<OpType, GLBackend::Creator*>; });
    return *creators;
};

void GLBackend::addCreator(OpType t, Creator* c) {
    gCreator()[t] = c;
}

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
    mContext = GLContext::create();
    mRuntime                       = new Runtime;
    mRuntime->mDownloadProgram     = getTreatedProgram(glsl_download_glsl);
    mRuntime->mUploadProgram       = getTreatedProgram(glsl_upload_glsl);
    mRuntime->mUploadCopyProgram   = getTreatedProgram(glsl_buffer2Image_glsl);
    mRuntime->mDownloadCopyProgram = getTreatedProgram(glsl_image2Buffer_glsl);
}

GLBackend::~GLBackend() {
    delete mRuntime;
    GLContext::destroy(mContext);
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
        mRuntime->mDownloadCopyProgram->useProgram();
    } else {
        mRuntime->mDownloadProgram->useProgram();
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
        mRuntime->mUploadCopyProgram->useProgram();
    } else {
        mRuntime->mUploadProgram->useProgram();
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
    auto& creators = gCreator();
    auto iter      = creators.find(op->type());
    if (iter == creators.end()) {
        MNN_PRINT("Don't support type %d, %s\n", op->type(), op->name()->c_str());
        return NULL;
    }
    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        MNN_PRINT("The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
        return NULL;
    }
    return exe;
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
}

bool GLBackend::onClearBuffer() {
    mRuntime->mBlocks.clear();
    mRuntime->mFreeTextures.clear();
    return true;
}

bool GLBackend::onReleaseBuffer(const Tensor *nativeTensor, Backend::StorageType storageType) {
    mRuntime->mFreeTextures.push_back(std::make_pair(nativeTensor, nativeTensor->buffer().device));
    return true;
}

bool GLBackend::onAcquireBuffer(const Tensor *nativeTensor, Backend::StorageType storageType) {
    auto tensor = (Tensor *)nativeTensor;
    for (auto iter = mRuntime->mFreeTextures.begin(); iter != mRuntime->mFreeTextures.end(); ++iter) {
        auto preiousTensor = iter->first;
        if (preiousTensor->width() >= nativeTensor->width() && preiousTensor->height() >= nativeTensor->height() &&
            UP_DIV(preiousTensor->channel(), 4) >= UP_DIV(nativeTensor->channel(), 4)) {
            mRuntime->mFreeTextures.erase(iter);
            tensor->buffer().device = iter->second;
            return true;
        }
    }

    std::shared_ptr<GLTexture> newTexture(new GLTexture(nativeTensor->width(), nativeTensor->height(), nativeTensor->channel()));
    tensor->buffer().device = newTexture->id();
    mRuntime->mBlocks.push_back(std::move(newTexture));
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

} // namespace OpenGL
} // namespace MNN
