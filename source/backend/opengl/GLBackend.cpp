//
//  GLBackend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <sstream>
#include "AllShader.hpp"
#include "GLSSBOBuffer.hpp"
#include "GLTexture.hpp"
#include <MNN/AutoTime.hpp>
#include "GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/BufferAllocator.hpp"
#include <mutex>
#include <MNN/Tensor.hpp>

namespace MNN {
namespace OpenGL {

std::map<OpType, GLBackend::Creator*>* gCreator() {
    static std::once_flag once;
    static std::map<OpType, GLBackend::Creator*>* creators = nullptr;
    std::call_once(once, [&]() { creators = new std::map<OpType, GLBackend::Creator*>; });
    return creators;
};

bool GLBackend::addCreator(OpType t, Creator* c) {
    auto map = gCreator();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

std::shared_ptr<GLProgram> GLBackend::getTreatedProgramWithPrefix(const char *content, const std::vector<std::string> &prefix) {
    std::ostringstream tc;
    tc << GLProgram::getHead(getImageFormat());
    for (auto &s : prefix) {
        tc << s << "\n";
    }
    tc << content;
    return std::shared_ptr<GLProgram>(new GLProgram(tc.str()));
}

std::shared_ptr<GLProgram> GLBackend::getTreatedProgram(const char *content) {
    std::ostringstream tc;
    tc << GLProgram::getHead(getImageFormat()) << content;
    return std::shared_ptr<GLProgram>(new GLProgram(tc.str()));
}

bool GLBackend::getOpenGLExtensions(const std::string& extStr) {
    const std::string extension_str((const char*)glGetString(GL_EXTENSIONS));
    return extension_str.find(extStr.c_str()) != std::string::npos;
}

bool GLBackend::isSupportHalf() const{
    return mIsSupportHalf;
}

GLenum GLBackend::getTextrueFormat() const{
    return mTextrueFormat;
}

std::string GLBackend::getImageFormat() const{
    return mImageFormat;
}

std::unique_ptr<GLContext> GLBackend::mContext = nullptr;
GLBackend::GLBackend(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power) : Backend(MNN_FORWARD_OPENGL) {
    if (mContext == nullptr) {
        mContext.reset(new GLContext());
        if(mContext != nullptr){
            if(mContext->isCreateError()){
                MNN_PRINT("mContext error !!! \n");
                mIsCreateError = true;
            }
        }else{
            MNN_PRINT("mContext == nullptr !!! \n");
            mIsCreateError = true;
        }
    }
    mIsSupportHalf = getOpenGLExtensions("GL_EXT_color_buffer_half_float");
    if(mIsSupportHalf && precision != BackendConfig::Precision_High) {
        mTextrueFormat = GL_RGBA16F;
        mImageFormat = "rgba16f";
    }else{
        MNN_PRINT("not support half \n");
        mTextrueFormat = GL_RGBA32F;
        mImageFormat = "rgba32f";
    }
    mRuntime                       = new Runtime;
    mRuntime->mImage2NchwProgram     = getTreatedProgram(glsl_image_to_nchw_buffer_glsl);
    mRuntime->mNchw2ImageProgram       = getTreatedProgram(glsl_nchw_buffer_to_image_glsl);
    mRuntime->mNc4hw42ImageProgram   = getTreatedProgram(glsl_nc4hw4_buffer_to_image_glsl);
    mRuntime->mImage2Nc4hw4Program = getTreatedProgram(glsl_image_to_nc4hw4_buffer_glsl);

    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    mRuntime->mNhwc2ImageProgram   = getProgram("nhwc_buffer_to_image", glsl_nhwc_buffer_to_image_glsl, prefix);
    mRuntime->mImage2NhwcProgram = getProgram("image_to_nhwc_buffer", glsl_image_to_nhwc_buffer_glsl, prefix);

    const GLubyte* renderer = glGetString(GL_RENDERER);
    if(renderer != nullptr){
        MNN_PRINT("gpu type : %s \n", (char*)renderer);
        if(strstr((char *) renderer, "Adreno")){
            mGpuType = ADRENO;
        }else if(strstr((char *) renderer, "Mali")){
            mGpuType = MALI;
        }else{
            mGpuType = OTHER;
        }
    }

    const GLubyte* version = glGetString(GL_VERSION);
    if(version != nullptr){
        MNN_PRINT("gl version : %s \n", version);
        char* p = strstr((char *) version, "V@");
        if(p != nullptr){
            p += strlen("V@");
            char* v = strtok(p, ".");
            if(v != nullptr){
                mVersion = atoi(v);
            }
        }
    }
}

GLBackend::~GLBackend() {
    if(mRuntime != nullptr){
        delete mRuntime;
    }
    if(mContext != nullptr){
        mContext.reset(nullptr);
    }
}

void GLBackend::copyImageToNhwcBuffer(GLuint textureId, float *outputData, int width, int height, int channel) const {
    width = std::max(1, width);
    height = std::max(1, height);
    channel = std::max(1, channel);

    wait();
    auto depthQuad = UP_DIV(channel, 4);
    auto size      = depthQuad * 4 * width * height * sizeof(float);

    auto buffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(size));

    mRuntime->mImage2NhwcProgram->useProgram();

    glBindImageTexture(0, textureId, 0, GL_TRUE, 0, GL_READ_ONLY, getTextrueFormat());
    OPENGL_CHECK_ERROR;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(2, width);
    glUniform1i(3, height);
    glUniform1i(4, channel);
    OPENGL_CHECK_ERROR;
    compute(UP_DIV(width, mLocalSize[0]), UP_DIV(height, mLocalSize[1]), UP_DIV(depthQuad, mLocalSize[2]));
    OPENGL_CHECK_ERROR;

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    OPENGL_CHECK_ERROR;

    auto gpuoutput = buffer->map(GL_MAP_READ_BIT);
    if(gpuoutput != nullptr){
        ::memcpy(outputData, gpuoutput, height * width * channel * sizeof(float));
    }
    buffer->unmap();
}

void GLBackend::copyNhwcBufferToImage(GLuint textureId, const float *inputData, int width, int height, int channel) const {

    int c_4 = UP_DIV(channel, 4);
    auto size      = ROUND_UP(channel, 4) * width * height * sizeof(float);
    auto buffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(size));

    auto gpuoutput = buffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if(gpuoutput != nullptr){
        ::memcpy(gpuoutput, inputData, channel*height*width * sizeof(float));
    }
    buffer->unmap();

    mRuntime->mNhwc2ImageProgram->useProgram();

    glBindImageTexture(0, textureId, 0, GL_TRUE, 0, GL_WRITE_ONLY, getTextrueFormat());
    OPENGL_CHECK_ERROR;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(2, width);
    glUniform1i(3, height);
    glUniform1i(4, channel);
    OPENGL_CHECK_ERROR;
    compute(UP_DIV(width, mLocalSize[0]), UP_DIV(height, mLocalSize[1]), UP_DIV(c_4, mLocalSize[2]));
    OPENGL_CHECK_ERROR;

}

    void GLBackend::wait() const {

#ifdef USE_GL_FINISH
        glFinish();
#else
        glFlush();
#endif

        }

void GLBackend::compute(int dim1, int dim2, int dim3, bool needWait) const {
    wait();
    glDispatchCompute(dim1, dim2, dim3);
}

void GLBackend::download(GLuint textureId, float *outputData, int d1, int d2, int d3, bool align) const {
    wait();
    auto depthQuad = UP_DIV(d3, 4);
    auto size      = depthQuad * 4 * d1 * d2 * sizeof(float);
    if (NULL == mRuntime->mTempBuffer.get() || mRuntime->mTempBuffer->size() < size) {
        mRuntime->mTempBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(size));
    }
    auto &buffer = mRuntime->mTempBuffer;
    if (align) {
        mRuntime->mImage2Nc4hw4Program->useProgram();
    } else {
        mRuntime->mImage2NchwProgram->useProgram();
    }
    glBindImageTexture(0, textureId, 0, GL_TRUE, 0, GL_READ_ONLY, getTextrueFormat());
    OPENGL_CHECK_ERROR;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(2, d1);
    glUniform1i(3, d2);
    OPENGL_CHECK_ERROR;

    compute(UP_DIV(d1, 8), UP_DIV(d2, 8), depthQuad);
    OPENGL_CHECK_ERROR;

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    OPENGL_CHECK_ERROR;

    auto gpuoutput = buffer->map(GL_MAP_READ_BIT);
    if(gpuoutput != nullptr){
        if (align) {
            ::memcpy(outputData, gpuoutput, size);
        } else {
            ::memcpy(outputData, gpuoutput, d1 * d2 * d3 * sizeof(float));
        }
    }
    buffer->unmap();
}

void GLBackend::upload(GLuint textureId, const float *inputData, int width, int height, int channel, bool align) const {
    int c_4 = UP_DIV(channel, 4);
    auto size      = ROUND_UP(channel, 4) * width * height * sizeof(float);
    if (NULL == mRuntime->mTempBuffer.get() || mRuntime->mTempBuffer->size() < size) {
        mRuntime->mTempBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(size));
    }
    auto &buffer = mRuntime->mTempBuffer;

    auto gpuoutput = buffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if(gpuoutput != nullptr){
        if (align) {
            ::memcpy(gpuoutput, inputData, size);
        } else {
            ::memcpy(gpuoutput, inputData, channel*height*width * sizeof(float));
        }
    }

    buffer->unmap();
    if (align) {
        mRuntime->mNc4hw42ImageProgram->useProgram();
    } else {
        mRuntime->mNchw2ImageProgram->useProgram();
    }
    glBindImageTexture(0, textureId, 0, GL_TRUE, 0, GL_WRITE_ONLY, getTextrueFormat());
    OPENGL_CHECK_ERROR;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(2, width);
    glUniform1i(3, height);
    OPENGL_CHECK_ERROR;

    compute(UP_DIV(width, 8), UP_DIV(height, 8), c_4);
    OPENGL_CHECK_ERROR;
}

Execution *GLBackend::onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const MNN::Op *op) {
    auto map  = gCreator();
    auto iter = map->find(op->type());
    if (iter == map->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type %d, %s\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type %d\n", op->type());
        }
        return nullptr;
    }
    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (nullptr == exe) {
        if (nullptr != op->name()) {
            MNN_PRINT("The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("The Creator Don't support type %d\n", op->type());
        }
        return nullptr;
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

    std::vector<int> inputShape  = tensorShapeFormat(srcTensor);
    int ib = inputShape.at(0);
    int ih = inputShape.at(1);
    int iw = inputShape.at(2);
    int ic = inputShape.at(3);

    // OpenGL -> Host
    if (NULL == srcTensor->buffer().host && srcTensor->buffer().device > 0) {
        if(TensorUtils::getDescribe(dstTensor)->dimensionFormat == MNN_DATA_FORMAT_NHWC){
            copyImageToNhwcBuffer((GLuint)srcTensor->deviceId(), dstTensor->host<float>(), iw, ih, ic);
        }else{
            download((GLuint)srcTensor->deviceId(), dstTensor->host<float>(), iw, ih, ic,
                     TensorUtils::getDescribe(dstTensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
        }

    // Host -> OpenGL
    }else if (NULL == dstTensor->buffer().host && dstTensor->buffer().device > 0) {
        if(TensorUtils::getDescribe(srcTensor)->dimensionFormat == MNN_DATA_FORMAT_NHWC){
            copyNhwcBufferToImage((GLuint)dstTensor->deviceId(), srcTensor->host<float>(), iw, ih, ic);
        }else{
            upload((GLuint)dstTensor->deviceId(), srcTensor->host<float>(), iw, ih, ic,
                   TensorUtils::getDescribe(srcTensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
        }
    }else{
        MNN_ASSERT(false);
    }

}

bool GLBackend::onClearBuffer() {
    mRuntime->mBlocks.clear();
    mRuntime->mFreeTextures.clear();
    return true;
}

class GLMemObj : public Backend::MemObj {
public:
    GLMemObj(const Tensor *nativeTensor, uint64_t device, GLBackend::Runtime* runtime) {
        mTensor = nativeTensor;
        mDevice = device;
        mRuntime = runtime;
    }
    virtual ~ GLMemObj() {
        mRuntime->mFreeTextures.push_back(std::make_pair(mTensor, mDevice));
    }
private:
    const Tensor* mTensor;
    uint64_t mDevice;
    GLBackend::Runtime* mRuntime;
};
Backend::MemObj* GLBackend::onAcquire(const Tensor *nativeTensor, Backend::StorageType storageType) {
    auto tensor = (Tensor *)nativeTensor;

    // reuse only for dynamic storage
    if (Backend::DYNAMIC == storageType) {
        for (auto iter = mRuntime->mFreeTextures.begin(); iter != mRuntime->mFreeTextures.end(); ++iter) {
            auto preiousTensor = iter->first;
            if (preiousTensor->width() >= nativeTensor->width() && preiousTensor->height() >= nativeTensor->height() &&
                UP_DIV(preiousTensor->channel(), 4) >= UP_DIV(nativeTensor->channel(), 4)) {
                tensor->buffer().device = iter->second;
                mRuntime->mFreeTextures.erase(iter);
                return new GLMemObj(nativeTensor, tensor->buffer().device, mRuntime);
            }
        }
    }

    std::shared_ptr<GLTexture> newTexture(new GLTexture(nativeTensor->width(), nativeTensor->height(), nativeTensor->channel(), getTextrueFormat()));
    tensor->buffer().device = newTexture->id();
    mRuntime->mBlocks.push_back(std::move(newTexture));
    if (Backend::DYNAMIC == storageType) {
        return new GLMemObj(nativeTensor, tensor->buffer().device, mRuntime);
    }
    return new Backend::MemObj;
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
bool GLBackend::isCreateError() const {
    return mIsCreateError;
}


Backend* GLRuntime::onCreate(const BackendConfig* config) const {
    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
    if (nullptr != mInfo.user) {
        precision = mInfo.user->precision;
        power     = mInfo.user->power;
    }
    auto backend = new GLBackend(precision, power);
    return backend;
}

int GLRuntime::onGetRuntimeStatus(RuntimeStatus statusEnum) const {
    MNN_ERROR("in GLRuntime\n");
    switch (statusEnum) {
        case STATUS_SUPPORT_FP16: {
            return GLBackend::getOpenGLExtensions("GL_EXT_color_buffer_half_float");
            break;
        }
        case STATUS_SUPPORT_DOT_PRODUCT: {
            return 0;
            break;
        }
        default: {
            MNN_ERROR("unsupported interface");
            break;
        }
    }
    return 0;
}

Runtime::CompilerType GLRuntime::onGetCompilerType() const {
    return Compiler_Origin;
}

class GLRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime *onCreate(const Backend::Info &info) const override {
        auto rt = new GLRuntime(info);
        auto bn = (GLBackend*)(rt->onCreate(nullptr));
        if (bn->isCreateError()) {
            delete bn;
            delete rt;
            return nullptr;
        }
        delete bn;
        return rt;
    }
};

bool placeholder = []() {
    static std::once_flag createOnce;
    std::call_once(createOnce, []() {
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_OPENGL, new GLRuntimeCreator, false);
    });
    return true;
}();

} // namespace OpenGL
} // namespace MNN
