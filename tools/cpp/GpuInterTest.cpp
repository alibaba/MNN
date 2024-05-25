//
//  ModuleBasic.cpp
//  MNN
//
//  Created by MNN on 2021/10/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "rapidjson/document.h"
#include "core/MemoryFormater.h"
#include <fstream>
#include <sstream>
#include <numeric>
#include "ExprDebug.hpp"
#define MNN_USE_LIB_WRAPPER
#define MNN_USER_SET_DEVICE
#define MNN_OPENCL_SVM_ENABLE
#include "MNN/MNNSharedContext.h"
using namespace MNN::Express;
using namespace MNN;

#ifdef __ANDROID__
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#include <EGL/egl.h>
class UserGLDeviceBuffer{
public:
    UserGLDeviceBuffer(){
        EGLDisplay mDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        int majorVersion;
        int minorVersion;
        eglInitialize(mDisplay, &majorVersion, &minorVersion);
        EGLint numConfigs;
        static const EGLint configAttribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                                            EGL_RED_SIZE, 8,
                                            EGL_GREEN_SIZE, 8,
                                            EGL_BLUE_SIZE, 8,
                                            EGL_ALPHA_SIZE, 8,
                                            EGL_NONE};

        EGLConfig surfaceConfig;
        eglChooseConfig(mDisplay, configAttribs, &surfaceConfig, 1, &numConfigs);
        static const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        mContext = eglCreateContext(mDisplay, surfaceConfig, NULL, contextAttribs);
        static const EGLint surfaceAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        mSurface = eglCreatePbufferSurface(mDisplay, surfaceConfig, surfaceAttribs);
        eglMakeCurrent(mDisplay, mSurface, mSurface, mContext);
        eglBindAPI(EGL_OPENGL_ES_API);
        int major;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
    }
    ~UserGLDeviceBuffer(){
        if (mDisplay != EGL_NO_DISPLAY) {
            if (mContext != EGL_NO_CONTEXT) {
                    eglDestroyContext(mDisplay, mContext);
                    mContext = EGL_NO_CONTEXT;
                }
                if (mSurface != EGL_NO_SURFACE) {
                    eglDestroySurface(mDisplay, mSurface);
                    mSurface = EGL_NO_SURFACE;
                }
                eglMakeCurrent(mDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
                eglTerminate(mDisplay);
                mDisplay = EGL_NO_DISPLAY;
            }
        eglReleaseThread();
    }
    GLuint CreateTexture(int width, int height, void* data) {
        GLuint textureID;
        glGenTextures(1, &textureID);

        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, data);

        glBindTexture(GL_TEXTURE_2D, 0);

        return textureID;
    }
    void ReleaseTexture(GLuint textureID){
        glDeleteTextures(1, &textureID);
    }
private:
    EGLContext mContext;
    EGLDisplay mDisplay;
    EGLSurface mSurface;
};
#endif
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
class UserCLDeviceBuffer{
public:
    UserCLDeviceBuffer(){
        OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance();
        std::vector<cl::Platform> platforms;
        cl_int res = cl::Platform::get(&platforms, 0);
        cl::Platform::setDefault(platforms[0]);
        std::vector<cl::Device> gpuDevices;
        res = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);
        mFirstGPUDevicePtr = std::make_shared<cl::Device>(gpuDevices[0]);
        mContext = std::shared_ptr<cl::Context>(new cl::Context(std::vector<cl::Device>({*mFirstGPUDevicePtr}), nullptr, nullptr, nullptr, &res));
        mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, 0, &res);
    }
    std::shared_ptr<cl::Context> getContext(){
        return mContext;
    }
    cl::Buffer *createBuffer(size_t size){
        cl_int res;
        return new cl::Buffer(*mContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(float), NULL, &res);
    }
    void copyToBuffer(cl::Buffer *buffer, int size, float* ptr){
        auto gpuptr = mCommandQueuePtr.get()->enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_WRITE, 0, size * sizeof(float));
        memcpy(gpuptr, ptr, size);
        mCommandQueuePtr.get()->enqueueUnmapMemObject(*buffer, gpuptr);
    }
    float *mapDevicePtr(cl::Buffer *buffer, int size){
        auto gpuptr = mCommandQueuePtr.get()->enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_WRITE, 0, size * sizeof(float));
        return (float*) gpuptr;
    }
    void *umapDevicePtr(cl::Buffer *buffer, void* ptr){
        mCommandQueuePtr.get()->enqueueUnmapMemObject(*buffer, ptr);
    }
private:
    std::shared_ptr<::cl::Context> mContext;
    std::shared_ptr<::cl::Device> mFirstGPUDevicePtr;
    std::shared_ptr<::cl::CommandQueue> mCommandQueuePtr;
};

int main(int argc, char *argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./GpuInterTest.out ${test.mnn} ${Dir} [testMode] [forwardType] [numberThread] [precision | memory]\n");
        return 0;
    }
    std::string modelName = argv[1];
    std::string directName = argv[2];
    MNN_PRINT("Test %s from input info: %s\n", modelName.c_str(), directName.c_str());
    std::map<std::string, float> inputInfo;
    std::map<std::string, std::vector<int>> inputShape;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    int repeatNumber = 1;
    bool shapeMutable = true;
    std::vector<VARP> inputs;
    std::vector<VARP> outputs;
    if (inputNames.empty()) {
        rapidjson::Document document;
        std::ostringstream jsonNameOs;
        jsonNameOs << directName << "/input.json";
        std::ifstream fileNames(jsonNameOs.str().c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        if (document.HasMember("inputs")) {
            auto inputsInfo = document["inputs"].GetArray();
            for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
                auto obj = iter->GetObject();
                std::string name = obj["name"].GetString();
                inputNames.emplace_back(name);
                MNN_PRINT("%s\n", name.c_str());
                if (obj.HasMember("value")) {
                    float value = obj["value"].GetFloat();
                    inputInfo.insert(std::make_pair(name, value));
                }
                if (obj.HasMember("shape")) {
                    auto dims = obj["shape"].GetArray();
                    std::vector<int> shapes;
                    for (auto iter = dims.begin(); iter != dims.end(); iter++) {
                        shapes.emplace_back(iter->GetInt());
                    }
                    inputShape.insert(std::make_pair(name, shapes));
                }
            }
        }
        if (document.HasMember("outputs")) {
            auto array = document["outputs"].GetArray();
            for (auto iter = array.begin(); iter !=array.end(); iter++) {
                std::string name = iter->GetString();
                MNN_PRINT("output: %s\n", name.c_str());
                outputNames.emplace_back(name);
            }
        }
        if (document.HasMember("shapeMutable")) {
            shapeMutable = document["shapeMutable"].GetBool();
        }
        if (document.HasMember("repeat")) {
            repeatNumber = document["repeat"].GetInt();
        }
    }
    int testMode = 0;
    //testMode = 0 OpenCL, testMode = 1 OpenGL
    if(argc > 3){
        testMode = atoi(argv[3]);
        MNN_PRINT("Use extra forward type: %d(0:OpenCL 1:OpenGL)\n", testMode);
    }

    auto type = MNN_FORWARD_CPU;
    if (argc > 4) {
        type = (MNNForwardType)atoi(argv[4]);
        MNN_PRINT("Use extra forward type: %d\n", type);
    }

    // Default single thread
    int modeNum = 1;
    if (argc > 5) {
        modeNum = ::atoi(argv[5]);
    }

    int precision = BackendConfig::Precision_Normal;
    int memory = BackendConfig::Memory_Normal;
    if (argc > 6) {
        int mask = atoi(argv[6]);
        precision = mask % 4;
        memory = (mask / 4) % 4;
    }
    const char* cacheFileName = ".tempcache";
    FUNC_PRINT(precision);
    FUNC_PRINT(memory);
    FUNC_PRINT_ALL(cacheFileName, s);
    // create session
    MNN::ScheduleConfig config;
    config.type      = type;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = modeNum;
    // If type not fount, let it failed
    config.backupType = type;
    BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
    backendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(memory);
    config.backendConfig     = &backendConfig;

    MNN::Express::Module::Config mConfig;
    mConfig.shapeMutable = shapeMutable;
    MNNDeviceContext DeviceContext;
    // set shared context for OpenCL, or context and display for OpenGL
#ifdef __ANDROID__
    std::vector<GLuint> GLdeviceInputPtrVec;
    std::vector<GLuint> GLdeviceOutputPtrVec;
    std::shared_ptr<UserGLDeviceBuffer> GLDeviceBuffer;
    if(testMode == 1){
        GLDeviceBuffer = std::shared_ptr<UserGLDeviceBuffer>(new UserGLDeviceBuffer);
        DeviceContext.contextPtr = eglGetCurrentContext();
        DeviceContext.glShared = eglGetCurrentDisplay();
    }
#endif
    std::vector<cl::Buffer*> CLdeviceInputPtrVec;
    std::vector<cl::Buffer*> CLdeviceOutputPtrVec;
    std::shared_ptr<UserCLDeviceBuffer> CLDeviceBuffer;
    if(testMode == 0){
        CLDeviceBuffer = std::shared_ptr<UserCLDeviceBuffer>(new UserCLDeviceBuffer);
        DeviceContext.contextPtr = CLDeviceBuffer.get()->getContext().get();
    }

    backendConfig.sharedContext = &DeviceContext;

    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setCache(cacheFileName);
    std::shared_ptr<Module> net;
    {
        AUTOTIME;
        net.reset(Module::load(inputNames, outputNames, modelName.c_str(), rtmgr, &mConfig));
        if (net == nullptr) {
            MNN_PRINT("Error: can't load module\n");
            return 0;
        }
    }
    auto mInfo = net->getInfo();
#ifdef __ANDROID__
    GLdeviceInputPtrVec.resize(mInfo->inputs.size());
    GLdeviceOutputPtrVec.resize(outputNames.size());
#endif
    CLdeviceInputPtrVec.resize(mInfo->inputs.size());
    CLdeviceOutputPtrVec.resize(outputNames.size());
    if (inputs.empty()) {
        inputs.resize(mInfo->inputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            inputs[i] = _Input(mInfo->inputs[i].dim, mInfo->inputs[i].order, mInfo->inputs[i].type);
        }
        for (int i=0; i<inputs.size(); ++i) {
            auto inputName = inputNames[i];
            // Resize
            auto info = inputs[i]->getInfo();
            int width = info->dim[3], height = info->dim[2], channel = info->dim[1];
            auto shapeIter = inputShape.find(inputName);
            if (shapeIter != inputShape.end()) {
                auto s = shapeIter->second;
                inputs[i] = _Input(s, mInfo->defaultFormat, mInfo->inputs[i].type);
                width = s[3];
                height = s[2];
                channel = s[1];
            }
            // set input device ptr
#ifdef __ANDROID__
            // OpenGL Texture defaultFormat NC4HW4
            if(testMode == 1){
                width = width * ((channel + 3) / 4);
                GLdeviceInputPtrVec[i] = (GLDeviceBuffer.get()->CreateTexture(width,height,nullptr));
                inputs[i]->setDevicePtr((void*)GLdeviceInputPtrVec[i], MNN_FORWARD_OPENGL);
            }
#endif
            if(testMode == 0){
                CLdeviceInputPtrVec[i] = CLDeviceBuffer.get()->createBuffer(info->size);
                inputs[i]->setDevicePtr(CLdeviceInputPtrVec[i], MNN_FORWARD_OPENCL);
            }
        }
    }

    bool modelError = false;
    for (int repeat = 0; repeat < repeatNumber; ++repeat) {
        AUTOTIME;
        auto outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Error in forward\n");
            return 0;
        }
        for (int i=0; i<outputNames.size(); ++i) {
            auto info = inputs[i]->getInfo();
            int width = info->dim[3], height = info->dim[2], channel = info->dim[1];
            // copy output to device ptr
#ifdef __ANDROID__
            if(testMode == 1){
                GLdeviceOutputPtrVec[i] = GLDeviceBuffer.get()->CreateTexture(width,height,nullptr);
                outputs[i]->copyToDevicePtr((void*)GLdeviceOutputPtrVec[i], MNN_FORWARD_OPENGL);
            }
#endif
            if(testMode == 0){
                CLdeviceOutputPtrVec[i] = CLDeviceBuffer.get()->createBuffer(info->size);
                outputs[i]->copyToDevicePtr(CLdeviceOutputPtrVec[i], MNN_FORWARD_OPENCL);
            }
        }

        // Print module's memory
        float memoryInMB = 0.0f;
        rtmgr->getInfo(Interpreter::MEMORY, &memoryInMB);
        FUNC_PRINT_ALL(memoryInMB, f);
    }
#ifdef __ANDROID__
    if(testMode == 1){
        for(int i = 0; i < GLdeviceInputPtrVec.size(); ++i){
            GLDeviceBuffer.get()->ReleaseTexture(GLdeviceInputPtrVec[i]);
        }
        for(int i = 0; i < GLdeviceOutputPtrVec.size(); ++i){
            GLDeviceBuffer.get()->ReleaseTexture(GLdeviceOutputPtrVec[i]);
        }
    }
#endif
    rtmgr->updateCache();
    return 0;
}

