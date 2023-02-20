//
//  mnnnetnative.cpp
//  MNN
//
//  Created by MNN on 2019/01/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <android/bitmap.h>
#include <jni.h>
#include <string.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>

extern "C" JNIEXPORT jlong JNICALL
Java_com_taobao_android_mnn_MNNNetNative_nativeCreateNetFromFile(JNIEnv *env, jclass type, jstring modelName_) {
    const char *modelName = env->GetStringUTFChars(modelName_, 0);
    auto interpreter      = MNN::Interpreter::createFromFile(modelName);
    env->ReleaseStringUTFChars(modelName_, modelName);

    return (jlong)interpreter;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_taobao_android_mnn_MNNNetNative_nativeCreateNetFromBuffer(JNIEnv *env, jclass type, jbyteArray jbuffer) {
    if (nullptr == jbuffer) {
        return 0;
    }

    auto length = env->GetArrayLength(jbuffer);
    auto destBuffer = env->GetByteArrayElements(jbuffer, nullptr);
    auto interpreter      = MNN::Interpreter::createFromBuffer(destBuffer, length);
    env->ReleaseByteArrayElements(jbuffer, destBuffer, 0);

    return (jlong)interpreter;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeReleaseNet(JNIEnv *env, jclass type,
                                                                                             jlong netPtr) {
    if (0 == netPtr) {
        return 0;
    }
    delete ((MNN::Interpreter *)netPtr);
    return 0;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeCreateSession(
    JNIEnv *env, jclass type, jlong netPtr, jint forwardType, jint numThread, jobjectArray jsaveTensors,
    jobjectArray joutputTensors) {
    MNN::ScheduleConfig config;
    config.type = (MNNForwardType)forwardType;
    if (numThread > 0) {
        config.numThread = numThread;
    }

    if (jsaveTensors != NULL) {
        int size = env->GetArrayLength(jsaveTensors);
        std::vector<std::string> saveNamesVector;

        for (int i = 0; i < size; i++) {
            jstring jname       = (jstring)env->GetObjectArrayElement(jsaveTensors, i);
            const char *name    = env->GetStringUTFChars(jname, NULL);
            std::string nameStr = name;
            saveNamesVector.push_back(nameStr);

            env->ReleaseStringUTFChars(jname, name);
        }
        config.saveTensors = saveNamesVector;
    }

    if (joutputTensors != NULL) {
        int size = env->GetArrayLength(joutputTensors);
        std::vector<std::string> saveNamesVector;

        for (int i = 0; i < size; i++) {
            jstring jname       = (jstring)env->GetObjectArrayElement(joutputTensors, i);
            const char *name    = env->GetStringUTFChars(jname, NULL);
            std::string nameStr = name;
            saveNamesVector.push_back(nameStr);

            env->ReleaseStringUTFChars(jname, name);
        }

        config.path.outputs = saveNamesVector;
    }

    auto session = ((MNN::Interpreter *)netPtr)->createSession(config);
    return (jlong)session;
}

extern "C" JNIEXPORT void JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeReleaseSession(JNIEnv *env,
                                                                                                jclass type,
                                                                                                jlong netPtr,
                                                                                                jlong sessionPtr) {
    auto net     = (MNN::Interpreter *)netPtr;
    auto session = (MNN::Session *)sessionPtr;
    net->releaseSession(session);
}

extern "C" JNIEXPORT jint JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeRunSession(JNIEnv *env, jclass type,
                                                                                            jlong netPtr,
                                                                                            jlong sessionPtr) {
    auto net     = (MNN::Interpreter *)netPtr;
    auto session = (MNN::Session *)sessionPtr;
    return net->runSession(session);
}

extern "C" JNIEXPORT jint JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeRunSessionWithCallback(
    JNIEnv *env, jclass type, jlong netPtr, jlong sessionPtr, jobjectArray nameArray, jlongArray jtensoraddrs) {
    int nameSize   = env->GetArrayLength(nameArray);
    int tensorSize = env->GetArrayLength(jtensoraddrs);
    if (tensorSize < nameSize) {
        MNN_ERROR("tensor array not enough!");
    }

    jlong *tensoraddrs = (jlong *)env->GetLongArrayElements(jtensoraddrs, nullptr);

    std::vector<std::string> nameVector;

    for (int i = 0; i < nameSize; i++) {
        jstring jname       = (jstring)env->GetObjectArrayElement(nameArray, i);
        const char *name    = env->GetStringUTFChars(jname, NULL);
        std::string nameStr = name;
        nameVector.push_back(nameStr);

        env->ReleaseStringUTFChars(jname, name);
        env->DeleteLocalRef(jname);
    }

    MNN::TensorCallBack beforeCallBack = [&](const std::vector<MNN::Tensor *> &ntensors, const std::string &opName) {
        return true;
    };

    MNN::TensorCallBack AfterCallBack = [&](const std::vector<MNN::Tensor *> &ntensors, const std::string &opName) {
        for (int i = 0; i < nameVector.size(); i++) {
            if (nameVector.at(i) == opName) {
                auto ntensor = ntensors[0];

                auto outputTensorUser = new MNN::Tensor(ntensor, MNN::Tensor::TENSORFLOW);
                ntensor->copyToHostTensor(outputTensorUser);
                tensoraddrs[i] = (long)outputTensorUser;
            }
        }
        return true;
    };

    auto net     = (MNN::Interpreter *)netPtr;
    auto session = (MNN::Session *)sessionPtr;

    net->runSessionWithCallBack(session, beforeCallBack, AfterCallBack, true);

    env->SetLongArrayRegion(jtensoraddrs, 0, tensorSize, tensoraddrs);

    env->ReleaseLongArrayElements(jtensoraddrs, tensoraddrs, 0);

    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeReshapeSession(JNIEnv *env,
                                                                                                jclass type,
                                                                                                jlong netPtr,
                                                                                                jlong sessionPtr) {
    auto net     = (MNN::Interpreter *)netPtr;
    auto session = (MNN::Session *)sessionPtr;
    net->resizeSession(session);
    return 0;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeGetSessionInput(
    JNIEnv *env, jclass type, jlong netPtr, jlong sessionPtr, jstring name_) {
    auto net     = (MNN::Interpreter *)netPtr;
    auto session = (MNN::Session *)sessionPtr;
    if (nullptr == name_) {
        return (jlong)net->getSessionInput(session, nullptr);
    }

    const char *name = env->GetStringUTFChars(name_, 0);
    auto tensor      = net->getSessionInput(session, name);

    env->ReleaseStringUTFChars(name_, name);
    return (jlong)tensor;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeGetSessionOutput(
    JNIEnv *env, jclass type, jlong netPtr, jlong sessionPtr, jstring name_) {
    auto net     = (MNN::Interpreter *)netPtr;
    auto session = (MNN::Session *)sessionPtr;
    if (nullptr == name_) {
        return (jlong)net->getSessionOutput(session, nullptr);
    }
    const char *name = env->GetStringUTFChars(name_, 0);
    auto tensor      = net->getSessionOutput(session, name);
    env->ReleaseStringUTFChars(name_, name);
    return (jlong)tensor;
}

extern "C" JNIEXPORT void JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeReshapeTensor(JNIEnv *env, jclass type,
                                                                                               jlong netPtr,
                                                                                               jlong tensorPtr,
                                                                                               jintArray dims_) {
    jint *dims   = env->GetIntArrayElements(dims_, NULL);
    auto dimSize = env->GetArrayLength(dims_);
    std::vector<int> dimVector(dimSize);
    for (int i = 0; i < dimSize; ++i) {
        dimVector[i] = dims[i];
    }
    auto net    = (MNN::Interpreter *)netPtr;
    auto tensor = (MNN::Tensor *)tensorPtr;
    net->resizeTensor(tensor, dimVector);
    env->ReleaseIntArrayElements(dims_, dims, 0);
}

extern "C" JNIEXPORT void JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeSetInputIntData(
    JNIEnv *env, jclass type, jlong netPtr, jlong tensorPtr, jintArray data_) {
    auto tensor = (MNN::Tensor *)tensorPtr;

    jint *data    = env->GetIntArrayElements(data_, NULL);
    auto dataSize = env->GetArrayLength(data_);

    for (int i = 0; i < dataSize; ++i) {
        tensor->host<int>()[i] = data[i];
    }

    env->ReleaseIntArrayElements(data_, data, 0);
}

extern "C" JNIEXPORT void JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeSetInputFloatData(
    JNIEnv *env, jclass type, jlong netPtr, jlong tensorPtr, jfloatArray data_) {
    auto tensor = (MNN::Tensor *)tensorPtr;

    jfloat *data  = env->GetFloatArrayElements(data_, NULL);
    auto dataSize = env->GetArrayLength(data_);

    for (int i = 0; i < dataSize; ++i) {
        tensor->host<float>()[i] = data[i];
    }

    env->ReleaseFloatArrayElements(data_, data, 0);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_taobao_android_mnn_MNNNetNative_nativeTensorGetDimensions(JNIEnv *env, jclass type, jlong tensorPtr) {
    auto tensor     = (MNN::Tensor *)tensorPtr;
    auto dimensions = tensor->buffer().dimensions;

    jintArray result = env->NewIntArray(dimensions);

    jint *destDims = env->GetIntArrayElements(result, NULL);
    for (int i = 0; i < dimensions; ++i) {
        destDims[i] = tensor->length(i);
    }
    env->ReleaseIntArrayElements(result, destDims, 0);
    return result;
}

extern "C" JNIEXPORT jint JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeTensorGetUINT8Data(JNIEnv *env,
                                                                                                    jclass type,
                                                                                                    jlong tensorPtr,
                                                                                                    jbyteArray jdest) {
    auto tensor = (MNN::Tensor *)tensorPtr;
    if (nullptr == jdest) {
        return tensor->elementSize();
    }

    auto length = env->GetArrayLength(jdest);
    std::unique_ptr<MNN::Tensor> hostTensor;
    if (tensor->host<int>() == nullptr) {
        // GPU buffer
        hostTensor.reset(new MNN::Tensor(tensor, tensor->getDimensionType(), true));
        tensor->copyToHostTensor(hostTensor.get());
        tensor = hostTensor.get();
    }

    auto size = tensor->elementSize();
    if (length < size) {
        MNN_ERROR("Can't copy buffer, length no enough");
        return JNI_FALSE;
    }

    auto destPtr = env->GetByteArrayElements(jdest, nullptr);
    ::memcpy(destPtr, tensor->host<uint8_t>(), size * sizeof(uint8_t));
    env->ReleaseByteArrayElements(jdest, destPtr, 0);

    return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeTensorGetIntData(JNIEnv *env,
                                                                                                  jclass type,
                                                                                                  jlong tensorPtr,
                                                                                                  jintArray dest) {
    auto tensor = (MNN::Tensor *)tensorPtr;
    if (nullptr == dest) {
        return tensor->elementSize();
    }

    std::unique_ptr<MNN::Tensor> hostTensor;
    auto length = env->GetArrayLength(dest);
    if (tensor->host<int>() == nullptr) {
        // GPU buffer
        hostTensor.reset(new MNN::Tensor(tensor, tensor->getDimensionType(), true));
        tensor->copyToHostTensor(hostTensor.get());
        tensor = hostTensor.get();
    }

    auto size = tensor->elementSize();
    if (length < size) {
        MNN_ERROR("Can't copy buffer, length no enough");
        return JNI_FALSE;
    }

    auto destPtr = env->GetIntArrayElements(dest, nullptr);
    ::memcpy(destPtr, tensor->host<int>(), size * sizeof(int));
    env->ReleaseIntArrayElements(dest, destPtr, 0);

    return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeTensorGetData(JNIEnv *env, jclass type,
                                                                                               jlong tensorPtr,
                                                                                               jfloatArray dest) {
    auto tensor = reinterpret_cast<MNN::Tensor *>(tensorPtr);
    if (nullptr == dest) {
        std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), false));
        return hostTensor->elementSize();
    }
    auto length = env->GetArrayLength(dest);
    std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), true));
    tensor->copyToHostTensor(hostTensor.get());
    tensor = hostTensor.get();

    auto size = tensor->elementSize();
    if (length < size) {
        MNN_ERROR("Can't copy buffer, length no enough");
        return JNI_FALSE;
    }
    auto destPtr = env->GetFloatArrayElements(dest, nullptr);
    ::memcpy(destPtr, tensor->host<float>(), size * sizeof(float));
    env->ReleaseFloatArrayElements(dest, destPtr, 0);

    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeConvertBufferToTensor(
    JNIEnv *env, jclass type, jbyteArray jbufferData, jint jwidth, jint jheight, jlong tensorPtr, jint srcType,
    jint destFormat, jint filterType, jint wrap, jfloatArray matrixValue_, jfloatArray mean_, jfloatArray normal_) {
    jbyte *bufferData = env->GetByteArrayElements(jbufferData, NULL);
    if (bufferData == NULL) {
        MNN_ERROR("Error Buffer Null!\n");
        return JNI_FALSE;
    }

    {
        auto size = env->GetArrayLength(matrixValue_);
        if (size < 9) {
            env->ReleaseByteArrayElements(jbufferData, bufferData, 0);
            MNN_ERROR("Error matrix length:%d\n", size);
            return JNI_FALSE;
        }
    }

    MNN::CV::ImageProcess::Config config;
    config.destFormat   = (MNN::CV::ImageFormat)destFormat;
    config.sourceFormat = (MNN::CV::ImageFormat)srcType;

    // mean、normal
    jfloat *mean   = env->GetFloatArrayElements(mean_, NULL);
    jfloat *normal = env->GetFloatArrayElements(normal_, NULL);
    ::memcpy(config.mean, mean, 3 * sizeof(float));
    ::memcpy(config.normal, normal, 3 * sizeof(float));
    // filterType、wrap
    config.filterType = (MNN::CV::Filter)filterType;
    config.wrap       = (MNN::CV::Wrap)wrap;
    env->ReleaseFloatArrayElements(mean_, mean, 0);
    env->ReleaseFloatArrayElements(normal_, normal, 0);

    // matrix
    jfloat *matrixValue = env->GetFloatArrayElements(matrixValue_, NULL);
    MNN::CV::Matrix transform;
    transform.set9((float *)matrixValue);
    env->ReleaseFloatArrayElements(matrixValue_, matrixValue, 0);

    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
    process->setMatrix(transform);

    auto tensor = (MNN::Tensor *)tensorPtr;

    process->convert((const unsigned char *)bufferData, jwidth, jheight, 0, tensor);
    env->ReleaseByteArrayElements(jbufferData, bufferData, 0);

    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_taobao_android_mnn_MNNNetNative_nativeConvertBitmapToTensor(
    JNIEnv *env, jclass type, jobject srcBitmap, jlong tensorPtr, jint destFormat, jint filterType, jint wrap,
    jfloatArray matrixValue_, jfloatArray mean_, jfloatArray normal_) {
    MNN::CV::ImageProcess::Config config;
    config.destFormat = (MNN::CV::ImageFormat)destFormat;

    AndroidBitmapInfo bitmapInfo;
    AndroidBitmap_getInfo(env, srcBitmap, &bitmapInfo);
    switch (bitmapInfo.format) {
        case ANDROID_BITMAP_FORMAT_RGBA_8888:
            config.sourceFormat = MNN::CV::RGBA;
            break;
        case ANDROID_BITMAP_FORMAT_A_8:
            config.sourceFormat = MNN::CV::GRAY;
            break;
        default:
            MNN_ERROR("Don't support bitmap type: %d\n", bitmapInfo.format);
            return JNI_FALSE;
    }

    {
        auto size = env->GetArrayLength(matrixValue_);
        if (size < 9) {
            MNN_ERROR("Error matrix length:%d\n", size);
            return JNI_FALSE;
        }
    }

    // mean、normal
    jfloat *mean   = env->GetFloatArrayElements(mean_, NULL);
    jfloat *normal = env->GetFloatArrayElements(normal_, NULL);
    ::memcpy(config.mean, mean, 3 * sizeof(float));
    ::memcpy(config.normal, normal, 3 * sizeof(float));
    // filterType、wrap
    config.filterType = (MNN::CV::Filter)filterType;
    config.wrap       = (MNN::CV::Wrap)wrap;
    env->ReleaseFloatArrayElements(mean_, mean, 0);
    env->ReleaseFloatArrayElements(normal_, normal, 0);

    // matrix
    jfloat *matrixValue = env->GetFloatArrayElements(matrixValue_, NULL);
    MNN::CV::Matrix transform;
    transform.set9((float *)matrixValue);
    env->ReleaseFloatArrayElements(matrixValue_, matrixValue, 0);

    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
    process->setMatrix(transform);

    auto tensor  = (MNN::Tensor *)tensorPtr;
    void *pixels = nullptr;
    AndroidBitmap_lockPixels(env, srcBitmap, &pixels);

    process->convert((const unsigned char *)pixels, bitmapInfo.width, bitmapInfo.height, 0, tensor);
    AndroidBitmap_unlockPixels(env, srcBitmap);
    return JNI_TRUE;
}
