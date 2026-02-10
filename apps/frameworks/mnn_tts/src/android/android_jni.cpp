#include "mnn_tts/common.h"
#include <jni.h>
#include <iostream>

extern "C" JNIEXPORT jstring JNICALL
Java_com_alibaba_mnn_tts_MNNTTS_getHelloWorldFromJNI(JNIEnv* env, jclass) {
    return env->NewStringUTF("Hello from JNI");
}

extern "C" JNIEXPORT void JNICALL
Java_com_alibaba_mnn_tts_Native_platformFunction(JNIEnv*, jobject) {
    mnn_tts::platformFunction();
}

namespace mnn_tts {
void platformFunction() {
    std::cout << "This is the android platform" << std::endl;
}
}
