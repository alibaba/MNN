#include <jni.h>
#include <string>
#include "MNN/MNNDefine.h"

#ifndef GIT_COMMIT_ID
#define GIT_COMMIT_ID "unknown"
#endif

extern "C" {

JNIEXPORT jstring JNICALL Java_com_alibaba_mnnllm_android_MNN_nativeGetVersion(JNIEnv *env, jobject thiz) {
   std::string version = std::string(MNN_VERSION) + " (" + GIT_COMMIT_ID + ")";
   return env->NewStringUTF(version.c_str());
}

} // extern "C"
