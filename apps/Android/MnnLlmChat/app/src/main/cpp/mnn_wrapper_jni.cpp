#include <jni.h>
#include "MNN/MNNDefine.h"

extern "C" {

JNIEXPORT jstring JNICALL Java_com_alibaba_mnnllm_android_MNN_nativeGetVersion(JNIEnv *env, jobject thiz) {
   return env->NewStringUTF(MNN_VERSION);
}

} // extern "C"
