//
//  JNIHelper.cpp
//  MNN
//
//  Created by MNN on 2021/11/01
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(ANDROID) || defined(__ANDROID__)

#include "utils/JNIHelper.hpp"

static JavaVM* gjvm = NULL;

// Attaches JVM if needed. Needs to remember to detach JVM if this thread is not
// initially attached.
// See https://stackoverflow.com/questions/27923917/cant-execute-javavm-detachcurrentthread-attempting-to-detach-while-still-r/51501448
JNIEnv* AttachCurrentThread(bool* needsDetach) {
   if (gjvm == nullptr) {
       return nullptr;
   }
   *needsDetach = false;
   JNIEnv* env = nullptr;
   jint ret = gjvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
   if (ret == JNI_EDETACHED || !env) {
       JavaVMAttachArgs args;
       args.version = JNI_VERSION_1_6;
       args.group = nullptr;
       args.name = nullptr;
       ret = gjvm->AttachCurrentThread(&env, &args);
       if (ret != JNI_OK) {
           MNN_ERROR("Get JVM env failed!\n");
           return nullptr;
       }
       *needsDetach = true;
   }
   return env;
}

jint DetachCurrentThread() {
    return gjvm->DetachCurrentThread();
}

std::string getPackageName() {
    FILE *cmdline = fopen("/proc/self/cmdline", "r");

    if (!cmdline) {
        MNN_ERROR("Cannot find %s\n", "/proc/self/cmdline");
        return "";
    }

    char packageName[128] = { 0 };
    fread(packageName, sizeof(packageName), 1, cmdline);
    fclose(cmdline);
    return std::string(packageName);
}

extern "C" jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    MNN_PRINT("JNI_OnLoad for MNN main package.");
    gjvm = vm;
    return JNI_VERSION_1_6;
}

#endif  // defined(ANDROID) || defined(__ANDROID__)
