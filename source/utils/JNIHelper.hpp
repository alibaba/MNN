//
//  JNIHelper.hpp
//  MNN
//
//  Created by MNN on 2021/11/01
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(ANDROID) || defined(__ANDROID__)
#include <jni.h>
#include <android/log.h>
#include <string>
#include <unistd.h>
#include <MNN/MNNDefine.h>
/*
 * Calls the JNI method `methodname`. When there is an exception, clears the exception and returns JNI_EXCEPTION_RET.
 * This macro assumes the caller's scope defines the "env" and "JNI_EXCEPTION_RET"
 */
#define MNN_JNI(methodName, ...)  \
    ({ \
       auto ret = env->methodName(__VA_ARGS__);     \
       if (env->ExceptionOccurred()) {  \
           MNN_ERROR("Exception for JNI function %s with args: %s\n", #methodName, #__VA_ARGS__); \
           env->ExceptionDescribe();    \
           env->ExceptionClear(); \
           return JNI_EXCEPTION_RET; \
       } \
       ret; \
   }) \

/*
 * Calls a void JNI method `methodname`. When there is an exception, clears the exception and returns JNI_EXCEPTION_RET.
 * This macro assumes the caller's scope defines the "env" and "JNI_EXCEPTION_RET"
 */
#define MNN_JNI_VOID(methodName, ...)  \
   ({ \
      env->methodName(__VA_ARGS__);     \
      if (env->ExceptionOccurred()) {  \
          env->ExceptionDescribe();    \
          MNN_ERROR("Exception for JNI function %s with args %s\n", #methodName, #__VA_ARGS__); \
          env->ExceptionClear(); \
          return JNI_EXCEPTION_RET; \
      } \
  }) \


// ============ JNI Helper functions ======
JNIEnv* AttachCurrentThread(bool* needsDetach);

jint DetachCurrentThread();

std::string getPackageName();

#endif  // defined(ANDROID) || defined(__ANDROID__)
