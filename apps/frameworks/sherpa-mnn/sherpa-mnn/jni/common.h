// sherpa-mnn/jni/common.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_JNI_COMMON_H_
#define SHERPA_ONNX_JNI_COMMON_H_

#include <string>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if defined(_WIN32)
#if defined(SHERPA_ONNX_BUILD_SHARED_LIBS)
#define SHERPA_ONNX_EXPORT __declspec(dllexport)
#define SHERPA_ONNX_IMPORT __declspec(dllimport)
#else
#define SHERPA_ONNX_EXPORT
#define SHERPA_ONNX_IMPORT
#endif
#else  // WIN32
#define SHERPA_ONNX_EXPORT __attribute__((visibility("default")))

#define SHERPA_ONNX_IMPORT SHERPA_ONNX_EXPORT
#endif  // WIN32

#if defined(SHERPA_ONNX_BUILD_MAIN_LIB)
#define SHERPA_ONNX_API SHERPA_ONNX_EXPORT
#else
#define SHERPA_ONNX_API SHERPA_ONNX_IMPORT
#endif

// If you use ndk, you can find "jni.h" inside
// android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
#include "jni.h"  // NOLINT

#define SHERPA_ONNX_EXTERN_C extern "C" SHERPA_ONNX_API

// defined in jni.cc
jobject NewInteger(JNIEnv *env, int32_t value);
jobject NewFloat(JNIEnv *env, float value);

// Template function for non-void return types
template <typename Func, typename ReturnType>
ReturnType SafeJNI(JNIEnv *env, const char *functionName, Func func,
                   ReturnType defaultValue) {
  try {
    return func();
  } catch (const std::exception &e) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) + ": " + e.what();
      env->ThrowNew(exClass, errorMessage.c_str());
    }
  } catch (...) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) +
                                 ": Native exception: caught unknown exception";
      env->ThrowNew(exClass, errorMessage.c_str());
    }
  }
  return defaultValue;
}

// Specialization for void return type
template <typename Func>
void SafeJNI(JNIEnv *env, const char *functionName, Func func) {
  try {
    func();
  } catch (const std::exception &e) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) + ": " + e.what();
      env->ThrowNew(exClass, errorMessage.c_str());
    }
  } catch (...) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) +
                                 ": Native exception: caught unknown exception";
      env->ThrowNew(exClass, errorMessage.c_str());
    }
  }
}

// Helper function to validate JNI pointers
inline bool ValidatePointer(JNIEnv *env, jlong ptr,
                            const char *functionName, const char *message) {
  if (ptr == 0) {
    jclass exClass = env->FindClass("java/lang/NullPointerException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) + ": " + message;
      env->ThrowNew(exClass, errorMessage.c_str());
    }
    return false;
  }
  return true;
}

#endif  // SHERPA_ONNX_JNI_COMMON_H_
