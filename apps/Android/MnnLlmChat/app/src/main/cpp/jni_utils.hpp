#pragma once

#include <jni.h>
#include <new>
#include <exception>
#include <utility>
#include "context_budget.hpp"

namespace mls {
namespace jni {

template <typename T>
class LocalRef {
public:
    LocalRef(JNIEnv* env, T value) : env_(env), value_(value) {}
    ~LocalRef() { if (value_) env_->DeleteLocalRef(value_); }
    LocalRef(const LocalRef&) = delete;
    LocalRef& operator=(const LocalRef&) = delete;
    T get() const { return value_; }
    T release() { return std::exchange(value_, nullptr); }
private:
    JNIEnv* env_;
    T value_;
};

inline void throwException(JNIEnv* env, const char* type, const char* message) {
    if (env->ExceptionCheck()) return;
    LocalRef<jclass> cls(env, env->FindClass(type));
    if (cls.get() && !env->ExceptionCheck()) env->ThrowNew(cls.get(), message);
}

template <typename F>
auto guarded(JNIEnv* env, F&& operation) -> decltype(operation()) {
    using Result = decltype(operation());
    if (env->ExceptionCheck()) return Result();
    try {
        return operation();
    } catch (const std::bad_alloc&) {
        throwException(env, "java/lang/OutOfMemoryError", "Native allocation failed");
    } catch (const std::exception&) {
        throwException(env, "java/lang/IllegalStateException", "Native operation failed");
    } catch (...) {
        throwException(env, "java/lang/IllegalStateException", "Unknown native failure");
    }
    return Result();
}

inline jstring newJString(JNIEnv* env, const std::string& utf8) {
    if (env->ExceptionCheck()) return nullptr;
    const std::u16string utf16 = context::utf8ToUtf16(utf8);
    return env->NewString(reinterpret_cast<const jchar*>(utf16.data()), static_cast<jsize>(utf16.size()));
}

inline std::string toStdString(JNIEnv* env, jstring value) {
    if (!value || env->ExceptionCheck()) return {};
    const jsize length = env->GetStringLength(value);
    if (env->ExceptionCheck()) return {};
    const jchar* chars = env->GetStringChars(value, nullptr);
    if (!chars) return {};
    struct Chars {
        JNIEnv* env;
        jstring value;
        const jchar* chars;
        ~Chars() { env->ReleaseStringChars(value, chars); }
    } acquired{env, value, chars};
    return context::utf16ToUtf8(reinterpret_cast<const char16_t*>(chars), static_cast<size_t>(length));
}

class JavaMap {
public:
    explicit JavaMap(JNIEnv* env) : env_(env) {
        if (env_->ExceptionCheck()) return;
        LocalRef<jclass> cls(env_, env_->FindClass("java/util/HashMap"));
        if (!cls.get() || env_->ExceptionCheck()) return;
        jmethodID ctor = env_->GetMethodID(cls.get(), "<init>", "()V");
        if (!ctor || env_->ExceptionCheck()) return;
        put_ = env_->GetMethodID(cls.get(), "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
        if (!put_ || env_->ExceptionCheck()) return;
        map_ = env_->NewObject(cls.get(), ctor);
    }
    ~JavaMap() { if (map_) env_->DeleteLocalRef(map_); }
    JavaMap(const JavaMap&) = delete;
    JavaMap& operator=(const JavaMap&) = delete;

    void putLong(const char* key, int64_t value) {
        if (!ready()) return;
        LocalRef<jclass> cls(env_, env_->FindClass("java/lang/Long"));
        if (!cls.get() || env_->ExceptionCheck()) return;
        jmethodID ctor = env_->GetMethodID(cls.get(), "<init>", "(J)V");
        if (!ctor || env_->ExceptionCheck()) return;
        put(key, env_->NewObject(cls.get(), ctor, static_cast<jlong>(value)));
    }
    void putBool(const char* key, bool value) {
        if (!ready()) return;
        LocalRef<jclass> cls(env_, env_->FindClass("java/lang/Boolean"));
        if (!cls.get() || env_->ExceptionCheck()) return;
        jmethodID ctor = env_->GetMethodID(cls.get(), "<init>", "(Z)V");
        if (!ctor || env_->ExceptionCheck()) return;
        put(key, env_->NewObject(cls.get(), ctor, static_cast<jboolean>(value)));
    }
    void putString(const char* key, const std::string& value) {
        if (ready()) put(key, newJString(env_, value));
    }
    jobject release() { return ready() ? std::exchange(map_, nullptr) : nullptr; }

private:
    bool ready() const { return !env_->ExceptionCheck() && map_ && put_; }
    void put(const char* key, jobject value) {
        LocalRef<jobject> boxed(env_, value);
        if (!ready() || !value) return;
        LocalRef<jstring> jkey(env_, env_->NewStringUTF(key));
        if (!jkey.get() || env_->ExceptionCheck()) return;
        LocalRef<jobject> previous(env_, env_->CallObjectMethod(map_, put_, jkey.get(), value));
    }
    JNIEnv* env_;
    jobject map_ = nullptr;
    jmethodID put_ = nullptr;
};

inline jobject errorMap(JNIEnv* env, const std::string& message, bool invalid = true) {
    if (env->ExceptionCheck()) return nullptr;
    JavaMap result(env);
    result.putBool("error", true);
    result.putString("message", message);
    if (invalid) result.putBool("session_invalid", true);
    return result.release();
}

class ProgressListener {
public:
    ProgressListener(JNIEnv* env, jobject listener) : env_(env), listener_(listener) {
        if (!listener_ || env_->ExceptionCheck()) return;
        LocalRef<jclass> cls(env_, env_->GetObjectClass(listener_));
        if (!cls.get() || env_->ExceptionCheck()) return;
        method_ = env_->GetMethodID(cls.get(), "onProgress", "(Ljava/lang/String;)Z");
    }
    bool onProgress(const std::string& text, bool end = false) const {
        if (env_->ExceptionCheck()) return true;
        if (!listener_ || !method_) return false;
        LocalRef<jstring> value(env_, end ? nullptr : newJString(env_, text));
        if (env_->ExceptionCheck() || (!end && !value.get())) return true;
        const jboolean stop = env_->CallBooleanMethod(listener_, method_, value.get());
        return env_->ExceptionCheck() || stop == JNI_TRUE;
    }
private:
    JNIEnv* env_;
    jobject listener_;
    jmethodID method_ = nullptr;
};

}
}
