//
// Created by ruoyi.sjd on 2025/5/9.
//
#include <jni.h>
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <string>
#include <android/log.h>
#include <sys/stat.h>
#include <unwind.h>
#include <dlfcn.h>
#include <cxxabi.h>
#include "mls_log.h"

static struct sigaction old_sigsegv;
static struct sigaction old_sigabrt;
static std::string      g_crash_dir;
static stack_t          g_altStack {};

static std::string currentTimestamp() {
    time_t now = time(nullptr);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_info);
    return std::string(buf);
}


struct BacktraceState {
    void** current;
    void** end;
};

static _Unwind_Reason_Code unwindCallback(struct _Unwind_Context* context, void* arg) {
    BacktraceState* state = static_cast<BacktraceState*>(arg);
    uintptr_t pc = _Unwind_GetIP(context);
    if (pc && state->current < state->end) {
        *state->current++ = reinterpret_cast<void*>(pc);
    }
    return _URC_NO_REASON;
}

static size_t captureBacktrace(void** buffer, size_t max) {
    BacktraceState state{buffer, buffer + max};
    _Unwind_Backtrace(unwindCallback, &state);
    return static_cast<size_t>(state.current - buffer);
}

static void writeStack(int fd) {
    constexpr int kMaxFrames = 50;
    void*         frames[kMaxFrames];
    int           count = static_cast<int>(captureBacktrace(frames, kMaxFrames));

    for (int i = 0; i < count; ++i) {
        Dl_info info;
        if (dladdr(frames[i], &info) && info.dli_sname) {
            int   status   = -1;
            char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
            const char* name = (status == 0 && demangled) ? demangled : info.dli_sname;

            char buf[512];
            snprintf(buf, sizeof(buf), "#%02d pc %p %s (%s+%lu)\n",
                     i, frames[i], name,
                     info.dli_fname,
                     (unsigned long)((uintptr_t)frames[i] - (uintptr_t)info.dli_saddr));
            write(fd, buf, strlen(buf));
            // -> logcat
            __android_log_print(ANDROID_LOG_FATAL, "CrashUtil", "%s", buf);

            free(demangled);
        } else {
            char buf[128];
            snprintf(buf, sizeof(buf), "#%02d pc %p\n", i, frames[i]);
            write(fd, buf, strlen(buf));
            __android_log_print(ANDROID_LOG_FATAL, "CrashUtil", "%s", buf);
        }
    }
}

static void signal_handler(int sig, siginfo_t* info, void* /*ucontext*/) {
    // log headline to logcat first so that developers see something *immediately*
    __android_log_print(ANDROID_LOG_FATAL, "CrashUtil",
                        ">>> Native crash: signal %d (%s) at address %p",
                        sig, strsignal(sig), info->si_addr);

    // make sure crash directory exists
    mkdir(g_crash_dir.c_str(), 0755);

    std::string filename = g_crash_dir + "/tombstone_" + currentTimestamp() + ".txt";
    int fd = open(filename.c_str(), O_CREAT | O_APPEND | O_WRONLY, 0644);
    if (fd >= 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Signal %d (%s) at address %p\n", sig, strsignal(sig), info->si_addr);
        write(fd, buf, strlen(buf));
        write(fd, "Stack trace:\n", strlen("Stack trace:\n"));
        writeStack(fd);
        close(fd);
    }
    // Restore the default handler and re‑raise so that Android still shows the standard crash dialog.
    if (sig == SIGSEGV) {
        sigaction(SIGSEGV, &old_sigsegv, nullptr);
    } else if (sig == SIGABRT) {
        sigaction(SIGABRT, &old_sigabrt, nullptr);
    }
    kill(getpid(), sig);
}


extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_utils_CrashUtil_initNative(JNIEnv *env, jobject thiz,
                                                           jstring crash_dir_j) {
    const char* crash_dir = env->GetStringUTFChars(crash_dir_j, nullptr);
    g_crash_dir = crash_dir;
    env->ReleaseStringUTFChars(crash_dir_j, crash_dir);
    constexpr size_t kAltStackSize = SIGSTKSZ;
    g_altStack.ss_sp    = malloc(kAltStackSize);
    g_altStack.ss_size  = kAltStackSize;
    g_altStack.ss_flags = 0;
    sigaltstack(&g_altStack, nullptr);

    // 3. set up sigaction for SIGSEGV and SIGABRT
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = signal_handler;
    sa.sa_flags     = SA_SIGINFO | SA_ONSTACK;   // SA_ONSTACK: use the alternate stack

    sigaction(SIGSEGV, &sa, &old_sigsegv);
    sigaction(SIGABRT, &sa, &old_sigabrt);
}