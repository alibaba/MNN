#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef NO_ERROR
#undef NO_ERROR
#endif
#include <vector>

namespace MNN {
class ThreadPool;
namespace WindowsAffinity {
// Internal helper implemented beside the existing mask-keyed pool map.
unsigned long poolMask(const ThreadPool* pool);
inline bool& threadTainted() noexcept {
    static thread_local bool value = false;
    return value;
}
inline unsigned long validateMask(const std::vector<int>& ids) noexcept {
    if (ids.empty() || GetActiveProcessorGroupCount() != 1) {
        return 0;
    }
    DWORD_PTR processMask = 0, systemMask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
        return 0;
    }
    unsigned long mask = 0;
    for (int id : ids) {
        if (id < 0 || id >= 32) {
            return 0;
        }
        mask |= 1UL << id;
    }
    return (DWORD_PTR(mask) & ~processMask) == 0 ? mask : 0;
}
inline DWORD_PTR setMask(DWORD_PTR mask) noexcept {
    return SetThreadAffinityMask(GetCurrentThread(), mask);
}
class Scope {
public:
    explicit Scope(unsigned long mask) noexcept {
        if (threadTainted()) {
            mEntryError = ERROR_INVALID_STATE;
            return;
        }
        if (mask == 0) {
            mReady = true;
            return;
        }
        mPrevious = setMask(DWORD_PTR(mask));
        if (mPrevious == 0) {
            mEntryError = GetLastError();
            return;
        }
        mReady = true;
    }
    ~Scope() noexcept { finish(); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    bool ready() const noexcept { return mReady; }
    DWORD entryError() const noexcept { return mEntryError; }
    DWORD restoreError() const noexcept { return mRestoreError; }
    bool finish() noexcept {
        if (!mFinished) {
            mFinished = true;
            if (mPrevious != 0 && setMask(mPrevious) == 0) {
                mRestoreError = GetLastError();
                threadTainted() = true;
            }
        }
        return mReady && mRestoreError == 0 && !threadTainted();
    }
private:
    DWORD_PTR mPrevious = 0;
    DWORD mEntryError = 0;
    DWORD mRestoreError = 0;
    bool mReady = false;
    bool mFinished = false;
};
} // namespace WindowsAffinity
} // namespace MNN
