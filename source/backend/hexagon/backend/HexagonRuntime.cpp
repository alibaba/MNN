#include "HexagonCommand.hpp"
// HexagonRuntime.cpp
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <MNN/MNNDefine.h>
#include "HexagonRuntime.hpp"
#include "HexagonBackend.hpp"
#include "dsprpc_interface.h"
#include "dsp_op_name.h"
#include "schema/current/Command_generated.h"
namespace MNN {
static const char * HTP_OPS_DL_PATH = "libMNN_htpops.so";
static constexpr size_t MAX_MSG_SIZE = 65536;
static constexpr int gCommandGroupCapacity = 4096;
static constexpr int gCommandEntrySize = 3;

static std::mutex& dspMappedBuffersMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::vector<HexagonBuffer*>& dspMappedBuffers() {
    static std::vector<HexagonBuffer*> buffers;
    return buffers;
}

static void registerDspMappedBuffer(HexagonBuffer* buffer) {
    if (buffer == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(dspMappedBuffersMutex());
    dspMappedBuffers().push_back(buffer);
}

static void unregisterDspMappedBuffer(HexagonBuffer* buffer) {
    if (buffer == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(dspMappedBuffersMutex());
    auto& buffers = dspMappedBuffers();
    buffers.erase(std::remove(buffers.begin(), buffers.end(), buffer), buffers.end());
}

static std::vector<HexagonBuffer*> snapshotDspMappedBuffers() {
    std::lock_guard<std::mutex> lock(dspMappedBuffersMutex());
    return dspMappedBuffers();
}

class HexagonMmapManager {
public:
    static HexagonMmapManager& instance() {
        static HexagonMmapManager manager;
        return manager;
    }

    bool map(int domain, HexagonBuffer* buffer) {
        if (buffer == nullptr || buffer->ptr == nullptr || buffer->fd < 0 || buffer->size == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        auto iter = mRecords.find(buffer->fd);
        if (iter != mRecords.end()) {
            auto& record = iter->second;
            if (record.domain == domain && record.ptr == buffer->ptr && record.size == buffer->size) {
                record.refCount++;
                return true;
            }
            MNN_ERROR("[Hexagon] fastrpc mmap conflict fd=%d old(domain=%d,ptr=%p,size=%zu) new(domain=%d,ptr=%p,size=%zu)\n",
                      buffer->fd, record.domain, record.ptr, record.size, domain, buffer->ptr, buffer->size);
            return false;
        }

        int err = fastrpc_mmap(domain, buffer->fd, buffer->ptr, 0, buffer->size, FASTRPC_MAP_FD);
        if (err != 0) {
            FUNC_PRINT(err);
            MNN_PRINT("size=%zu in %s, %d \n", buffer->size, __func__, __LINE__);
            return false;
        }
        MmapRecord record;
        record.domain = domain;
        record.ptr = buffer->ptr;
        record.size = buffer->size;
        record.refCount = 1;
        mRecords[buffer->fd] = record;
        return true;
    }

    void unmap(int domain, HexagonBuffer* buffer) {
        if (buffer == nullptr || buffer->fd < 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        auto iter = mRecords.find(buffer->fd);
        if (iter == mRecords.end()) {
            return;
        }
        auto& record = iter->second;
        if (record.domain != domain || record.ptr != buffer->ptr || record.size != buffer->size) {
            MNN_ERROR("[Hexagon] fastrpc munmap conflict fd=%d old(domain=%d,ptr=%p,size=%zu) new(domain=%d,ptr=%p,size=%zu)\n",
                      buffer->fd, record.domain, record.ptr, record.size, domain, buffer->ptr, buffer->size);
            return;
        }
        record.refCount--;
        if (record.refCount <= 0) {
            fastrpc_munmap(record.domain, buffer->fd, record.ptr, record.size);
            mRecords.erase(iter);
        }
    }

    void unmapAll() {
        std::lock_guard<std::mutex> lock(mMutex);
        for (auto& iter : mRecords) {
            auto fd = iter.first;
            auto& record = iter.second;
            fastrpc_munmap(record.domain, fd, record.ptr, record.size);
        }
        mRecords.clear();
    }

private:
    struct MmapRecord {
        int domain = 0;
        void* ptr = nullptr;
        size_t size = 0;
        int refCount = 0;
    };

    std::mutex mMutex;
    std::unordered_map<int, MmapRecord> mRecords;
};

static HexagonMmapManager& gMmapManager() {
    return HexagonMmapManager::instance();
}

#ifdef MNN_HEXAGON_ASAN
static constexpr size_t gHexagonAsanGuardSize = 4096;
static constexpr size_t gHexagonAsanPreciseGuardSize = 256;
static constexpr size_t gHexagonAsanPageSize = 4096;
static constexpr uint8_t gHexagonAsanGuardValue = 0x7b;

struct HexagonAsanRangeRecord {
    const void* owner = nullptr;
    void* base = nullptr;
    size_t offset = 0;
    size_t requestedSize = 0;
    size_t guardSize = 0;
    const char* tag = nullptr;
};

static std::vector<HexagonAsanRangeRecord>& hexagonAsanRanges() {
    static std::vector<HexagonAsanRangeRecord> ranges;
    return ranges;
}

static bool hexagonAsanInitGuard(HexagonBuffer* buffer) {
    if (buffer == nullptr || buffer->ptr == nullptr || buffer->guardSize == 0) {
        return true;
    }
    auto guard = static_cast<uint8_t*>(buffer->ptr) + buffer->requestedSize;
    ::memset(guard, buffer->guardValue, buffer->guardSize);
    return true;
}

static bool hexagonAsanCheckGuard(const HexagonBuffer* buffer, const char* tag) {
    if (buffer == nullptr || buffer->ptr == nullptr || buffer->guardSize == 0) {
        return true;
    }
    const auto guard = static_cast<const uint8_t*>(buffer->ptr) + buffer->requestedSize;
    for (size_t i = 0; i < buffer->guardSize; ++i) {
        if (guard[i] != buffer->guardValue) {
            MNN_ERROR("[MNN::Hexagon][ASAN] buffer guard corrupted: tag=%s fd=%d requested=%zu mapped=%zu guardOffset=%zu expected=0x%x got=0x%x\n",
                      tag != nullptr ? tag : "", buffer->fd, buffer->requestedSize, buffer->size, i,
                      (unsigned)buffer->guardValue, (unsigned)guard[i]);
            return false;
        }
    }
    return true;
}

static uint8_t* hexagonAsanRangeGuardPtr(const HexagonAsanRangeRecord& record) {
    auto buffer = static_cast<HexagonBuffer*>(record.base);
    if (buffer == nullptr || buffer->ptr == nullptr) {
        return nullptr;
    }
    return static_cast<uint8_t*>(buffer->ptr) + record.offset + record.requestedSize;
}

static bool hexagonAsanCheckRange(const HexagonAsanRangeRecord& record, const char* checkTag) {
    auto guard = hexagonAsanRangeGuardPtr(record);
    if (guard == nullptr || record.guardSize == 0) {
        return true;
    }
    for (size_t i = 0; i < record.guardSize; ++i) {
        if (guard[i] != gHexagonAsanGuardValue) {
            auto buffer = static_cast<HexagonBuffer*>(record.base);
            MNN_ERROR("[MNN::Hexagon][ASAN] precise redzone corrupted: check=%s alloc=%s fd=%d offset=%zu requested=%zu guardOffset=%zu expected=0x%x got=0x%x\n",
                      checkTag != nullptr ? checkTag : "", record.tag != nullptr ? record.tag : "",
                      buffer != nullptr ? buffer->fd : -1, record.offset, record.requestedSize, i,
                      (unsigned)gHexagonAsanGuardValue, (unsigned)guard[i]);
            return false;
        }
    }
    return true;
}

static bool hexagonAsanCheckRanges(const void* owner, const char* tag) {
    bool valid = true;
    for (const auto& record : hexagonAsanRanges()) {
        if (owner != nullptr && record.owner != owner) {
            continue;
        }
        if (!hexagonAsanCheckRange(record, tag)) {
            valid = false;
        }
    }
    return valid;
}

static void hexagonAsanRegisterRange(const void* owner, const MemChunk& chunk, size_t requestedSize,
                                     size_t guardSize, const char* tag) {
    if (chunk.first == nullptr || requestedSize == 0 || guardSize == 0) {
        return;
    }
    HexagonAsanRangeRecord record;
    record.owner = owner;
    record.base = chunk.first;
    record.offset = chunk.second;
    record.requestedSize = requestedSize;
    record.guardSize = guardSize;
    record.tag = tag;
    auto guard = hexagonAsanRangeGuardPtr(record);
    if (guard == nullptr) {
        return;
    }
    ::memset(guard, gHexagonAsanGuardValue, guardSize);
    hexagonAsanRanges().emplace_back(record);
}

static void hexagonAsanUnregisterRange(const void* owner, const MemChunk& chunk) {
    auto& ranges = hexagonAsanRanges();
    for (auto iter = ranges.begin(); iter != ranges.end(); ++iter) {
        if (iter->owner == owner && iter->base == chunk.first && iter->offset == chunk.second) {
            hexagonAsanCheckRange(*iter, "free");
            ranges.erase(iter);
            return;
        }
    }
}

static void hexagonAsanClearRanges(const void* owner) {
    auto& ranges = hexagonAsanRanges();
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [owner](const HexagonAsanRangeRecord& record) {
        return owner == nullptr || record.owner == owner;
    }), ranges.end());
}

class HexagonAsanBufferAllocator : public BufferAllocator {
public:
    HexagonAsanBufferAllocator(std::shared_ptr<BufferAllocator> parent, const char* tag) : mParent(parent), mTag(tag) {
        syncTotalSize();
    }
    ~HexagonAsanBufferAllocator() override {
        hexagonAsanCheckRanges(this, "allocator destroy");
        hexagonAsanClearRanges(this);
    }

    MemChunk alloc(size_t size, bool separate = false, size_t align = 0) override {
        if (mParent == nullptr) {
            return MemChunk();
        }
        if (size > static_cast<size_t>(-1) - gHexagonAsanPreciseGuardSize) {
            MNN_ERROR("[MNN::Hexagon][ASAN] precise allocation overflow: size=%zu guard=%zu tag=%s\n",
                      size, gHexagonAsanPreciseGuardSize, mTag != nullptr ? mTag : "");
            return MemChunk();
        }
        auto chunk = mParent->alloc(size + gHexagonAsanPreciseGuardSize, separate, align);
        if (!chunk.invalid()) {
            hexagonAsanRegisterRange(this, chunk, size, gHexagonAsanPreciseGuardSize, mTag);
        }
        syncTotalSize();
        return chunk;
    }

    bool free(MemChunk chunk) override {
        hexagonAsanUnregisterRange(this, chunk);
        bool result = mParent != nullptr && mParent->free(chunk);
        syncTotalSize();
        return result;
    }

    void release(bool allRelease = true) override {
        hexagonAsanCheckRanges(this, "allocator release");
        hexagonAsanClearRanges(this);
        if (mParent != nullptr) {
            mParent->release(allRelease);
        }
        syncTotalSize();
    }

    void barrierBegin() override {
        if (mParent != nullptr) {
            mParent->barrierBegin();
        }
    }
    void barrierEnd() override {
        if (mParent != nullptr) {
            mParent->barrierEnd();
        }
        syncTotalSize();
    }
    void beginGroup() override {
        if (mParent != nullptr) {
            mParent->beginGroup();
        }
    }
    void endGroup() override {
        if (mParent != nullptr) {
            mParent->endGroup();
        }
    }
    void reset() override {
        hexagonAsanCheckRanges(this, "allocator reset");
        hexagonAsanClearRanges(this);
        if (mParent != nullptr) {
            mParent->reset();
        }
        syncTotalSize();
    }
    ErrorCode compute() override {
        if (mParent == nullptr) {
            return NO_ERROR;
        }
        auto code = mParent->compute();
        syncTotalSize();
        return code;
    }
    ErrorCode apply() override {
        if (mParent == nullptr) {
            return NO_ERROR;
        }
        auto code = mParent->apply();
        syncTotalSize();
        return code;
    }
    void sync() override {
        if (mParent != nullptr) {
            mParent->sync();
        }
    }

private:
    void syncTotalSize() {
        mTotalSize = mParent != nullptr ? mParent->totalSize() : 0;
    }

    std::shared_ptr<BufferAllocator> mParent;
    const char* mTag = nullptr;
};
#endif

class HexagonAllocator : public BufferAllocator::Allocator {
public:
    HexagonAllocator() {
        // Do nothing
    }
    virtual ~ HexagonAllocator() = default;
    virtual MemChunk onAlloc(size_t size, size_t align) override {
#ifdef MNN_HEXAGON_ASAN
        if (size > static_cast<size_t>(-1) - gHexagonAsanGuardSize) {
            MNN_ERROR("[MNN::Hexagon][ASAN] allocation size overflow: size=%zu guard=%zu\n", size, gHexagonAsanGuardSize);
            return MemChunk((void*)nullptr);
        }
        const size_t requestedSize = size;
        const size_t guardedSize = size + gHexagonAsanGuardSize;
        if (guardedSize > static_cast<size_t>(-1) - (gHexagonAsanPageSize - 1)) {
            MNN_ERROR("[MNN::Hexagon][ASAN] mapped size overflow: size=%zu guard=%zu page=%zu\n",
                      size, gHexagonAsanGuardSize, gHexagonAsanPageSize);
            return MemChunk((void*)nullptr);
        }
        const size_t mappedSize = ((guardedSize + gHexagonAsanPageSize - 1) / gHexagonAsanPageSize) * gHexagonAsanPageSize;
#else
        const size_t mappedSize = size;
#endif
        void * data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, mappedSize);
        if (nullptr == data) {
            FUNC_PRINT(1);
            return MemChunk((void*)nullptr);
        }
        auto fd = rpcmem_to_fd(data);
        if (fd == -1) {
            FUNC_PRINT(1);
            rpcmem_free(data);
            return MemChunk((void*)nullptr);
        }
        auto buf = new HexagonBuffer;
        buf->ptr = data;
        buf->fd = fd;
        buf->size = mappedSize;
#ifdef MNN_HEXAGON_ASAN
        buf->requestedSize = requestedSize;
        buf->guardSize = mappedSize - requestedSize;
        buf->guardValue = gHexagonAsanGuardValue;
        hexagonAsanInitGuard(buf);
#endif
        if (!gMmapManager().map(CDSP_DOMAIN_ID, buf)) {
            rpcmem_free(data);
            delete buf;
            return MemChunk((void*)nullptr);
        }
        registerDspMappedBuffer(buf);
//        MNN_PRINT("[Hexagon] Alloc %d, size=%d\n", buf->fd, buf->size);
        return buf;
    }
    virtual void onRelease(MemChunk ptr) override {
        auto buf = (HexagonBuffer*)ptr.first;
        if (buf == nullptr) {
            return;
        }
//        MNN_PRINT("[Hexagon] Free %d, size=%d\n", buf->fd, buf->size);
        unregisterDspMappedBuffer(buf);
#ifdef MNN_HEXAGON_ASAN
        hexagonAsanCheckGuard(buf, "release");
#endif
        gMmapManager().unmap(CDSP_DOMAIN_ID, buf);
        rpcmem_free(buf->ptr);
        delete buf;
    }
private:
};

// singleton HTP backend context
struct HexagonContext {
    // HTP ops backend library
    void * ops_dl_handle;
    bool init = false;
    int(*getHtpInfo)(int fd, int offset);
    int(*getHtpInfoProfile)(int fd, int offset);
    HexagonFunctions functions;
};

static HexagonContext* getContext() {
    static HexagonContext gContext;
    static std::mutex gContextMutex;
    std::lock_guard<std::mutex> lock(gContextMutex);
    if (!gContext.init) {
        gContext.ops_dl_handle = dlopen(HTP_OPS_DL_PATH, RTLD_LAZY | RTLD_LOCAL);
        if (nullptr == gContext.ops_dl_handle) {
            MNN_ERROR("[MNN::Hexagon] Open %s, error=%s\n", HTP_OPS_DL_PATH, dlerror());
            return nullptr;
        }
        gContext.getHtpInfo = (decltype(gContext.getHtpInfo))dlsym(gContext.ops_dl_handle, "htp_ops_rpc_getInfo");
        gContext.getHtpInfoProfile = (decltype(gContext.getHtpInfoProfile))dlsym(gContext.ops_dl_handle, "htp_ops_rpc_getInfoProfile");
        gContext.functions.execute_command_group = (decltype(gContext.functions.execute_command_group))dlsym(gContext.ops_dl_handle, "htp_rpc_execute_command_group");
        gContext.functions.execute_command_group_profile = (decltype(gContext.functions.execute_command_group_profile))dlsym(gContext.ops_dl_handle, "htp_rpc_execute_command_group_profile");
        if (gContext.functions.execute_command_group == nullptr) {
            MNN_ERROR("[MNN::Hexagon] Failed to dlsym htp_rpc_execute_command_group, error=%s\n", dlerror());
        } else {
            MNN_PRINT("[MNN::Hexagon] Successfully loaded htp_rpc_execute_command_group at %p\n", gContext.functions.execute_command_group);
        }

        gContext.init = true;
    }
    return &gContext;
}

static int gDspSessionRefCount = 0;

static std::mutex& dspSessionMutex() {
    static std::mutex mutex;
    return mutex;
}

static void closeDspSessionLocked(HexagonContext* context);

static bool openDspSessionLocked(HexagonContext* context) {
    if (context == nullptr || context->ops_dl_handle == nullptr) {
        return false;
    }
    using open_session_fn_type = int(int, int);
    using init_htp_ops_fn_type = int();
    auto open_session = reinterpret_cast<open_session_fn_type*>(dlsym(context->ops_dl_handle, "open_dsp_session"));
    auto init_htp_backend = reinterpret_cast<init_htp_ops_fn_type*>(dlsym(context->ops_dl_handle, "init_htp_backend"));
    if (open_session == nullptr || init_htp_backend == nullptr) {
        MNN_ERROR("[Hexagon] failed to find DSP session symbols\n");
        return false;
    }
    int err = open_session(CDSP_DOMAIN_ID, 1);
    if (err != 0) {
        FUNC_PRINT(err);
        return false;
    }
    err = init_htp_backend();
    if (err != 0) {
        FUNC_PRINT(err);
        closeDspSessionLocked(context);
        return false;
    }
    return true;
}

static void closeDspSessionLocked(HexagonContext* context) {
    gMmapManager().unmapAll();
    if (context != nullptr && context->ops_dl_handle != nullptr) {
        using close_session_fn = void();
        auto close_session = reinterpret_cast<close_session_fn*>(dlsym(context->ops_dl_handle, "close_dsp_session"));
        if (close_session != nullptr) {
            close_session();
        }
    }
}

static bool ensureDspSessionOnCurrentThreadLocked(HexagonContext* context) {
    return context != nullptr && context->ops_dl_handle != nullptr && gDspSessionRefCount > 0;
}

static bool acquireDspSession(HexagonContext* context) {
    std::lock_guard<std::mutex> lock(dspSessionMutex());
    if (context == nullptr || context->ops_dl_handle == nullptr) {
        return false;
    }
    if (gDspSessionRefCount > 0) {
        ++gDspSessionRefCount;
        return true;
    }

    rpcmem_init();
    if (!openDspSessionLocked(context)) {
        rpcmem_deinit();
        return false;
    }
    gDspSessionRefCount = 1;
    return true;
}

static void releaseDspSession(HexagonContext* context) {
    std::lock_guard<std::mutex> lock(dspSessionMutex());
    if (gDspSessionRefCount <= 0) {
        return;
    }
    --gDspSessionRefCount;
    if (gDspSessionRefCount > 0) {
        return;
    }
    closeDspSessionLocked(context);
    rpcmem_deinit();
}

const HexagonFunctions* HexagonRuntime::getDstFunctions() {
    auto context = getContext();
    if (nullptr != context) {
        return &context->functions;
    }
    return nullptr;
}

MemChunk HexagonRuntime::allocCommandSlot(int size) const {
    MNN_ASSERT(size <= MAX_MSG_SIZE);
    return mCommandAlloc->alloc(size);
}

void HexagonRuntime::freeCommandSlot(const MemChunk& chunk) const {
    if (chunk.first == nullptr) {
        return;
    }
    mCommandAlloc->free(chunk);
}

void HexagonRuntime::addSyncRecord(std::vector<SyncTensorRecord>& records, int fd, int offset, int size) const {
    if (fd < 0 || size <= 0) {
        return;
    }
    for (const auto& record : records) {
        if (record.fd == fd && record.offset == offset && record.size == size) {
            return;
        }
    }
    SyncTensorRecord record;
    record.fd = fd;
    record.offset = offset;
    record.size = size;
    records.push_back(record);
}

void HexagonRuntime::markHostInput(int fd, int offset, int size) const {
    addSyncRecord(mPendingHostInputs, fd, offset, size);
}

void HexagonRuntime::markHostOutput(int fd, int offset, int size) const {
    addSyncRecord(mPendingHostOutputs, fd, offset, size);
}

void HexagonRuntime::markHexagonOutput(int fd, int offset, int size) const {
    addSyncRecord(mPendingHexagonOutputs, fd, offset, size);
}

bool HexagonRuntime::hasPendingHexagonWrite(int fd, int offset, int size) const {
    if (fd < 0 || size <= 0) {
        return false;
    }
    const int64_t begin = offset;
    const int64_t end = begin + size;
    for (const auto& record : mPendingHexagonOutputs) {
        if (record.fd != fd || record.size <= 0) {
            continue;
        }
        const int64_t recordBegin = record.offset;
        const int64_t recordEnd = recordBegin + record.size;
        if (begin < recordEnd && recordBegin < end) {
            return true;
        }
    }
    return false;
}

bool HexagonRuntime::hasPendingCommand() const {
    return mCommandGroupCount > 0;
}

int HexagonRuntime::commandSerial() const {
    return mCommandSerial;
}

#ifdef MNN_HEXAGON_ASAN
bool HexagonRuntime::asanCheckAllBuffers(const char* tag) const {
    bool valid = true;
    for (auto buffer : snapshotDspMappedBuffers()) {
        if (!hexagonAsanCheckGuard(buffer, tag)) {
            valid = false;
        }
    }
    if (!hexagonAsanCheckRanges(nullptr, tag)) {
        valid = false;
    }
    return valid;
}

std::shared_ptr<BufferAllocator> HexagonRuntime::asanWrapAllocator(std::shared_ptr<BufferAllocator> allocator, const char* tag) {
    if (allocator == nullptr) {
        return allocator;
    }
    return std::shared_ptr<BufferAllocator>(new HexagonAsanBufferAllocator(allocator, tag));
}

size_t HexagonRuntime::asanPreciseGuardSize() {
    return gHexagonAsanPreciseGuardSize;
}

void HexagonRuntime::asanRegisterRange(const void* owner, const MemChunk& chunk, size_t requestedSize, size_t guardSize, const char* tag) {
    hexagonAsanRegisterRange(owner, chunk, requestedSize, guardSize, tag);
}

void HexagonRuntime::asanUnregisterRange(const void* owner, const MemChunk& chunk) {
    hexagonAsanUnregisterRange(owner, chunk);
}

void HexagonRuntime::asanClearRanges(const void* owner) {
    hexagonAsanClearRanges(owner);
}
#endif

#ifdef MNN_GPU_TIME_PROFILE
void HexagonRuntime::recordCopyBuffer(int direction, size_t bytes, uint64_t us, uint64_t flushUs) const {
    if (direction < 0 || direction >= 4) {
        return;
    }
    mProfileCopyCalls[direction]++;
    mProfileCopyBytes[direction] += bytes;
    mProfileCopyUs[direction] += us;
    mProfileCopyFlushUs[direction] += flushUs;
}
#endif

void HexagonRuntime::pushCommand(const MemChunk& cmdChunk, int cmdSize, bool needCopy, bool dirty) const {
    if (mCommandGroup == nullptr || cmdChunk.first == nullptr) {
        return;
    }
    MNN_ASSERT(cmdSize <= MAX_MSG_SIZE);
    if (mCommandGroupCount >= gCommandGroupCapacity) {
        flushCommand();
    }
    MemChunk queuedChunk = cmdChunk;
    if (needCopy) {
        const int copyAllocSize = cmdSize;
        queuedChunk = mCommandAlloc->alloc(copyAllocSize);
        if (queuedChunk.first == nullptr && mCommandGroupCount > 0) {
            flushCommand();
            queuedChunk = mCommandAlloc->alloc(copyAllocSize);
        }
        if (queuedChunk.first == nullptr) {
            MNN_ERROR("[Hexagon] pushCommand: failed to allocate queued command\n");
            return;
        }
        ::memcpy(HexagonBackend::getPtr(queuedChunk), HexagonBackend::getPtr(cmdChunk), cmdSize);
        mQueuedCommandChunks.emplace_back(queuedChunk);
    }
    auto cmdDev = HexagonBackend::getDevicePtr(queuedChunk);
    int idx = mCommandGroupCount;
    mCommandGroup->commands[idx * gCommandEntrySize] = cmdDev.first;
    mCommandGroup->commands[idx * gCommandEntrySize + 1] = cmdDev.second;
    mCommandGroup->commands[idx * gCommandEntrySize + 2] = dirty ? -cmdSize : cmdSize;
#ifdef MNN_GPU_TIME_PROFILE
    if (dirty) {
        mProfileDirtyCommandCount++;
    }
    if (needCopy) {
        mProfileCopiedCommandCount++;
    }
#endif
    mCommandGroupCount++;

    if (mCommandGroupCount >= gCommandGroupCapacity) {
        flushCommand();
    }
}

void HexagonRuntime::flushCommand() const {
    if (mCommandGroupCount == 0 && mPendingHostOutputs.empty() && mPendingHostInputs.empty()) {
        return;
    }
    if (mCommandGroup == nullptr || mCommandGroupChunk.first == nullptr) {
        MNN_ERROR("[Hexagon] flushCommand: command group buffer is not allocated\n");
        return;
    }
    auto context = getContext();
    auto functions = context != nullptr ? &context->functions : nullptr;
    MemChunk syncGroupChunk;
    int syncGroupSize = 0;
    int syncGroupFd = -1;
    int syncGroupOffset = 0;
    if (!mPendingHostInputs.empty() || !mPendingHostOutputs.empty()) {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputTensors;
        std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputTensors;
        inputTensors.reserve(mPendingHostInputs.size());
        outputTensors.reserve(mPendingHostOutputs.size());
        for (const auto& record : mPendingHostInputs) {
            inputTensors.emplace_back(DSPCOMMAND::CreateTensor(builder, record.fd, record.offset, record.size));
        }
        for (const auto& record : mPendingHostOutputs) {
            outputTensors.emplace_back(DSPCOMMAND::CreateTensor(builder, record.fd, record.offset, record.size));
        }
        auto syncGroup = DSPCOMMAND::CreateSyncGroup(builder, builder.CreateVector(inputTensors), builder.CreateVector(outputTensors));
        builder.Finish(syncGroup);
        syncGroupSize = (int)builder.GetSize();
        syncGroupChunk = mCommandAlloc->alloc(syncGroupSize);
        if (syncGroupChunk.first != nullptr) {
            ::memcpy(HexagonBackend::getPtr(syncGroupChunk), builder.GetBufferPointer(), builder.GetSize());
            auto syncDev = HexagonBackend::getDevicePtr(syncGroupChunk);
            syncGroupFd = syncDev.first;
            syncGroupOffset = syncDev.second;
        } else {
            if (mCommandGroupCount == 0) {
                MNN_ERROR("[Hexagon] flushCommand: failed to allocate sync group for sync-only flush\n");
                return;
            }
            syncGroupSize = 0;
        }
    }
    const int commandCount = mCommandGroupCount;
#ifdef MNN_HEXAGON_ASAN
    const bool asanBeforeFlush = asanCheckAllBuffers("before flushCommand");
#endif
#ifdef MNN_GPU_TIME_PROFILE
    mProfileFlushCount++;
    mProfileCommandCount += commandCount;
    mProfileMaxCommandsPerFlush = ALIMAX(mProfileMaxCommandsPerFlush, commandCount);
    if (commandCount == 0) {
        mProfileSyncOnlyFlushCount++;
    }
    if (
#ifdef MNN_HEXAGON_ASAN
        asanBeforeFlush &&
#endif
        functions && functions->execute_command_group_profile && mProfileChunk.first != nullptr) {
        auto groupDev = HexagonBackend::getDevicePtr(mCommandGroupChunk);
        auto profileDev = HexagonBackend::getDevicePtr(mProfileChunk);
        int ret = 0;
        {
            std::lock_guard<std::mutex> lock(dspSessionMutex());
            if (ensureDspSessionOnCurrentThreadLocked(context)) {
                ret = functions->execute_command_group_profile(groupDev.first, groupDev.second, commandCount,
                                                               syncGroupFd,
                                                               syncGroupOffset,
                                                               syncGroupSize,
                                                               profileDev.first, profileDev.second, 256 * sizeof(int));
            } else {
                ret = -1;
            }
        }
        if (ret != 0) {
            MNN_ERROR("[Hexagon] execute_command_group_profile failed with code %d\n", ret);
        }
    } else if (
#ifdef MNN_HEXAGON_ASAN
        asanBeforeFlush &&
#endif
        functions && functions->execute_command_group) {
        auto groupDev = HexagonBackend::getDevicePtr(mCommandGroupChunk);
        int ret = 0;
        {
            std::lock_guard<std::mutex> lock(dspSessionMutex());
            if (ensureDspSessionOnCurrentThreadLocked(context)) {
                ret = functions->execute_command_group(groupDev.first, groupDev.second, commandCount,
                                                       syncGroupFd,
                                                       syncGroupOffset,
                                                       syncGroupSize);
            } else {
                ret = -1;
            }
        }
        if (ret != 0) {
            MNN_ERROR("[Hexagon] execute_command_group failed with code %d\n", ret);
        }
#ifdef MNN_HEXAGON_ASAN
    } else if (!asanBeforeFlush) {
        MNN_ERROR("[MNN::Hexagon][ASAN] skip execute_command_group because guard is already corrupted\n");
#endif
    } else {
        MNN_ERROR("[Hexagon] execute_command_group function pointer is null! functions=%p\n", functions);
    }
#else
    if (
#ifdef MNN_HEXAGON_ASAN
        asanBeforeFlush &&
#endif
        functions && functions->execute_command_group) {
        auto groupDev = HexagonBackend::getDevicePtr(mCommandGroupChunk);
        int ret = 0;
        {
            std::lock_guard<std::mutex> lock(dspSessionMutex());
            if (ensureDspSessionOnCurrentThreadLocked(context)) {
                ret = functions->execute_command_group(groupDev.first, groupDev.second, commandCount,
                                                       syncGroupFd,
                                                       syncGroupOffset,
                                                       syncGroupSize);
            } else {
                ret = -1;
            }
        }
        if (ret != 0) {
            MNN_ERROR("[Hexagon] execute_command_group failed with code %d\n", ret);
        }
#ifdef MNN_HEXAGON_ASAN
    } else if (!asanBeforeFlush) {
        MNN_ERROR("[MNN::Hexagon][ASAN] skip execute_command_group because guard is already corrupted\n");
#endif
    } else {
        MNN_ERROR("[Hexagon] execute_command_group function pointer is null! functions=%p\n", functions);
    }
#endif
#ifdef MNN_HEXAGON_ASAN
    asanCheckAllBuffers("after flushCommand");
#endif
    if (syncGroupChunk.first != nullptr) {
        mCommandAlloc->free(syncGroupChunk);
    }
    for (const auto& chunk : mQueuedCommandChunks) {
        mCommandAlloc->free(chunk);
    }
    mQueuedCommandChunks.clear();
    mPendingHostInputs.clear();
    mPendingHostOutputs.clear();
    mPendingHexagonOutputs.clear();
    mCommandGroupCount = 0;
    mCommandSerial++;
    if (mCommandSerial == 0) {
        mCommandSerial = 1;
    }
}

float HexagonRuntime::onGetMemoryInMB() {
    auto staticMemoryInMB = mStaticAlloc->totalSize() / 1024.0f / 1024.0f;
    auto commandMemoryInMB = mCommandAlloc->totalSize() / 1024.0f / 1024.0f;
    auto weightInMB = mWeightAlloc->totalSize()  / 1024.0f / 1024.0f;
    float dynamicMemoryInMB = 0.0f;
//    for (auto& buf : mDynamic) {
//        dynamicMemoryInMB += buf.currentSize / 1024.0f / 1024.0f;
//    }
    return staticMemoryInMB + dynamicMemoryInMB + commandMemoryInMB + weightInMB;
}

HexagonRuntime::HexagonRuntime(const Backend::Info& info) {
    auto context = getContext();
    if (!acquireDspSession(context)) {
        MNN_ERROR("[Hexagon] failed to acquire DSP session\n");
    }
    std::shared_ptr<EagerBufferAllocator::Allocator> allocator(new HexagonAllocator());
    constexpr size_t staticMinAllocSize = 16 * 1024 * 1024;
    constexpr size_t commandMinAllocSize = 4 * 1024 * 1024;
    constexpr size_t weightMinAllocSize = 16 * 1024 * 1024;
    std::shared_ptr<BufferAllocator> staticAlloc(new EagerBufferAllocator(allocator, 128, staticMinAllocSize));
#ifdef MNN_HEXAGON_ASAN
    staticAlloc = asanWrapAllocator(staticAlloc, "static");
#endif
    mStaticAlloc = staticAlloc;
    mDynamicBuffer.root = BufferAllocator::Allocator::createRecurse(mStaticAlloc.get());
    std::shared_ptr<BufferAllocator> commandAlloc(new EagerBufferAllocator(allocator, 128, commandMinAllocSize));
#ifdef MNN_HEXAGON_ASAN
    commandAlloc = asanWrapAllocator(commandAlloc, "command");
#endif
    mCommandAlloc = commandAlloc;
    size_t groupSize = sizeof(DSPCommandGroup) + gCommandGroupCapacity * gCommandEntrySize * sizeof(int);
    mCommandGroupChunk = mCommandAlloc->alloc(groupSize);
    if (mCommandGroupChunk.first != nullptr) {
        mCommandGroup = (DSPCommandGroup*)HexagonBackend::getPtr(mCommandGroupChunk);
        uint8_t* ptr = (uint8_t*)mCommandGroup + sizeof(DSPCommandGroup);
        mCommandGroup->commands = (int*)ptr;
        mQueuedCommandChunks.reserve(gCommandGroupCapacity);
    } else {
        MNN_ERROR("[Hexagon] failed to allocate command group buffer\n");
    }
#ifdef MNN_GPU_TIME_PROFILE
    mProfileChunk = mCommandAlloc->alloc(256 * sizeof(int));
    if (mProfileChunk.first != nullptr) {
        ::memset(HexagonBackend::getPtr(mProfileChunk), 0, 256 * sizeof(int));
    }
#endif
    {
        auto infoMem = mStaticAlloc->alloc(256);
        auto buf = (HexagonBuffer*)infoMem.first;
        if (buf == nullptr) {
            MNN_ERROR("[Hexagon] failed to allocate HTP info buffer\n");
        } else {
#ifdef MNN_GPU_TIME_PROFILE
        int err = 0;
        {
            std::lock_guard<std::mutex> lock(dspSessionMutex());
            if (!ensureDspSessionOnCurrentThreadLocked(context)) {
                err = -1;
            } else if (context != nullptr && context->getHtpInfoProfile != nullptr) {
                err = context->getHtpInfoProfile(buf->fd, infoMem.second);
            } else if (context != nullptr && context->getHtpInfo != nullptr) {
                err = context->getHtpInfo(buf->fd, infoMem.second);
            } else {
                err = -1;
            }
        }
#else
        int err = 0;
        {
            std::lock_guard<std::mutex> lock(dspSessionMutex());
            if (ensureDspSessionOnCurrentThreadLocked(context) && context != nullptr && context->getHtpInfo != nullptr) {
                err = context->getHtpInfo(buf->fd, infoMem.second);
            } else {
                err = -1;
            }
        }
#endif
            if (0 != err) {
                FUNC_PRINT(err);
            }
            auto info = ((int*)((uint8_t*)buf->ptr + infoMem.second));
            ::memset(&mInfo, 0, sizeof(mInfo));
            ::memcpy(&mInfo, info, sizeof(mInfo));
            mStaticAlloc->free(infoMem);
        }
    }
    std::shared_ptr<BufferAllocator> weightAlloc(new EagerBufferAllocator(allocator, 128, weightMinAllocSize));
#ifdef MNN_HEXAGON_ASAN
    weightAlloc = asanWrapAllocator(weightAlloc, "weight");
#endif
    mWeightAlloc = weightAlloc;
    if (mInfo.maxThreads <= 0) {
        mInfo.maxThreads = 1;
    }
#ifdef MNN_GPU_TIME_PROFILE
    for (int i=0; i<6; ++i) {
        FUNC_PRINT_ALL(mInfo.flops[i], f);
    }
#endif
    MNN_PRINT("[MNN::Hexagon] vectorSize=%d, vtcmSize=%d, maxThreads=%d\n",
              mInfo.vectorSize, mInfo.vtcmSize, mInfo.maxThreads);

    if (false) {
        int err = 0;
        auto aMem = mStaticAlloc->alloc(mInfo.EP * mInfo.LP * sizeof(float));
        auto APtr = (int16_t*)((uint8_t*)((HexagonBuffer*)aMem.first)->ptr + aMem.second);
        for (int i=0; i<mInfo.EP; ++i) {
            for (int j=0; j<mInfo.LP; ++j) {
                APtr[mInfo.LP * i + j] = (i * mInfo.LP + j) / 10.0f;
            }
        }
        auto bMem = mStaticAlloc->alloc(mInfo.HP * mInfo.LP * sizeof(int16_t));
        auto BPtr = (int16_t*)((uint8_t*)((HexagonBuffer*)bMem.first)->ptr + bMem.second);
        for (int i=0; i<mInfo.HP; ++i) {
            for (int j=0; j<mInfo.LP; ++j) {
                BPtr[mInfo.LP * i + j] = (rand() % 100) / 100.0f;

//                if (i ==j) {
//                    BPtr[mInfo.LP * i + j] = 1.0f;
//                } else {
//                    BPtr[mInfo.LP * i + j] = 0.0f;
//                }
            }
        }
        std::vector<float> CTarget(mInfo.EP * mInfo.HP);
        for (int i=0; i<mInfo.HP; ++i) {
            for (int j=0; j<mInfo.EP; ++j) {
                float summer = 0.0f;
                for (int k=0; k<mInfo.LP; ++k) {
                    summer += APtr[k+j*mInfo.LP] * BPtr[i+k*mInfo.LP];
                }
                CTarget[j*mInfo.HP+i] = summer;
            }
        }

        auto testMem = mStaticAlloc->alloc(mInfo.EP * mInfo.HP * sizeof(float));
        auto buf = (HexagonBuffer*)testMem.first;

        if (0 != err) {
            FUNC_PRINT(err);
        }
        auto res = (int16_t*)((uint8_t*)buf->ptr + testMem.second);
        for (int i=0; i<mInfo.HP; ++i) {
            std::ostringstream os;
            for (int j=0; j<mInfo.EP; ++j)
            {
                os << (float)res[j*mInfo.HP+i] << " : " << CTarget[j*mInfo.HP+i] << ", ";
            }
            MNN_PRINT("%s\n", os.str().c_str());
        }
        mStaticAlloc->free(testMem);
        mStaticAlloc->free(aMem);
        mStaticAlloc->free(bMem);
    }
}

HexagonRuntime::~HexagonRuntime() {
    flushCommand();
#ifdef MNN_GPU_TIME_PROFILE
    if (mProfileChunk.first) {
        int* profile_data = (int*)HexagonBackend::getPtr(mProfileChunk);
        MNN_PRINT("Hexagon DSP Profile:\n");
        MNN_PRINT("Command groups: %d, commands: %d, sync-only groups: %d, max commands/group: %d\n",
                  mProfileFlushCount, mProfileCommandCount, mProfileSyncOnlyFlushCount, mProfileMaxCommandsPerFlush);
        MNN_PRINT("Command dirty: %d, clean-skip: %d, copied: %d\n",
                  mProfileDirtyCommandCount,
                  mProfileCommandCount - mProfileDirtyCommandCount,
                  mProfileCopiedCommandCount);
        for (int i = 0; i < 256; i++) {
            if (profile_data[i] > 0) {
                MNN_PRINT("DSPOpType %s (%d): %f ms\n", getDSPOpName(i), i, profile_data[i] / 1000.0f);
            }
        }
        static const char* copyNames[4] = {"HEXAGON_TO_HEXAGON", "HEXAGON_TO_CPU", "CPU_TO_HEXAGON", "CPU_TO_CPU"};
        uint64_t totalCopyUs = 0;
        uint64_t totalCopyFlushUs = 0;
        int totalCopyCalls = 0;
        for (int i = 0; i < 4; ++i) {
            totalCopyUs += mProfileCopyUs[i];
            totalCopyFlushUs += mProfileCopyFlushUs[i];
            totalCopyCalls += mProfileCopyCalls[i];
        }
        if (totalCopyCalls > 0) {
            MNN_PRINT("Hexagon onCopyBuffer Profile:\n");
            MNN_PRINT("onCopyBuffer total: calls=%d, time=%f ms, flush=%f ms, host=%f ms\n",
                      totalCopyCalls, totalCopyUs / 1000.0f, totalCopyFlushUs / 1000.0f,
                      (totalCopyUs - totalCopyFlushUs) / 1000.0f);
            for (int i = 0; i < 4; ++i) {
                if (mProfileCopyCalls[i] > 0) {
                    MNN_PRINT("onCopyBuffer %s: calls=%d, bytes=%llu, time=%f ms, flush=%f ms, host=%f ms\n",
                              copyNames[i], mProfileCopyCalls[i],
                              (unsigned long long)mProfileCopyBytes[i], mProfileCopyUs[i] / 1000.0f,
                              mProfileCopyFlushUs[i] / 1000.0f,
                              (mProfileCopyUs[i] - mProfileCopyFlushUs[i]) / 1000.0f);
                }
            }
        }
        mCommandAlloc->free(mProfileChunk);
        mProfileChunk = MemChunk();
    }
#endif
    if (mCommandGroup) {
        mCommandAlloc->free(mCommandGroupChunk);
        mCommandGroupChunk = MemChunk();
        mCommandGroup = nullptr;
    }

    mDynamicBuffer.release();
    mDynamicBuffer.root.reset();
    mWeightAlloc.reset();
    mCommandAlloc.reset();
    mStaticAlloc.reset();

    releaseDspSession(getContext());
}

Backend* HexagonRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    Backend::Info info;
    info.type = MNN_FORWARD_HEXAGON;
    info.numThread = 1; // Hexagon 通常单线程调度
    info.mode = Backend::Info::DIRECT;
    auto res = new HexagonBackend(info, this);
    return res;
}

void HexagonRuntime::onGabageCollect(int level) {
    flushCommand();
    if (mStaticAlloc) {
        mStaticAlloc->release(false);
    }
    if (mCommandAlloc) {
        mCommandAlloc->release(false);
    }
    if (level >= 100 && mWeightAlloc) {
        mWeightAlloc->release(false);
    }
    if (level >= 100) {
        mDynamicBuffer.release();
    }
}

bool HexagonRuntime::onCheckInfo(Backend::Info& info) const {
    if (info.type != MNN_FORWARD_HEXAGON) return false;
    info.numThread = 1; // 强制单线程
    return true;
}

// Creator
Runtime* HexagonRuntimeCreator::onCreate(const Backend::Info& info) const {
    auto* runtime = new HexagonRuntime(info);
    return runtime;
}

bool HexagonRuntimeCreator::onValid(Backend::Info& info) const {
    if (info.type != MNN_FORWARD_HEXAGON) return false;
    info.mode = Backend::Info::DIRECT;
    return true;
}

extern void registerHexagon() {
    auto context = getContext();
    if (nullptr == context) {
        return;
    }
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_HEXAGON, new HexagonRuntimeCreator);
}

} // namespace MNN
