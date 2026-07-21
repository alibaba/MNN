// HexagonRuntime.hpp
#ifndef HexagonRuntime_hpp
#define HexagonRuntime_hpp

#include "core/Backend.hpp"
#include "HexagonFunctions.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/hexagon/htp-ops-lib/include/htp_command.h"
#include "core/BufferAllocator.hpp"
#include <cstdint>
#include <memory>
#include <vector>
namespace MNN {
struct HexagonBuffer {
public:
    void* ptr;
    int fd;
    size_t size;
#ifdef MNN_HEXAGON_ASAN
    size_t requestedSize = 0;
    size_t guardSize = 0;
    uint8_t guardValue = 0;
#endif
};
struct HexagonInfo {
    int vectorSize = 0;
    int EP = 0;
    int LP = 0;
    int HP = 0;
    int vtcmSize = 0;
    int maxThreads = 0;
    float flops[6];// 32, 64, 128, 256, 512, 1024
    int hvxArch = 0;
};
class HexagonRuntime : public Runtime {
public:
    HexagonRuntime(const Backend::Info& info);
    virtual ~HexagonRuntime();

    virtual Backend* onCreate(const BackendConfig* config = nullptr, Backend* origin = nullptr) const override;
    virtual void onGabageCollect(int level) override;
    virtual bool onCheckInfo(Backend::Info& info) const override;
    virtual CompilerType onGetCompilerType() const override { return Compiler_Loop; }
    virtual float onGetMemoryInMB() override;
    friend class HexagonBackend;
    const HexagonInfo& info() const {
        return mInfo;
    }
    static const HexagonFunctions* getDstFunctions();
#ifdef MNN_HEXAGON_ASAN
    static std::shared_ptr<BufferAllocator> asanWrapAllocator(std::shared_ptr<BufferAllocator> allocator, const char* tag);
    static size_t asanPreciseGuardSize();
    static void asanRegisterRange(const void* owner, const MemChunk& chunk, size_t requestedSize, size_t guardSize, const char* tag);
    static void asanUnregisterRange(const void* owner, const MemChunk& chunk);
    static void asanClearRanges(const void* owner);
#endif
    MemChunk allocCommandSlot(int size) const;
    void freeCommandSlot(const MemChunk& chunk) const;
    void pushCommand(const MemChunk& cmdChunk, int cmdSize, bool needCopy, bool dirty) const;
    void flushCommand() const;
    void markHostInput(int fd, int offset, int size) const;
    void markHostOutput(int fd, int offset, int size) const;
    void markHexagonOutput(int fd, int offset, int size) const;
    bool hasPendingHexagonWrite(int fd, int offset, int size) const;
    bool hasPendingCommand() const;
    int commandSerial() const;
#ifdef MNN_HEXAGON_ASAN
    bool asanCheckAllBuffers(const char* tag) const;
#endif
#ifdef MNN_GPU_TIME_PROFILE
    void recordCopyBuffer(int direction, size_t bytes, uint64_t us, uint64_t flushUs) const;
#endif
private:
    struct SyncTensorRecord {
        int fd = -1;
        int offset = 0;
        int size = 0;
    };
    void addSyncRecord(std::vector<SyncTensorRecord>& records, int fd, int offset, int size) const;

    std::shared_ptr<BufferAllocator> mStaticAlloc;
    std::shared_ptr<BufferAllocator> mCommandAlloc;
    std::shared_ptr<BufferAllocator> mWeightAlloc;
    mutable SingleBufferWithAllocator mDynamicBuffer;
    HexagonInfo mInfo;
    mutable DSPCommandGroup* mCommandGroup = nullptr;
    mutable int mCommandGroupCount = 0;
    mutable int mCommandSerial = 1;
    mutable MemChunk mCommandGroupChunk;
    mutable std::vector<MemChunk> mQueuedCommandChunks;
    mutable std::vector<SyncTensorRecord> mPendingHostInputs;
    mutable std::vector<SyncTensorRecord> mPendingHostOutputs;
    mutable std::vector<SyncTensorRecord> mPendingHexagonOutputs;
#ifdef MNN_GPU_TIME_PROFILE
    mutable MemChunk mProfileChunk;
    mutable int mProfileFlushCount = 0;
    mutable int mProfileSyncOnlyFlushCount = 0;
    mutable int mProfileCommandCount = 0;
    mutable int mProfileDirtyCommandCount = 0;
    mutable int mProfileCopiedCommandCount = 0;
    mutable int mProfileMaxCommandsPerFlush = 0;
    mutable uint64_t mProfileCopyUs[4] = {0, 0, 0, 0};
    mutable uint64_t mProfileCopyFlushUs[4] = {0, 0, 0, 0};
    mutable uint64_t mProfileCopyBytes[4] = {0, 0, 0, 0};
    mutable int mProfileCopyCalls[4] = {0, 0, 0, 0};
#endif
};

class HexagonRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override;
    virtual bool onValid(Backend::Info& info) const override;
};

} // namespace MNN

#endif
