
// HexagonBackend.hpp
#ifndef HexagonBackend_hpp
#define HexagonBackend_hpp

#include "core/Backend.hpp"
#include "backend/hexagon/htp-ops-lib/include/htp_command.h"
#include "core/BufferAllocator.hpp"

#include "MNN_generated.h"
#include <memory>
#include <vector>

namespace MNN {

class HexagonRuntime;
class HexagonCommand;
struct HexagonBuffer;

class HexagonBackend : public Backend {
public:
    HexagonBackend(const Backend::Info& info, const Runtime* runtime);
    virtual ~HexagonBackend();

    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;

    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;

    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    virtual const Runtime* getRuntime() override;

    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) override;
    BufferAllocator* getAllocator(int type = 0) const;
    static uint8_t* getPtr(const MemChunk& chunk);
    static uint8_t* getPtr(const Tensor* tensor);
    static int getBytes(const Tensor* tensor);
    static size_t getElementSize(const Tensor* tensor, int pack);
    static void fp32ToFp16(const float* src, int16_t* dst, size_t size);
    static void fp16ToFp32(const int16_t* src, float* dst, size_t size);
    size_t getElementSize(const Tensor* tensor) const;
    size_t getSize(const Tensor* tensor) const;

    // FD, offset
    static std::pair<int, int> getDevicePtr(const Tensor* tensor);
    static std::pair<int, int> getDevicePtr(const MemChunk& chunk);
    MemChunk allocCommandSlot(int size) const;
    void freeCommandSlot(const MemChunk& chunk) const;
    void pushCommand(const MemChunk& cmdChunk, int cmdSize, bool needCopy, bool dirty) const;
    int commandSerial() const;
    void flushCommand() const;
    void markHostInput(const Tensor* tensor) const;
    void markHostInput(const MemChunk& chunk, int size) const;
    void markHostOutput(const Tensor* tensor) const;
    void markHostOutput(const MemChunk& chunk, int size) const;
    void markHexagonOutput(const Tensor* tensor) const;
    bool hasPendingHexagonWrite(const Tensor* tensor) const;

private:
    void markDynamicHostOutput() const;
#ifdef MNN_HEXAGON_ASAN
    void asanRegisterDynamicTensor(const Tensor* tensor, size_t requestedSize) const;
    void asanUnregisterDynamicTensor(const Tensor* tensor) const;
    void asanRefreshDynamicTensorGuards() const;
    void asanClearDynamicTensorGuards() const;

    struct AsanDynamicTensorRecord {
        const Tensor* tensor = nullptr;
        size_t requestedSize = 0;
    };
#endif

    const HexagonRuntime* mRuntime;
    std::shared_ptr<BufferAllocator> mDynamicAlloc;
    std::shared_ptr<size_t> mDynamicGeneration;
    std::shared_ptr<BufferAllocator> mSeparateAlloc;
    std::shared_ptr<size_t> mSeparateGeneration;
    bool mAllOpSupport = true;
    int mDynamicPlaceholderFd = 0x3f000000;
    std::vector<std::unique_ptr<HexagonBuffer>> mDynamicPlaceholders;
#ifdef MNN_HEXAGON_ASAN
    mutable std::vector<AsanDynamicTensorRecord> mAsanDynamicTensors;
#endif
};

} // namespace MNN

#endif
