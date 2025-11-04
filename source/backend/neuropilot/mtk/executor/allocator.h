#pragma once

#include <cstddef>
#include <unordered_map>

struct AHardwareBuffer;
struct NeuronMemory;

namespace mtk {

struct IOBuffer {
    void* buffer = nullptr;
    int fd = -1;
    size_t sizeBytes = 0;
    size_t usedSizeBytes = 0;

    // Optionally used by AHWB
    AHardwareBuffer* ahwbHandle = nullptr;

    // Optionally used by USDK
    NeuronMemory* neuronMemory = nullptr;

    // Helper functions
    explicit operator bool() const { return isAllocated(); }
    bool isAllocated() const { return buffer != nullptr && sizeBytes != 0; }
};

class Allocator {
public:
    virtual ~Allocator();

    IOBuffer allocate(const size_t size, const bool disableTracking = false);
    bool release(void* addr);

    void releaseAll();

    virtual bool allocateMemory(IOBuffer& ioBuffer) = 0;
    virtual bool releaseMemory(IOBuffer& ioBuffer) = 0;

protected:
    // A record of buffers allocated by this allocator instance, using the mapped addr as key.
    std::unordered_map<void*, IOBuffer> mAllocatedBuffers;
};
class AndroidHardwareBufferCompat;
class AhwBufferAllocator final : public Allocator {
public:
    AhwBufferAllocator();
    virtual ~AhwBufferAllocator() override;
    virtual bool allocateMemory(IOBuffer& ioBuffer) override;
    virtual bool releaseMemory(IOBuffer& ioBuffer) override;
private:
    std::shared_ptr<AndroidHardwareBufferCompat> mFuncPtr;
};

// An IOBuffer wrapper that handles its lifecycle via RAII.
// The buffer is only allocated during construction, and released during destruction.
class SmartIOBuffer {
public:
    explicit SmartIOBuffer(const size_t nbytes, const std::shared_ptr<Allocator>& allocator)
        : mBuffer(allocator->allocate(nbytes, /*disableTracking*/ true)), mAllocator(allocator) {}

    ~SmartIOBuffer() { mAllocator->releaseMemory(mBuffer); }

    // Get the underlying IOBuffer object
    IOBuffer& getIOBuffer() { return mBuffer; }

    // Get the underlying IOBuffer object
    const IOBuffer& getIOBuffer() const { return mBuffer; }

    // Get the allocated buffer address
    void* addr() const { return mBuffer.buffer; }

    // Get the allocated buffer size in bytes
    size_t size() const { return mBuffer.sizeBytes; }

private:
    IOBuffer mBuffer;
    std::shared_ptr<Allocator> mAllocator;
};

} // namespace mtk