// Copyright (c) Alibaba Group Holding Limited. All rights reserved.

#include <cstring>
#include <map>
#include "MNNTestSuite.h"
#include "backend/cpu/CPUBackend.hpp"
#include "core/MNNMemoryUtils.h"
#include "core/TensorUtils.hpp"

using namespace MNN;

// Keep released blocks allocated until teardown so the regression fails
// deterministically, without relying on a platform's use-after-free behavior.
class CopyQuarantineAllocator : public BufferAllocator::Allocator {
public:
    ~CopyQuarantineAllocator() override {
        for (auto& block : mBlocks) {
            MNNMemoryFreeAlign(block.first);
        }
    }
    MemChunk onAlloc(size_t size, size_t align) override {
        auto ptr = MNNMemoryAllocAlign(size, align);
        if (ptr) {
            mBlocks[ptr] = size;
        }
        return MemChunk(ptr);
    }
    void onRelease(MemChunk chunk) override {
        ++releases;
        ::memset(chunk.first, 0xdd, mBlocks.find(chunk.first)->second);
    }
    int releases = 0;

private:
    std::map<void*, size_t> mBlocks;
};

class CPUCopyLifetimeTest : public MNNTestCase {
    bool checkBackup() {
        BackendConfig config;
        config.precision = BackendConfig::Precision_High;
        config.flags = 4;
        Backend::Info info;
        info.type = MNN_FORWARD_CPU;
        info.numThread = 1;
        info.user = &config;
        auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
        std::unique_ptr<Runtime> runtime(creator->onCreate(info));
        auto cpuRuntime = static_cast<CPURuntime*>(runtime.get());
        auto guard = std::make_shared<CopyQuarantineAllocator>();
        cpuRuntime->buffer(0)->root = guard;
        cpuRuntime->buffer(1)->root = guard;
        std::unique_ptr<Backend> producer(runtime->onCreate(&config));
        std::unique_ptr<Backend> consumer(runtime->onCreate(&config));
        std::unique_ptr<Tensor> scratch(Tensor::createDevice<int>({4096}));
        std::unique_ptr<Tensor> source(Tensor::createDevice<int>({16}));
        std::unique_ptr<Tensor> destination(Tensor::createDevice<int>({16}));
        std::unique_ptr<Tensor> sentinel(Tensor::createDevice<int>({16}));
        auto plan = [](Backend* backend, Tensor* first, Tensor* second = nullptr) {
            backend->onResizeBegin();
            if (!backend->onAcquireBuffer(first, Backend::DYNAMIC) ||
                (second && !backend->onAcquireBuffer(second, Backend::DYNAMIC))) {
                return false;
            }
            backend->onReleaseBuffer(first, Backend::DYNAMIC);
            if (second) {
                backend->onReleaseBuffer(second, Backend::DYNAMIC);
            }
            return backend->onResizeEnd() == NO_ERROR;
        };
        // A has an old, large plan for buffer 1. B now owns a smaller live plan there.
        producer->onSelectDynamicAllocator(1, 2);
        if (!plan(producer.get(), scratch.get())) {
            return false;
        }
        runtime->onGabageCollect(100);
        consumer->onSelectDynamicAllocator(1, 2);
        if (!plan(consumer.get(), destination.get(), sentinel.get())) {
            return false;
        }
        consumer->onExecuteBegin();
        sentinel->host<int>()[0] = 12345;
        // Copying A's dynamic source in buffer 0 must not activate A's old buffer 1 plan.
        producer->onSelectDynamicAllocator(0, 2);
        if (!plan(producer.get(), source.get())) {
            consumer->onExecuteEnd();
            return false;
        }
        source->host<int>()[0] = 7;
        auto parent = cpuRuntime->buffer(1);
        auto address = parent->current.first;
        auto releases = guard->releases;
        producer->onCopyBuffer(source.get(), destination.get());
        consumer->onExecuteEnd();
        if (parent->current.first != address || guard->releases != releases || destination->host<int>()[0] != 7 ||
            sentinel->host<int>()[0] != 12345) {
            MNN_ERROR("Copy applied an unrelated backup allocator\n");
            return false;
        }
        return true;
    }

    template <typename T>
    bool checkCopy(BackendConfig::PrecisionMode precision, int bufferIndex) {
        auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
        BackendConfig config;
        config.precision = precision;
        Backend::Info info;
        info.type = MNN_FORWARD_CPU;
        info.numThread = 1;
        info.user = &config;
        std::unique_ptr<Runtime> runtime(creator->onCreate(info));
        auto cpuRuntime = static_cast<CPURuntime*>(runtime.get());
        auto parent = cpuRuntime->buffer(bufferIndex);
        auto guard = std::make_shared<CopyQuarantineAllocator>();
        parent->root = guard;
        std::unique_ptr<Backend> producer(runtime->onCreate(&config));
        config.precision = BackendConfig::Precision_High;
        config.flags = 4; // MNN_CPU_USE_DEFAULT_BACKEND: also exercise AVX2 -> CPU copies on x86.
        std::unique_ptr<Backend> consumer(runtime->onCreate(&config));
        producer->onSelectDynamicAllocator(bufferIndex, 2);
        consumer->onSelectDynamicAllocator(bufferIndex, 2);

        const int count = 513;
        const std::vector<int> shape = {1, 9, 3, 19}; // Channel tail and ndim > 1 reach packed AVX2 copies.
        std::unique_ptr<Tensor> host(Tensor::create<T>(shape, nullptr, Tensor::CAFFE));
        std::unique_ptr<Tensor> source(Tensor::createDevice<T>(shape, Tensor::CAFFE_C4));
        std::unique_ptr<Tensor> scratch(Tensor::createDevice<float>({65536}));
        std::unique_ptr<Tensor> destination(Tensor::createDevice<T>(shape, Tensor::CAFFE));
        std::unique_ptr<Tensor> sentinel(Tensor::createDevice<int>({64}));
        for (int i = 0; i < count; ++i) {
            host->host<T>()[i] = static_cast<T>(i - 256);
        }

        // The old inference graph has large scratch memory and a STATIC output.
        producer->onResizeBegin();
        if (!producer->onAcquireBuffer(source.get(), Backend::STATIC) ||
            !producer->onAcquireBuffer(scratch.get(), Backend::DYNAMIC)) {
            return false;
        }
        producer->onReleaseBuffer(scratch.get(), Backend::DYNAMIC);
        if (producer->onResizeEnd() != NO_ERROR) {
            return false;
        }
        producer->onCopyBuffer(host.get(), source.get());
        parent->release(); // The same shared-buffer release as LLM's post-Prefill GC.

        // The new CPU graph binds a smaller arena, with another live tensor.
        consumer->onResizeBegin();
        if (!consumer->onAcquireBuffer(sentinel.get(), Backend::DYNAMIC) ||
            !consumer->onAcquireBuffer(destination.get(), Backend::DYNAMIC)) {
            return false;
        }
        consumer->onReleaseBuffer(sentinel.get(), Backend::DYNAMIC);
        consumer->onReleaseBuffer(destination.get(), Backend::DYNAMIC);
        if (consumer->onResizeEnd() != NO_ERROR) {
            return false;
        }
        consumer->onExecuteBegin();
        for (int i = 0; i < 64; ++i) {
            sentinel->host<int>()[i] = 12345 + i;
        }
        auto address = parent->current.first;
        auto size = parent->currentSize;
        auto releases = guard->releases;
        std::unique_ptr<Tensor> reference(Tensor::createDevice<T>(shape, Tensor::CAFFE_C4));
        TensorUtils::refTensorContent(reference.get(), source.get());
        std::unique_ptr<Tensor> clone(Tensor::clone(source.get()));
        for (auto input : {source.get(), reference.get(), clone.get()}) {
            auto mapped = input->map(Tensor::MAP_TENSOR_READ, Tensor::CAFFE_C4);
            if (mapped == nullptr) {
                return false;
            }
            input->unmap(Tensor::MAP_TENSOR_READ, Tensor::CAFFE_C4, mapped);
            producer->onCopyBuffer(input, destination.get());
            for (int i = 0; i < count; ++i) {
                if (destination->host<T>()[i] != host->host<T>()[i]) {
                    MNN_ERROR("STATIC copy / alias mismatch at %d\n", i);
                    return false;
                }
            }
        }
        consumer->onExecuteEnd();
        if (parent->current.first != address || parent->currentSize != size || guard->releases != releases) {
            MNN_ERROR("Copy reallocated an unrelated graph's arena: precision=%d, type=%d, buffer=%d\n", precision,
                      host->getType().code, bufferIndex);
            return false;
        }
        for (int i = 0; i < 64; ++i) {
            if (sentinel->host<int>()[i] != 12345 + i) {
                MNN_ERROR("Copy corrupted another live tensor\n");
                return false;
            }
        }

        // A real dynamic upload still has to rebind after GC, even though
        // onReleaseBuffer cleared its MemObj during memory planning.
        parent->release();
        consumer->onCopyBuffer(host.get(), destination.get());
        if (parent->current.first == nullptr || parent->current.first == address) {
            MNN_ERROR("Dynamic upload did not prepare its allocator\n");
            return false;
        }
        for (int i = 0; i < count; ++i) {
            if (destination->host<T>()[i] != host->host<T>()[i]) {
                return false;
            }
        }

        // Also exercise an upload into the producer's own dynamic storage
        // (FP16 for a Low float tensor), then read it back for exact comparison.
        std::unique_ptr<Tensor> dynamicInput(Tensor::createDevice<T>(shape, Tensor::CAFFE));
        std::unique_ptr<Tensor> roundTrip(Tensor::create<T>(shape, nullptr, Tensor::CAFFE));
        producer->onResizeBegin();
        if (!producer->onAcquireBuffer(dynamicInput.get(), Backend::DYNAMIC)) {
            return false;
        }
        producer->onReleaseBuffer(dynamicInput.get(), Backend::DYNAMIC);
        if (producer->onResizeEnd() != NO_ERROR) {
            return false;
        }
        std::unique_ptr<Tensor> dynamicRef(Tensor::createDevice<T>(shape, Tensor::CAFFE));
        TensorUtils::refTensorContent(dynamicRef.get(), dynamicInput.get());
        std::unique_ptr<Tensor> dynamicClone(Tensor::clone(dynamicInput.get()));
        for (Tensor* target : {dynamicInput.get(), dynamicRef.get(), dynamicClone.get()}) {
            for (bool useMap : {false, true}) {
                parent->release();
                if (useMap) {
                    auto mapped = target->map(Tensor::MAP_TENSOR_WRITE, Tensor::CAFFE);
                    if (mapped == nullptr) {
                        return false;
                    }
                    ::memcpy(mapped, host->host<T>(), count * sizeof(T));
                    target->unmap(Tensor::MAP_TENSOR_WRITE, Tensor::CAFFE, mapped);
                } else {
                    producer->onCopyBuffer(host.get(), target);
                }
                // Check addresses before readback: another copy could hide a missing rebind.
                auto base = reinterpret_cast<uintptr_t>(parent->current.ptr());
                auto address = reinterpret_cast<uintptr_t>(target->host<void>());
                if (base == 0 || address < base || address >= base + parent->currentSize ||
                    target->host<void>() != dynamicInput->host<void>()) {
                    MNN_ERROR("Dynamic upload did not rebind the tensor and its alias\n");
                    return false;
                }
                producer->onCopyBuffer(dynamicInput.get(), roundTrip.get());
                for (int i = 0; i < count; ++i) {
                    if (roundTrip->host<T>()[i] != host->host<T>()[i]) {
                        MNN_ERROR("Dynamic upload / alias / map mismatch at %d\n", i);
                        return false;
                    }
                }
            }
        }
        MNN_PRINT("Copy lifetime passed: producer=%d, precision=%d, type=%d, buffer=%d\n", producer->type(), precision,
                  host->getType().code, bufferIndex);
        return true;
    }

public:
    bool run(int precision) override {
        bool success = checkBackup();
        for (auto mode : {BackendConfig::Precision_High, BackendConfig::Precision_Low}) {
            for (int index = 0; index < 2; ++index) {
                success = checkCopy<float>(mode, index) && success;
                success = checkCopy<int>(mode, index) && success;
            }
        }
        return success;
    }
};

MNNTestSuiteRegister(CPUCopyLifetimeTest, "core/cpu_copy_lifetime");
