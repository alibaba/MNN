// HexagonBackend.cpp
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "HexagonExecutionFactory.hpp"
#include "HexagonCommand.hpp"
#include <MNN/AutoTime.hpp>
#include "MNN_generated.h"
#include "schema/current/Command_generated.h"
#include <memory>
#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace MNN {

static uint16_t fp32ToFp16Scalar(float value) {
    if (value > 65504.0f) {
        value = 65504.0f;
    } else if (value < -65504.0f) {
        value = -65504.0f;
    }

    uint32_t bits;
    ::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000;
    const uint32_t exp = (bits >> 23) & 0xff;
    uint32_t mantissa = bits & 0x7fffff;

    if (exp == 0xff) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7c00);
        }
        return static_cast<uint16_t>(sign | 0x7e00);
    }

    int32_t halfExp = static_cast<int32_t>(exp) - 127 + 15;
    if (halfExp >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }

    if (halfExp <= 0) {
        if (halfExp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000;
        const int shift = 14 - halfExp;
        uint32_t halfMantissa = mantissa >> shift;
        const uint32_t roundBit = (mantissa >> (shift - 1)) & 1;
        const uint32_t stickyMask = (1u << (shift - 1)) - 1;
        if (roundBit != 0 && ((mantissa & stickyMask) != 0 || (halfMantissa & 1) != 0)) {
            ++halfMantissa;
        }
        return static_cast<uint16_t>(sign | halfMantissa);
    }

    uint32_t halfMantissa = mantissa >> 13;
    const uint32_t roundBits = mantissa & 0x1fff;
    if (roundBits > 0x1000 || (roundBits == 0x1000 && (halfMantissa & 1) != 0)) {
        ++halfMantissa;
        if (halfMantissa == 0x400) {
            halfMantissa = 0;
            ++halfExp;
            if (halfExp >= 0x1f) {
                return static_cast<uint16_t>(sign | 0x7c00);
            }
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExp) << 10) | halfMantissa);
}

static float fp16ToFp32Scalar(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000)) << 16;
    uint32_t exp = (value >> 10) & 0x1f;
    uint32_t mantissa = value & 0x03ff;

    uint32_t bits = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t fp32Exp = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                --fp32Exp;
            }
            mantissa &= 0x03ff;
            bits = sign | (static_cast<uint32_t>(fp32Exp + 127) << 23) | (mantissa << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
    }

    float result;
    ::memcpy(&result, &bits, sizeof(result));
    return result;
}

static void _HexagonMemChunkApplyToTensor(uint8_t* ptr, size_t offset, Tensor* tensor) {
    auto buffer = reinterpret_cast<HexagonBuffer*>(ptr);
    tensor->buffer().device = reinterpret_cast<uint64_t>(buffer);
    TensorUtils::getDescribeOrigin(tensor)->offset = offset;
}

HexagonBackend::HexagonBackend(const Backend::Info& info, const Runtime* runtime)
    : Backend(MNN_FORWARD_HEXAGON), mRuntime(static_cast<const HexagonRuntime*>(runtime)) {
    if (mRuntime->mDynamicBuffer.root == nullptr) {
        mRuntime->mDynamicBuffer.root = BufferAllocator::Allocator::createRecurse(mRuntime->mStaticAlloc.get());
    }
    mDynamicAlloc.reset(new DeferBufferAllocator(&mRuntime->mDynamicBuffer, 128, _HexagonMemChunkApplyToTensor));
    mDynamicGeneration.reset(new size_t(0));
    std::shared_ptr<BufferAllocator> separateAlloc(
        new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(mRuntime->mStaticAlloc.get()), 128));
#ifdef MNN_HEXAGON_ASAN
    separateAlloc = HexagonRuntime::asanWrapAllocator(separateAlloc, "separate");
#endif
    mSeparateAlloc = separateAlloc;
    mSeparateGeneration.reset(new size_t(0));
}

HexagonBackend::~HexagonBackend() {
    flushCommand();
    // TODO:
}
class EmptyExe : public Execution {
private:
    std::shared_ptr<Execution> mOriginExe;

public:
    EmptyExe(Backend* backend) : Execution(backend) {}
    virtual ~EmptyExe() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        return NO_ERROR;
    }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) { return NO_ERROR; }
};

BufferAllocator* HexagonBackend::getAllocator(int type) const {
    switch (type) {
        case 1:
            return mDynamicAlloc.get();
        case 2:
            return mRuntime->mWeightAlloc.get();
        default:
            return mRuntime->mStaticAlloc.get();
    }
}

Execution* HexagonBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op) {
    //    std::set<OpType> validOpSets {
    //        OpType_RoPE,
    //        OpType_LayerNorm,
    //        OpType_BinaryOp,
    //        OpType_While,
    //        OpType_Convolution,
    //        OpType_Attention,
    //        OpType_Scale,
    //        OpType_Raster,
    //    };
    //    if (validOpSets.find(op->type()) == validOpSets.end()) {
    //        return nullptr;
    //    }
    auto exe = HexagonExecutionFactory::create(op, inputs, outputs, this);
    if (nullptr != exe) {
        return exe;
    }
    mAllOpSupport = false;
    return nullptr;
}

void HexagonBackend::onResizeBegin() {
    flushCommand();
#ifdef MNN_HEXAGON_ASAN
    asanClearDynamicTensorGuards();
#endif
    ++(*mDynamicGeneration);
    mDynamicAlloc->reset();
}

ErrorCode HexagonBackend::onResizeEnd() {
    auto res = mDynamicAlloc->compute();
    if (res != NO_ERROR) {
        MNN_ERROR("[HexagonBackend] dynamic defer compute failed: %d\n", res);
    }
    return res;
}

void HexagonBackend::onExecuteBegin() const {
    auto res = mDynamicAlloc->apply();
    if (res != NO_ERROR) {
        MNN_ERROR("[HexagonBackend] dynamic defer apply failed: %d\n", res);
    }
#ifdef MNN_HEXAGON_ASAN
    asanRefreshDynamicTensorGuards();
#endif
    mRuntime->pCurrentStatus = res;
}

void HexagonBackend::onExecuteEnd() const {}

const Runtime* HexagonBackend::getRuntime() {
    return mRuntime;
}

#ifdef MNN_HEXAGON_ASAN
void HexagonBackend::asanRegisterDynamicTensor(const Tensor* tensor, size_t requestedSize) const {
    if (tensor == nullptr || requestedSize == 0) {
        return;
    }
    for (auto& record : mAsanDynamicTensors) {
        if (record.tensor == tensor) {
            record.requestedSize = requestedSize;
            return;
        }
    }
    AsanDynamicTensorRecord record;
    record.tensor = tensor;
    record.requestedSize = requestedSize;
    mAsanDynamicTensors.emplace_back(record);
}

void HexagonBackend::asanUnregisterDynamicTensor(const Tensor* tensor) const {
    if (tensor == nullptr) {
        return;
    }
    auto device = tensor->buffer().device;
    if (device != 0) {
        auto buffer = reinterpret_cast<HexagonBuffer*>(device);
        if (buffer != nullptr && buffer->ptr != nullptr) {
            const size_t offset = TensorUtils::getDescribeOrigin(tensor)->offset;
            HexagonRuntime::asanUnregisterRange(this, MemChunk(buffer, offset));
        }
    }
    for (auto iter = mAsanDynamicTensors.begin(); iter != mAsanDynamicTensors.end(); ++iter) {
        if (iter->tensor == tensor) {
            mAsanDynamicTensors.erase(iter);
            return;
        }
    }
}

void HexagonBackend::asanRefreshDynamicTensorGuards() const {
    const size_t guardSize = HexagonRuntime::asanPreciseGuardSize();
    for (const auto& record : mAsanDynamicTensors) {
        if (record.tensor == nullptr || record.requestedSize == 0) {
            continue;
        }
        auto device = record.tensor->buffer().device;
        if (device == 0) {
            continue;
        }
        auto buffer = reinterpret_cast<HexagonBuffer*>(device);
        if (buffer == nullptr || buffer->ptr == nullptr) {
            continue;
        }
        const size_t offset = TensorUtils::getDescribeOrigin(record.tensor)->offset;
        MemChunk chunk(buffer, offset);
        HexagonRuntime::asanUnregisterRange(this, chunk);
        HexagonRuntime::asanRegisterRange(this, chunk, record.requestedSize, guardSize, "dynamic-tensor");
    }
}

void HexagonBackend::asanClearDynamicTensorGuards() const {
    HexagonRuntime::asanClearRanges(this);
    mAsanDynamicTensors.clear();
}
#endif

int HexagonBackend::getBytes(const Tensor* tensor) {
    int bytes = tensor->getType().bytes();
    if (tensor->getType().code == halide_type_float) {
        bytes = 2;
    }
    return bytes;
}

void HexagonBackend::fp32ToFp16(const float* src, int16_t* dst, size_t size) {
    if (src == nullptr || dst == nullptr || size == 0) {
        return;
    }
    size_t i = 0;
    auto dstU16 = reinterpret_cast<uint16_t*>(dst);
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const float minMax[2] = {-65504.0f, 65504.0f};
    auto srcPtr = src;
    auto dstPtr = dstU16;
    size_t block16 = size / 16;
    if (block16 > 0) {
        asm volatile(
            "ldr w10, [%[minMax]]\n"
            "ldr w11, [%[minMax], #4]\n"
            "dup v30.4s, w10\n"
            "dup v31.4s, w11\n"
            "1:\n"
            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[src]], #64\n"
            "fmax v0.4s, v0.4s, v30.4s\n"
            "fmax v1.4s, v1.4s, v30.4s\n"
            "fmax v2.4s, v2.4s, v30.4s\n"
            "fmax v3.4s, v3.4s, v30.4s\n"
            "fmin v0.4s, v0.4s, v31.4s\n"
            "fmin v1.4s, v1.4s, v31.4s\n"
            "fmin v2.4s, v2.4s, v31.4s\n"
            "fmin v3.4s, v3.4s, v31.4s\n"
            "fcvtn v0.4h, v0.4s\n"
            "fcvtn v1.4h, v1.4s\n"
            "fcvtn v2.4h, v2.4s\n"
            "st1 {v0.4h, v1.4h}, [%[dst]], #16\n"
            "fcvtn v3.4h, v3.4s\n"
            "st1 {v2.4h, v3.4h}, [%[dst]], #16\n"
            "subs %x[block], %x[block], #1\n"
            "b.ne 1b\n"
            : [src] "+r"(srcPtr), [dst] "+r"(dstPtr), [block] "+r"(block16)
            : [minMax] "r"(minMax)
            : "cc", "memory", "x10", "x11", "v0", "v1", "v2", "v3", "v30", "v31");
    }
    i = (size / 16) * 16;
    srcPtr = src + i;
    dstPtr = dstU16 + i;
    size_t block4 = (size - i) / 4;
    if (block4 > 0) {
        asm volatile(
            "ldr w10, [%[minMax]]\n"
            "ldr w11, [%[minMax], #4]\n"
            "dup v30.4s, w10\n"
            "dup v31.4s, w11\n"
            "1:\n"
            "ld1 {v0.4s}, [%[src]], #16\n"
            "fmax v0.4s, v0.4s, v30.4s\n"
            "fmin v0.4s, v0.4s, v31.4s\n"
            "fcvtn v0.4h, v0.4s\n"
            "st1 {v0.4h}, [%[dst]], #8\n"
            "subs %x[block], %x[block], #1\n"
            "b.ne 1b\n"
            : [src] "+r"(srcPtr), [dst] "+r"(dstPtr), [block] "+r"(block4)
            : [minMax] "r"(minMax)
            : "cc", "memory", "x10", "x11", "v0", "v30", "v31");
    }
    i += ((size - i) / 4) * 4;
#else
    const float32x4_t minValue = vdupq_n_f32(-65504.0f);
    const float32x4_t maxValue = vdupq_n_f32(65504.0f);
    for (; i + 16 <= size; i += 16) {
        const float32x4_t v0 = vmaxq_f32(minValue, vminq_f32(maxValue, vld1q_f32(src + i)));
        const float32x4_t v1 = vmaxq_f32(minValue, vminq_f32(maxValue, vld1q_f32(src + i + 4)));
        const float32x4_t v2 = vmaxq_f32(minValue, vminq_f32(maxValue, vld1q_f32(src + i + 8)));
        const float32x4_t v3 = vmaxq_f32(minValue, vminq_f32(maxValue, vld1q_f32(src + i + 12)));
        const float16x4_t h0 = vcvt_f16_f32(v0);
        const float16x4_t h1 = vcvt_f16_f32(v1);
        const float16x4_t h2 = vcvt_f16_f32(v2);
        const float16x4_t h3 = vcvt_f16_f32(v3);
        vst1_u16(dstU16 + i, vreinterpret_u16_f16(h0));
        vst1_u16(dstU16 + i + 4, vreinterpret_u16_f16(h1));
        vst1_u16(dstU16 + i + 8, vreinterpret_u16_f16(h2));
        vst1_u16(dstU16 + i + 12, vreinterpret_u16_f16(h3));
    }
    for (; i + 8 <= size; i += 8) {
        const float32x4_t v0 = vmaxq_f32(minValue, vminq_f32(maxValue, vld1q_f32(src + i)));
        const float32x4_t v1 = vmaxq_f32(minValue, vminq_f32(maxValue, vld1q_f32(src + i + 4)));
        const float16x4_t h0 = vcvt_f16_f32(v0);
        const float16x4_t h1 = vcvt_f16_f32(v1);
        vst1_u16(dstU16 + i, vreinterpret_u16_f16(h0));
        vst1_u16(dstU16 + i + 4, vreinterpret_u16_f16(h1));
    }
    for (; i + 4 <= size; i += 4) {
        const float32x4_t v = vmaxq_f32(minValue, vminq_f32(maxValue, vld1q_f32(src + i)));
        const float16x4_t h = vcvt_f16_f32(v);
        vst1_u16(dstU16 + i, vreinterpret_u16_f16(h));
    }
#endif
    for (; i < size; ++i) {
        dstU16[i] = fp32ToFp16Scalar(src[i]);
    }
}

void HexagonBackend::fp16ToFp32(const int16_t* src, float* dst, size_t size) {
    if (src == nullptr || dst == nullptr || size == 0) {
        return;
    }
    size_t i = 0;
    const auto srcU16 = reinterpret_cast<const uint16_t*>(src);
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    auto srcPtr = srcU16;
    auto dstPtr = dst;
    size_t block16 = size / 16;
    if (block16 > 0) {
        asm volatile(
            "1:\n"
            "ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [%[src]], #32\n"
            "fcvtl v4.4s, v0.4h\n"
            "fcvtl v5.4s, v1.4h\n"
            "fcvtl v6.4s, v2.4h\n"
            "fcvtl v7.4s, v3.4h\n"
            "st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[dst]], #64\n"
            "subs %x[block], %x[block], #1\n"
            "b.ne 1b\n"
            : [src] "+r"(srcPtr), [dst] "+r"(dstPtr), [block] "+r"(block16)
            :
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }
    i = (size / 16) * 16;
    srcPtr = srcU16 + i;
    dstPtr = dst + i;
    size_t block4 = (size - i) / 4;
    if (block4 > 0) {
        asm volatile(
            "1:\n"
            "ld1 {v0.4h}, [%[src]], #8\n"
            "fcvtl v4.4s, v0.4h\n"
            "st1 {v4.4s}, [%[dst]], #16\n"
            "subs %x[block], %x[block], #1\n"
            "b.ne 1b\n"
            : [src] "+r"(srcPtr), [dst] "+r"(dstPtr), [block] "+r"(block4)
            :
            : "cc", "memory", "v0", "v4");
    }
    i += ((size - i) / 4) * 4;
#else
    for (; i + 16 <= size; i += 16) {
        const float16x4_t h0 = vreinterpret_f16_u16(vld1_u16(srcU16 + i));
        const float16x4_t h1 = vreinterpret_f16_u16(vld1_u16(srcU16 + i + 4));
        const float16x4_t h2 = vreinterpret_f16_u16(vld1_u16(srcU16 + i + 8));
        const float16x4_t h3 = vreinterpret_f16_u16(vld1_u16(srcU16 + i + 12));
        vst1q_f32(dst + i, vcvt_f32_f16(h0));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(h1));
        vst1q_f32(dst + i + 8, vcvt_f32_f16(h2));
        vst1q_f32(dst + i + 12, vcvt_f32_f16(h3));
    }
    for (; i + 8 <= size; i += 8) {
        const float16x4_t h0 = vreinterpret_f16_u16(vld1_u16(srcU16 + i));
        const float16x4_t h1 = vreinterpret_f16_u16(vld1_u16(srcU16 + i + 4));
        vst1q_f32(dst + i, vcvt_f32_f16(h0));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(h1));
    }
    for (; i + 4 <= size; i += 4) {
        const float16x4_t h = vreinterpret_f16_u16(vld1_u16(srcU16 + i));
        vst1q_f32(dst + i, vcvt_f32_f16(h));
    }
#endif
    for (; i < size; ++i) {
        dst[i] = fp16ToFp32Scalar(srcU16[i]);
    }
}

size_t HexagonBackend::getElementSize(const Tensor* tensor, int pack) {
    if (pack <= 0) {
        pack = 4;
    }
    size_t dataSize = 1;
    auto des = TensorUtils::getDescribe(tensor);
    for (int i = 0; i < tensor->dimensions(); i++) {
        size_t currentDimSize = tensor->length(i);
        if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && tensor->dimensions() <= 4 && 1 == i) {
            currentDimSize = UP_DIV(currentDimSize, pack) * pack;
        }
        dataSize *= currentDimSize;
    }
    return dataSize;
}

size_t HexagonBackend::getElementSize(const Tensor* tensor) const {
    return getElementSize(tensor, mRuntime->info().vectorSize);
}

size_t HexagonBackend::getSize(const Tensor* tensor) const {
    return getElementSize(tensor) * getBytes(tensor);
}

Backend::MemObj* HexagonBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    auto size = getSize(tensor);
    size_t allocSize = size;
#ifdef MNN_HEXAGON_ASAN
    if (storageType != STATIC && storageType != DYNAMIC_SEPERATE) {
        if (allocSize > static_cast<size_t>(-1) - HexagonRuntime::asanPreciseGuardSize()) {
            return nullptr;
        }
        allocSize += HexagonRuntime::asanPreciseGuardSize();
    }
#endif

    MemChunk chunk;
    if (storageType == STATIC) {
        chunk = mRuntime->mStaticAlloc->alloc(allocSize);
    } else if (storageType == DYNAMIC_SEPERATE) {
        chunk = mSeparateAlloc->alloc(allocSize, true);
    } else {
        chunk = mDynamicAlloc->alloc(allocSize, false);
    }

    if (chunk.invalid()) {
        return nullptr;
    }
    if (storageType == STATIC || chunk.first != nullptr) {
        auto buf = (HexagonBuffer*)chunk.first;
        ((Tensor*)tensor)->buffer().device = reinterpret_cast<uint64_t>(buf);
        TensorUtils::getDescribeOrigin(tensor)->offset = chunk.second;
    } else {
        std::unique_ptr<HexagonBuffer> placeholder(new HexagonBuffer);
        placeholder->ptr = nullptr;
        placeholder->fd = mDynamicPlaceholderFd--;
        placeholder->size = allocSize;
        ((Tensor*)tensor)->buffer().device = reinterpret_cast<uint64_t>(placeholder.get());
        TensorUtils::getDescribeOrigin(tensor)->offset = 0;
        mDynamicPlaceholders.emplace_back(std::move(placeholder));
        chunk.attach(const_cast<Tensor*>(tensor));
    }
#ifdef MNN_HEXAGON_ASAN
    if (storageType != STATIC && storageType != DYNAMIC_SEPERATE) {
        asanRegisterDynamicTensor(tensor, size);
    }
#endif

    class HexagonMemObj : public MemObj {
    public:
        HexagonMemObj(MemChunk c, std::shared_ptr<BufferAllocator> alloc, const HexagonRuntime* runtime,
                      std::shared_ptr<size_t> generation = nullptr
#ifdef MNN_HEXAGON_ASAN
                      ,
                      const HexagonBackend* backend = nullptr, const Tensor* tensor = nullptr, bool asanDynamic = false
#endif
                      )
            : mChunk(c),
              mAlloc(alloc),
              mRuntime(runtime),
              mGeneration(generation)
#ifdef MNN_HEXAGON_ASAN
              ,
              mBackend(backend),
              mTensor(tensor),
              mAsanDynamic(asanDynamic)
#endif
        {
            if (mGeneration) {
                mGenerationValue = *mGeneration;
            }
        }
        virtual MemChunk chunk() override { return mChunk; }
        virtual ~HexagonMemObj() {
            if (mRuntime != nullptr) {
                mRuntime->flushCommand();
            }
            if (mGeneration && *mGeneration != mGenerationValue) {
                return;
            }
#ifdef MNN_HEXAGON_ASAN
            if (mAsanDynamic && mBackend != nullptr) {
                mBackend->asanUnregisterDynamicTensor(mTensor);
            }
#endif
            mAlloc->free(mChunk);
        }

    private:
        MemChunk mChunk;
        std::shared_ptr<BufferAllocator> mAlloc;
        const HexagonRuntime* mRuntime;
        std::shared_ptr<size_t> mGeneration;
        size_t mGenerationValue = 0;
#ifdef MNN_HEXAGON_ASAN
        const HexagonBackend* mBackend = nullptr;
        const Tensor* mTensor = nullptr;
        bool mAsanDynamic = false;
#endif
    };

    if (storageType == STATIC) {
        return new HexagonMemObj(chunk, mRuntime->mStaticAlloc, mRuntime);
    }
    if (storageType == DYNAMIC_SEPERATE) {
        return new HexagonMemObj(chunk, mSeparateAlloc, mRuntime, mSeparateGeneration);
    }
    return new HexagonMemObj(chunk, mDynamicAlloc, mRuntime, mDynamicGeneration
#ifdef MNN_HEXAGON_ASAN
                             ,
                             this, tensor, true
#endif
    );
}

bool HexagonBackend::onClearBuffer() {
    flushCommand();
#ifdef MNN_HEXAGON_ASAN
    asanClearDynamicTensorGuards();
#endif
    ++(*mDynamicGeneration);
    mDynamicAlloc->release(true);
    ++(*mSeparateGeneration);
    mSeparateAlloc->release(true);
    return true;
}

static bool isNc4hw4Tensor(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    return des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
}

static size_t getNc4hw4HostElementCount(const Tensor* tensor) {
    size_t size = 1;
    for (int i = 0; i < tensor->dimensions(); ++i) {
        size_t current = (size_t)tensor->length(i);
        if (i == 1) {
            current = (size_t)UP_DIV((int)current, 4) * 4;
        }
        size *= current;
    }
    return size;
}

static size_t getNc4hw4DeviceElementCount(const Tensor* tensor, int pack) {
    size_t size = 1;
    for (int i = 0; i < tensor->dimensions(); ++i) {
        size_t current = (size_t)tensor->length(i);
        if (i == 1) {
            current = (size_t)UP_DIV((int)current, pack) * pack;
        }
        size *= current;
    }
    return size;
}

static void nc4hw4HostToDevice(const float* src, int16_t* dst, const Tensor* tensor, int pack,
                               void (*fp32tofp16)(const float*, int16_t*, size_t)) {
    int n = tensor->dimensions() > 0 ? tensor->length(0) : 1;
    int c = tensor->dimensions() > 1 ? tensor->length(1) : 1;
    int h = tensor->dimensions() > 2 ? tensor->length(2) : 1;
    int w = tensor->dimensions() > 3 ? tensor->length(3) : 1;
    const int plane = n * h * w;
    const int c4 = UP_DIV(c, 4);
    const int cPack = UP_DIV(c, pack);
    ::memset(dst, 0, (size_t)cPack * plane * pack * sizeof(int16_t));

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const float32x4_t minValue = vdupq_n_f32(-65504.0f);
    const float32x4_t maxValue = vdupq_n_f32(65504.0f);
    for (int cb = 0; cb < c4; ++cb) {
        const int ci = cb * 4;
        const int valid = std::min(4, c - ci);
        const int dstPack = ci / pack;
        const int dstOffset = ci % pack;
        const float* srcChannel = src + (size_t)cb * plane * 4;
        int16_t* dstChannel = dst + ((size_t)dstPack * plane * pack + dstOffset);
        if (valid == 4) {
            for (int pi = 0; pi < plane; ++pi) {
                float32x4_t v = vld1q_f32(srcChannel + pi * 4);
                v = vmaxq_f32(minValue, vminq_f32(maxValue, v));
                vst1_u16(reinterpret_cast<uint16_t*>(dstChannel + (size_t)pi * pack),
                         vreinterpret_u16_f16(vcvt_f16_f32(v)));
            }
            continue;
        }
        for (int pi = 0; pi < plane; ++pi) {
            float32x4_t v = vld1q_f32(srcChannel + pi * 4);
            v = vmaxq_f32(minValue, vminq_f32(maxValue, v));
            const uint16x4_t hv = vreinterpret_u16_f16(vcvt_f16_f32(v));
            uint16_t tmp[4];
            vst1_u16(tmp, hv);
            uint16_t* dstRow = reinterpret_cast<uint16_t*>(dstChannel + (size_t)pi * pack);
            for (int k = 0; k < valid; ++k) {
                dstRow[k] = tmp[k];
            }
        }
    }
#else
    std::vector<int16_t> fp16Temp((size_t)c4 * 4 * plane, 0);
    fp32tofp16(src, fp16Temp.data(), fp16Temp.size());
    for (int cb = 0; cb < c4; ++cb) {
        const int ci = cb * 4;
        const int valid = std::min(4, c - ci);
        const int dstPack = ci / pack;
        const int dstOffset = ci % pack;
        const int16_t* srcChannel = fp16Temp.data() + (size_t)cb * plane * 4;
        int16_t* dstChannel = dst + ((size_t)dstPack * plane * pack + dstOffset);
        for (int pi = 0; pi < plane; ++pi) {
            int16_t* dstRow = dstChannel + (size_t)pi * pack;
            for (int k = 0; k < valid; ++k) {
                dstRow[k] = srcChannel[pi * 4 + k];
            }
        }
    }
#endif
}

static void nc4hw4DeviceToHost(const int16_t* src, float* dst, const Tensor* tensor, int pack,
                               void (*fp16tofp32)(const int16_t*, float*, size_t)) {
    int n = tensor->dimensions() > 0 ? tensor->length(0) : 1;
    int c = tensor->dimensions() > 1 ? tensor->length(1) : 1;
    int h = tensor->dimensions() > 2 ? tensor->length(2) : 1;
    int w = tensor->dimensions() > 3 ? tensor->length(3) : 1;
    int c4 = UP_DIV(c, 4);
    int cPack = UP_DIV(c, pack);

    size_t hostElementCount = (size_t)n * c4 * 4 * h * w;
    std::vector<int16_t> srcFp16(hostElementCount, 0);

    for (int ni = 0; ni < n; ++ni) {
        for (int hi = 0; hi < h; ++hi) {
            for (int wi = 0; wi < w; ++wi) {
                for (int ci = 0; ci < c; ++ci) {
                    int srcIdx = ((ci / pack) * n * h * w + ni * h * w + hi * w + wi) * pack + (ci % pack);
                    int dstIdx = ((ci / 4) * n * h * w + ni * h * w + hi * w + wi) * 4 + (ci % 4);
                    srcFp16[dstIdx] = src[srcIdx];
                }
            }
        }
    }

    fp16tofp32(srcFp16.data(), dst, hostElementCount);
}

static void nchwHostToDevice(const float* src, int16_t* dst, const Tensor* tensor, int pack,
                             void (*fp32tofp16)(const float*, int16_t*, size_t)) {
    int n = tensor->dimensions() > 0 ? tensor->length(0) : 1;
    int c = tensor->dimensions() > 1 ? tensor->length(1) : 1;
    int h = tensor->dimensions() > 2 ? tensor->length(2) : 1;
    int w = tensor->dimensions() > 3 ? tensor->length(3) : 1;

    std::vector<int16_t> fp16Temp(tensor->elementSize(), 0);
    fp32tofp16(src, fp16Temp.data(), tensor->elementSize());

    for (int ni = 0; ni < n; ++ni) {
        for (int hi = 0; hi < h; ++hi) {
            for (int wi = 0; wi < w; ++wi) {
                for (int ci = 0; ci < c; ++ci) {
                    int srcIdx = ni * c * h * w + ci * h * w + hi * w + wi;
                    int dstIdx = ((ci / pack) * n * h * w + ni * h * w + hi * w + wi) * pack + (ci % pack);
                    dst[dstIdx] = fp16Temp[srcIdx];
                }
            }
        }
    }

    const size_t deviceElementCount = getNc4hw4DeviceElementCount(tensor, pack);
    for (size_t i = 0; i < deviceElementCount; ++i) {
        int linearIdx = (int)i;
        int innerCi = linearIdx % pack;
        linearIdx /= pack;
        int wi = linearIdx % w;
        linearIdx /= w;
        int hi = linearIdx % h;
        linearIdx /= h;
        int ni = linearIdx % n;
        linearIdx /= n;
        int cPackIdx = linearIdx;

        int ci = cPackIdx * pack + innerCi;
        if (!(ci < c && ni < n && hi < h && wi < w)) {
            dst[i] = 0;
        }
    }
}

static void deviceToNchwHost(const int16_t* src, float* dst, const Tensor* tensor, int pack,
                             void (*fp16tofp32)(const int16_t*, float*, size_t)) {
    int n = tensor->dimensions() > 0 ? tensor->length(0) : 1;
    int c = tensor->dimensions() > 1 ? tensor->length(1) : 1;
    int h = tensor->dimensions() > 2 ? tensor->length(2) : 1;
    int w = tensor->dimensions() > 3 ? tensor->length(3) : 1;

    size_t hostElementCount = tensor->elementSize();
    std::vector<int16_t> srcFp16(hostElementCount, 0);

    for (int ni = 0; ni < n; ++ni) {
        for (int hi = 0; hi < h; ++hi) {
            for (int wi = 0; wi < w; ++wi) {
                for (int ci = 0; ci < c; ++ci) {
                    int srcIdx = ((ci / pack) * n * h * w + ni * h * w + hi * w + wi) * pack + (ci % pack);
                    int dstIdx = ni * c * h * w + ci * h * w + hi * w + wi;
                    srcFp16[dstIdx] = src[srcIdx];
                }
            }
        }
    }

    fp16tofp32(srcFp16.data(), dst, hostElementCount);
}

void HexagonBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef MNN_GPU_TIME_PROFILE
    Timer copyTimer;
    uint64_t copyFlushUs = 0;
#endif
    bool srcIsNc4hw4 = isNc4hw4Tensor(srcTensor);
    bool dstIsNc4hw4 = isNc4hw4Tensor(dstTensor);
    int pack = mRuntime->info().vectorSize;
    if (pack <= 0) {
        pack = 4;
    }

    // For non-float types, just memcpy directly.
    // This avoids the issue where Arm82Backend rejects HEXAGON type tensors
    bool srcIsFloat = (srcTensor->getType().code == halide_type_float);
    bool dstIsFloat = (dstTensor->getType().code == halide_type_float);
    auto srcBackend = TensorUtils::getDescribeOrigin(srcTensor)->getBackend();
    auto dstBackend = TensorUtils::getDescribeOrigin(dstTensor)->getBackend();
    MNNForwardType srcType = srcBackend ? srcBackend->type() : MNN_FORWARD_CPU;
    MNNForwardType dstType = dstBackend ? dstBackend->type() : MNN_FORWARD_CPU;
    size_t profileBytes = std::min(getSize(srcTensor), getSize(dstTensor));
#ifdef MNN_GPU_TIME_PROFILE
    int profileDirection = 3;
    if (srcType == MNN_FORWARD_HEXAGON && dstType == MNN_FORWARD_HEXAGON) {
        profileDirection = 0;
    } else if (srcType == MNN_FORWARD_HEXAGON) {
        profileDirection = 1;
    } else if (dstType == MNN_FORWARD_HEXAGON) {
        profileDirection = 2;
    }
#define MNN_HEXAGON_RECORD_COPY_BUFFER()                                                                   \
    do {                                                                                                   \
        mRuntime->recordCopyBuffer(profileDirection, profileBytes, copyTimer.durationInUs(), copyFlushUs); \
    } while (0)
#define MNN_HEXAGON_PROFILE_FLUSH_COMMAND()       \
    do {                                          \
        Timer flushTimer;                         \
        flushCommand();                           \
        copyFlushUs += flushTimer.durationInUs(); \
    } while (0)
#else
#define MNN_HEXAGON_RECORD_COPY_BUFFER() \
    do {                                 \
    } while (0)
#define MNN_HEXAGON_PROFILE_FLUSH_COMMAND() flushCommand()
#endif
    if (srcType == MNN_FORWARD_HEXAGON && dstType == MNN_FORWARD_HEXAGON) {
        auto srcDev = getDevicePtr(srcTensor);
        auto dstDev = getDevicePtr(dstTensor);
        if (srcDev == dstDev) {
            MNN_HEXAGON_RECORD_COPY_BUFFER();
            return;
        }
        const size_t copySize = profileBytes;
        if (copySize == 0) {
            MNN_HEXAGON_RECORD_COPY_BUFFER();
            return;
        }
        int params[] = {1, 1, (int)copySize, 1, 2};
        HexagonCommand cmd;
        cmd.build(const_cast<HexagonBackend*>(this), DSP_OP_TENSOR_CONVERT, params, sizeof(params), {srcDev}, {dstDev});
        cmd.execute(true);
        MNN_HEXAGON_RECORD_COPY_BUFFER();
        return;
    }
    if (srcType == MNN_FORWARD_HEXAGON) {
        markHostOutput(srcTensor);
        MNN_HEXAGON_PROFILE_FLUSH_COMMAND();
    }
    if (dstType == MNN_FORWARD_HEXAGON && hasPendingHexagonWrite(dstTensor)) {
        MNN_HEXAGON_PROFILE_FLUSH_COMMAND();
    }
    uint8_t* srcHost = srcType == MNN_FORWARD_HEXAGON ? getPtr(srcTensor) : srcTensor->host<uint8_t>();
    uint8_t* dstHost = dstType == MNN_FORWARD_HEXAGON ? getPtr(dstTensor) : dstTensor->host<uint8_t>();
    if (!srcIsFloat || !dstIsFloat) {
        size_t bytes = srcTensor->elementSize() * srcTensor->getType().bytes();
        memcpy(dstHost, srcHost, bytes);
        if (dstType == MNN_FORWARD_HEXAGON) {
            markHostInput(dstTensor);
        }
        MNN_HEXAGON_RECORD_COPY_BUFFER();
        return;
    }

    // Check tensor backend type to determine data format
    // Hexagon device tensors are fp16, CPU tensors are fp32
    bool srcIsFp16 = (srcType == MNN_FORWARD_HEXAGON);
    bool dstIsFp16 = (dstType == MNN_FORWARD_HEXAGON);

    if (!srcIsFp16 && dstIsFp16) {
        // CPU (fp32) to Hexagon (fp16)
        if (dstIsNc4hw4) {
            if (srcIsNc4hw4) {
                nc4hw4HostToDevice((float*)srcHost, (int16_t*)dstHost, srcTensor, pack, HexagonBackend::fp32ToFp16);
            } else {
                nchwHostToDevice((float*)srcHost, (int16_t*)dstHost, srcTensor, pack, HexagonBackend::fp32ToFp16);
            }
        } else {
            size_t srcSize = srcTensor->elementSize();
            HexagonBackend::fp32ToFp16((float*)srcHost, (int16_t*)dstHost, srcSize);
        }
    } else if (srcIsFp16 && !dstIsFp16) {
        // Hexagon (fp16) to CPU (fp32)
        if (srcIsNc4hw4) {
            if (dstIsNc4hw4) {
                nc4hw4DeviceToHost((int16_t*)srcHost, (float*)dstHost, dstTensor, pack, HexagonBackend::fp16ToFp32);
            } else {
                deviceToNchwHost((int16_t*)srcHost, (float*)dstHost, dstTensor, pack, HexagonBackend::fp16ToFp32);
            }
        } else {
            size_t dstSize = dstTensor->elementSize();
            HexagonBackend::fp16ToFp32((int16_t*)srcHost, (float*)dstHost, dstSize);
        }
    } else {
        // Same format - just copy
        size_t elementCount = srcTensor->elementSize();
        size_t bytes = elementCount * srcTensor->getType().bytes();

        memcpy(dstHost, srcHost, bytes);
    }
    if (dstType == MNN_FORWARD_HEXAGON) {
        markHostInput(dstTensor);
    }
    MNN_HEXAGON_RECORD_COPY_BUFFER();
#undef MNN_HEXAGON_PROFILE_FLUSH_COMMAND
#undef MNN_HEXAGON_RECORD_COPY_BUFFER
}

int HexagonBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    if (toCpu) {
        markHostOutput(dstTensor);
        flushCommand();
    }
    return 0;
}
uint8_t* HexagonBackend::getPtr(const Tensor* tensor) {
    auto offset = TensorUtils::getDescribeOrigin(tensor)->offset;
    auto buffer = (HexagonBuffer*)(tensor->buffer().device);
    return (uint8_t*)buffer->ptr + offset;
}

uint8_t* HexagonBackend::getPtr(const MemChunk& chunk) {
    auto buffer = (HexagonBuffer*)chunk.first;
    return (uint8_t*)buffer->ptr + chunk.second;
}
std::pair<int, int> HexagonBackend::getDevicePtr(const Tensor* tensor) {
    if (tensor == nullptr) {
        return std::make_pair(0, 0);
    }
    auto device = tensor->buffer().device;
    if (device == 0) {
        return std::make_pair(0, 0);
    }
    auto offset = TensorUtils::getDescribeOrigin(tensor)->offset;
    auto buffer = (HexagonBuffer*)device;
    if (buffer == nullptr) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer->fd, offset);
}
std::pair<int, int> HexagonBackend::getDevicePtr(const MemChunk& chunk) {
    auto offset = chunk.second;
    auto buffer = (HexagonBuffer*)chunk.first;
    return std::make_pair(buffer->fd, offset);
}

void HexagonBackend::markHostInput(const Tensor* tensor) const {
    if (tensor == nullptr) {
        return;
    }
    auto dev = getDevicePtr(tensor);
    mRuntime->markHostInput(dev.first, dev.second, (int)getSize(tensor));
}

void HexagonBackend::markHostInput(const MemChunk& chunk, int size) const {
    auto dev = getDevicePtr(chunk);
    mRuntime->markHostInput(dev.first, dev.second, size);
}

void HexagonBackend::markHostOutput(const Tensor* tensor) const {
    if (tensor == nullptr) {
        return;
    }
    auto dev = getDevicePtr(tensor);
    mRuntime->markHostOutput(dev.first, dev.second, (int)getSize(tensor));
}

void HexagonBackend::markHostOutput(const MemChunk& chunk, int size) const {
    auto dev = getDevicePtr(chunk);
    mRuntime->markHostOutput(dev.first, dev.second, size);
}

void HexagonBackend::markHexagonOutput(const Tensor* tensor) const {
    if (tensor == nullptr) {
        return;
    }
    auto dev = getDevicePtr(tensor);
    mRuntime->markHexagonOutput(dev.first, dev.second, (int)getSize(tensor));
}

bool HexagonBackend::hasPendingHexagonWrite(const Tensor* tensor) const {
    if (tensor == nullptr) {
        return false;
    }
    auto dev = getDevicePtr(tensor);
    return mRuntime->hasPendingHexagonWrite(dev.first, dev.second, (int)getSize(tensor));
}

void HexagonBackend::markDynamicHostOutput() const {
    if (mRuntime->mDynamicBuffer.current.first == nullptr || mRuntime->mDynamicBuffer.currentSize == 0) {
        return;
    }
    markHostOutput(mRuntime->mDynamicBuffer.current, (int)mRuntime->mDynamicBuffer.currentSize);
}

MemChunk HexagonBackend::allocCommandSlot(int size) const {
    return mRuntime->allocCommandSlot(size);
}

void HexagonBackend::freeCommandSlot(const MemChunk& chunk) const {
    mRuntime->freeCommandSlot(chunk);
}

void HexagonBackend::pushCommand(const MemChunk& cmdChunk, int cmdSize, bool needCopy, bool dirty) const {
    mRuntime->pushCommand(cmdChunk, cmdSize, needCopy, dirty);
}

int HexagonBackend::commandSerial() const {
    return mRuntime->commandSerial();
}

void HexagonBackend::flushCommand() const {
    if (!mAllOpSupport) {
        markDynamicHostOutput();
    }
    mRuntime->flushCommand();
}
} // namespace MNN
