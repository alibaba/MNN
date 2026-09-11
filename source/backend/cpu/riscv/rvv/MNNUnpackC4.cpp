#include <riscv_vector.h>

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const ptrdiff_t srcStrideBytes = 4 * (ptrdiff_t)sizeof(float);
    const size_t depthC4 = (depth + 3) / 4;

    for (size_t z = 0; z < depthC4; ++z) {
        const size_t cBase = z * 4;
        const float* srcZ = src + z * srcAreaStride * 4;
        size_t valid = depth - cBase;
        if (valid > 4) {
            valid = 4;
        }

        for (size_t y = 0; y < valid; ++y) {
            const float* srcChannel = srcZ + y;
            float* dstChannel = dst + (cBase + y) * dstAreaStride;

            size_t x = 0;
            while (x < area) {
                const size_t vl = __riscv_vsetvl_e32m8(area - x);

                vfloat32m8_t v = __riscv_vlse32_v_f32m8(srcChannel + 4 * x, srcStrideBytes, vl);
                __riscv_vse32_v_f32m8(dstChannel + x, v, vl);

                x += vl;
            }
        }
    }
}

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
// C4 is the tensor ABI, independent of the hardware vector register width.
static constexpr int _rvvChannelPack() {
    return 4;
}

template <typename T>
static void _unpackCUnit(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    const int pack = _rvvChannelPack();
    const int depthC = static_cast<int>(depth / pack);
    const int remain = static_cast<int>(depth - static_cast<size_t>(depthC * pack));
    const int srcAreaOffset = areaOffset[0];
    const int dstAreaOffset = areaOffset[1];
    for (int z = 0; z < depthC; ++z) {
        const T* srcPlane = src + z * srcAreaOffset * pack;
        T* dstPlane = dst + z * dstAreaOffset * pack;
        for (int y = 0; y < pack; ++y) {
            auto dstChannel = dstPlane + y * dstAreaOffset;
            auto srcChannel = srcPlane + y;
            for (size_t x = 0; x < area; ++x) {
                dstChannel[x] = srcChannel[x * pack];
            }
        }
    }
    if (remain > 0) {
        const T* srcPlane = src + depthC * srcAreaOffset * pack;
        T* dstPlane = dst + depthC * dstAreaOffset * pack;
        for (int y = 0; y < remain; ++y) {
            auto dstChannel = dstPlane + y * dstAreaOffset;
            auto srcChannel = srcPlane + y;
            for (size_t x = 0; x < area; ++x) {
                dstChannel[x] = srcChannel[x * pack];
            }
        }
    }
}

static void _unpackCUnitFloat(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    if (area == 1) {
        for (size_t c = 0; c < depth; ++c) {
            dst[c * areaOffset[1]] = src[(c / 4 * areaOffset[0] * 4 + c % 4)];
        }
        return;
    }

    const int pack = _rvvChannelPack();
    const int depthC = static_cast<int>(depth / pack);
    const int remain = static_cast<int>(depth - static_cast<size_t>(depthC * pack));
    const int srcAreaOffset = areaOffset[0];
    const int dstAreaOffset = areaOffset[1];
    const ptrdiff_t srcStride = static_cast<ptrdiff_t>(pack * sizeof(float));
    for (int z = 0; z < depthC; ++z) {
        const float* srcPlane = src + z * srcAreaOffset * pack;
        float* dstPlane = dst + z * dstAreaOffset * pack;
        for (int y = 0; y < pack; ++y) {
            size_t x = 0;
            auto dstChannel = dstPlane + y * dstAreaOffset;
            while (x < area) {
                size_t vl = __riscv_vsetvl_e32m8(area - x);
                auto value = __riscv_vlse32_v_f32m8(srcPlane + x * pack + y, srcStride, vl);
                __riscv_vse32_v_f32m8(dstChannel + x, value, vl);
                x += vl;
            }
        }
    }
    if (remain > 0) {
        const float* srcPlane = src + depthC * srcAreaOffset * pack;
        float* dstPlane = dst + depthC * dstAreaOffset * pack;
        for (int y = 0; y < remain; ++y) {
            size_t x = 0;
            auto dstChannel = dstPlane + y * dstAreaOffset;
            while (x < area) {
                size_t vl = __riscv_vsetvl_e32m8(area - x);
                auto value = __riscv_vlse32_v_f32m8(srcPlane + x * pack + y, srcStride, vl);
                __riscv_vse32_v_f32m8(dstChannel + x, value, vl);
                x += vl;
            }
        }
    }
}

template <typename T>
static void _unpackCUnitTranspose(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    const int pack = _rvvChannelPack();
    const int c = static_cast<int>(depth);
    const int cDiv = c / pack;
    const int cAlign = cDiv * pack;
    const int srcAreaOffset = areaOffset[0];
    const int dstDepthOffset = areaOffset[1];
    for (size_t hi = 0; hi < area; ++hi) {
        const T* srcHeight = src + hi * pack;
        T* dstHeight = dst + hi * dstDepthOffset;
        for (int ci = 0; ci < cDiv; ++ci) {
            memcpy(dstHeight + ci * pack, srcHeight + ci * srcAreaOffset * pack, pack * sizeof(T));
        }
    }
    if (cAlign == c) {
        return;
    }
    const int cRemain = c - cAlign;
    const T* srcAlign = src + srcAreaOffset * cAlign;
    T* dstAlign = dst + cAlign;
    for (size_t hi = 0; hi < area; ++hi) {
        const T* srcHeight = srcAlign + hi * pack;
        T* dstHeight = dstAlign + hi * dstDepthOffset;
        memcpy(dstHeight, srcHeight, cRemain * sizeof(T));
    }
}

void MNNUnpackCUnit_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    _unpackCUnitFloat(dst, src, area, depth, areaOffset);
}

void MNNUnpackCUnitInt8_RVV(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    _unpackCUnit(dst, src, area, depth, areaOffset);
}

void MNNUnpackCUnitInt16_RVV(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset) {
    _unpackCUnit(dst, src, area, depth, areaOffset);
}

void MNNUnpackCUnitTranspose_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    _unpackCUnitTranspose(dst, src, area, depth, areaOffset);
}

void MNNUnpackCUnitTransposeInt8_RVV(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    _unpackCUnitTranspose(dst, src, area, depth, areaOffset);
}

void MNNUnpackCUnitTransposeInt16_RVV(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset) {
    _unpackCUnitTranspose(dst, src, area, depth, areaOffset);
}
