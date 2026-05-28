#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "backend/cpu/CPURuntime.hpp"

static inline int _rvvChannelPack() {
    auto cpuInfo = MNNGetCPUInfo();
    if (nullptr == cpuInfo || cpuInfo->channel_pack <= 0) {
        return 4;
    }
    return cpuInfo->channel_pack;
}

template <typename T>
static void _packCUnit(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    const int pack = _rvvChannelPack();
    const int depthC = static_cast<int>(depth / pack);
    const int remain = static_cast<int>(depth - static_cast<size_t>(depthC * pack));
    const int srcAreaOffset = areaOffset[0];
    const int dstAreaOffset = areaOffset[1];
    for (int z = 0; z < depthC; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * pack;
        auto srcPlane = src + z * srcAreaOffset * pack;
        for (size_t x = 0; x < area; ++x) {
            auto dstX = dstPlane + x * pack;
            for (int y = 0; y < pack; ++y) {
                dstX[y] = srcPlane[y * srcAreaOffset + x];
            }
        }
    }
    if (remain > 0) {
        auto dstPlane = dst + depthC * dstAreaOffset * pack;
        auto srcPlane = src + depthC * srcAreaOffset * pack;
        for (size_t x = 0; x < area; ++x) {
            auto dstX = dstPlane + x * pack;
            for (int y = 0; y < remain; ++y) {
                dstX[y] = srcPlane[y * srcAreaOffset + x];
            }
            for (int y = remain; y < pack; ++y) {
                dstX[y] = 0;
            }
        }
    }
}

static void _packCUnitFloat(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const int pack = _rvvChannelPack();
    const int depthC = static_cast<int>(depth / pack);
    const int remain = static_cast<int>(depth - static_cast<size_t>(depthC * pack));
    const int srcAreaOffset = areaOffset[0];
    const int dstAreaOffset = areaOffset[1];
    const ptrdiff_t dstStride = static_cast<ptrdiff_t>(pack * sizeof(float));
    for (int z = 0; z < depthC; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * pack;
        auto srcPlane = src + z * srcAreaOffset * pack;
        size_t x = 0;
        while (x < area) {
            size_t vl = __riscv_vsetvl_e32m8(area - x);
            auto dstX = dstPlane + x * pack;
            for (int y = 0; y < pack; ++y) {
                auto value = __riscv_vle32_v_f32m8(srcPlane + y * srcAreaOffset + x, vl);
                __riscv_vsse32_v_f32m8(dstX + y, dstStride, value, vl);
            }
            x += vl;
        }
    }
    if (remain > 0) {
        auto dstPlane = dst + depthC * dstAreaOffset * pack;
        auto srcPlane = src + depthC * srcAreaOffset * pack;
        size_t x = 0;
        while (x < area) {
            size_t vl = __riscv_vsetvl_e32m8(area - x);
            auto dstX = dstPlane + x * pack;
            for (int y = 0; y < remain; ++y) {
                auto value = __riscv_vle32_v_f32m8(srcPlane + y * srcAreaOffset + x, vl);
                __riscv_vsse32_v_f32m8(dstX + y, dstStride, value, vl);
            }
            auto zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int y = remain; y < pack; ++y) {
                __riscv_vsse32_v_f32m8(dstX + y, dstStride, zero, vl);
            }
            x += vl;
        }
    }
}

template <typename T>
static void _packCUnitTranspose(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    const int pack = _rvvChannelPack();
    const int c = static_cast<int>(depth);
    const int cDiv = c / pack;
    const int cAlign = cDiv * pack;
    const int dstAreaOffset = areaOffset[1];
    for (size_t hi = 0; hi < area; ++hi) {
        const T* srcHeight = src + hi * c;
        T* dstHeight = dst + hi * pack;
        for (int ci = 0; ci < cDiv; ++ci) {
            memcpy(dstHeight + ci * dstAreaOffset * pack, srcHeight + ci * pack, pack * sizeof(T));
        }
    }
    if (cAlign == c) {
        return;
    }
    const int cRemain = c - cAlign;
    const T* srcAlign = src + cAlign;
    T* dstAlign = dst + dstAreaOffset * cAlign;
    for (size_t hi = 0; hi < area; ++hi) {
        const T* srcHeight = srcAlign + hi * c;
        T* dstHeight = dstAlign + hi * pack;
        memset(dstHeight, 0, pack * sizeof(T));
        memcpy(dstHeight, srcHeight, cRemain * sizeof(T));
    }
}

void MNNPackCUnit_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    _packCUnitFloat(dst, src, area, depth, areaOffset);
}

void MNNPackCUnitInt8_RVV(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    _packCUnit(dst, src, area, depth, areaOffset);
}

void MNNPackCUnitInt16_RVV(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset) {
    _packCUnit(dst, src, area, depth, areaOffset);
}

void MNNPackCUnitTranspose_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    _packCUnitTranspose(dst, src, area, depth, areaOffset);
}

void MNNPackCUnitTransposeInt8_RVV(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    _packCUnitTranspose(dst, src, area, depth, areaOffset);
}

void MNNPackCUnitTransposeInt16_RVV(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset) {
    _packCUnitTranspose(dst, src, area, depth, areaOffset);
}
