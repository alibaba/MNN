#include <riscv_vector.h>

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const ptrdiff_t dstStrideBytes = 4 * (ptrdiff_t)sizeof(float);
    const size_t depthC4 = (depth + 3) / 4;

    for (size_t z = 0; z < depthC4; ++z) {
        const size_t cBase = z * 4;
        float* dstZ = dst + z * dstAreaStride * 4;
        size_t valid = depth - cBase;
        if (valid > 4)
            valid = 4;

        for (size_t y = 0; y < valid; ++y) {
            const float* srcChannel = src + (cBase + y) * srcAreaStride;

            size_t x = 0;
            while (x < area) {
                const size_t vl = __riscv_vsetvl_e32m8(area - x);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(srcChannel + x, vl);
                __riscv_vsse32_v_f32m8(dstZ + 4 * x + y, dstStrideBytes, v, vl);
                x += vl;
            }
        }

        for (size_t y = valid; y < 4; ++y) {
            size_t x = 0;
            while (x < area) {
                const size_t vl = __riscv_vsetvl_e32m8(area - x);
                vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                __riscv_vsse32_v_f32m8(dstZ + 4 * x + y, dstStrideBytes, zero, vl);
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
    if (area == 1 && areaOffset[0] == 1 && areaOffset[1] == 1) {
        for (size_t c = 0; c < depth;) {
            size_t vl = __riscv_vsetvl_e32m8(depth - c);
            __riscv_vse32_v_f32m8(dst + c, __riscv_vle32_v_f32m8(src + c, vl), vl);
            c += vl;
        }
        for (size_t c = depth; c % 4; ++c)
            dst[c] = 0.0f;
        return;
    }
    if (area == 1) {
        for (size_t c = 0; c < depth; ++c) {
            dst[(c / 4 * areaOffset[1] * 4 + c % 4)] = src[c * areaOffset[0]];
        }
        for (size_t c = depth; c % 4; ++c)
            dst[c / 4 * areaOffset[1] * 4 + c % 4] = 0.0f;
        return;
    }

    const int pack = _rvvChannelPack();
    const int depthC = static_cast<int>(depth / pack);
    const int remain = static_cast<int>(depth - static_cast<size_t>(depthC * pack));
    const int srcAreaOffset = areaOffset[0];
    const int dstAreaOffset = areaOffset[1];
    const ptrdiff_t dstStride = static_cast<ptrdiff_t>(pack * sizeof(float));
    for (int z = 0; z < depthC; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * pack;
        auto srcPlane = src + z * srcAreaOffset * pack;
        for (size_t x = 0; x < area;) {
            size_t vl = __riscv_vsetvl_e32m2(area - x);
            vfloat32m2x4_t data;
            data = __riscv_vset_v_f32m2_f32m2x4(data, 0, __riscv_vle32_v_f32m2(srcPlane + x, vl));
            data = __riscv_vset_v_f32m2_f32m2x4(data, 1, __riscv_vle32_v_f32m2(srcPlane + srcAreaOffset + x, vl));
            data = __riscv_vset_v_f32m2_f32m2x4(data, 2, __riscv_vle32_v_f32m2(srcPlane + 2 * srcAreaOffset + x, vl));
            data = __riscv_vset_v_f32m2_f32m2x4(data, 3, __riscv_vle32_v_f32m2(srcPlane + 3 * srcAreaOffset + x, vl));
            __riscv_vsseg4e32_v_f32m2x4(dstPlane + 4 * x, data, vl);
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
