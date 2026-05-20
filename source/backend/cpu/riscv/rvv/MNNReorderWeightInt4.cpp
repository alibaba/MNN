#include <riscv_vector.h>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef MNN_ASSERT
#define MNN_ASSERT(x)
#endif

static inline int32_t _rvvReduceAddI32(vint32m4_t value, size_t vl) {
    const size_t vl1 = __riscv_vsetvl_e32m1(1);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl1);
    const vint32m1_t sum = __riscv_vredsum_vs_i32m4_i32m1(value, zero, vl);
    return __riscv_vmv_x_s_i32m1_i32(sum);
}

void MNNReorderWeightInt4_RVV(uint8_t* dest, const uint8_t* source, int32_t* shape, size_t size, float* kernelsum) {
    MNN_ASSERT(size > 4);
    const int32_t blocknum = shape[0];
    const int32_t hu = shape[1];
    const int32_t lu = shape[2];
    const int32_t hp = shape[3];
    const int32_t lp = shape[4];
    const int32_t ic = blocknum * lu * lp;
    const int32_t stride0 = blocknum * hp * lu * lp;
    const int32_t stride1 = lu * hp * lp;
    const int32_t stride2 = hp * lp;

    // [oc,ic] -> [hu,blocknum,lu,hp,lp]
    for (int32_t i = 0; i < hu; ++i) {
        for (int32_t k = 0; k < hp; ++k) {
            for (int32_t bl = 0; bl < blocknum; ++bl) {
                for (int32_t j = 0; j < lu; ++j) {
                    const int32_t srcIndex = (i * hp + k) * ic + bl * (lu * lp) + j * lp;
                    const int32_t dstIndex = i * stride0 + bl * stride1 + j * stride2 + k * lp;
                    int32_t x = 0;
                    while (x < lp) {
                        const size_t vl = __riscv_vsetvl_e8m1(lp - x);
                        const vuint8m1_t v = __riscv_vle8_v_u8m1(source + srcIndex + x, vl);
                        __riscv_vse8_v_u8m1(dest + dstIndex + x, v, vl);
                        x += static_cast<int32_t>(vl);
                    }
                }
            }
        }
    }

    // [hu,blocknum,lu,hp,lp] address [hp,lp] for int4
    const int32_t inside = lp * hp;
    const int32_t outside = blocknum * hu;
    const int32_t half = inside / 2;
    std::vector<uint8_t> buffer(static_cast<size_t>(inside));

    for (int32_t i = 0; i < outside; ++i) {
        float* sumBase = kernelsum + i * hp;
        std::memset(sumBase, 0, static_cast<size_t>(hp) * sizeof(float));
        for (int32_t k = 0; k < lu; ++k) {
            uint8_t* dstBase = dest + (i * lu + k) * inside;
            int32_t j = 0;
            while (j < half) {
                const int32_t h0 = j / lp;
                const int32_t h1 = (j + half) / lp;

                int32_t chunk = half - j;
                const int32_t remain0 = lp - (j % lp);
                const int32_t remain1 = lp - ((j + half) % lp);
                if (chunk > remain0) {
                    chunk = remain0;
                }
                if (chunk > remain1) {
                    chunk = remain1;
                }

                int32_t p = 0;
                while (p < chunk) {
                    const int32_t offset = j + p;
                    const size_t vl = __riscv_vsetvl_e8m1(chunk - p);

                    const vuint8m1_t d0 = __riscv_vle8_v_u8m1(dstBase + offset, vl);
                    const vuint8m1_t d1 = __riscv_vle8_v_u8m1(dstBase + offset + half, vl);

                    const vuint8m1_t w0 = __riscv_vsrl_vx_u8m1(d0, 4, vl);
                    const vuint8m1_t w1 = __riscv_vand_vx_u8m1(d0, 0x0f, vl);
                    const vuint8m1_t w2 = __riscv_vsrl_vx_u8m1(d1, 4, vl);
                    const vuint8m1_t w3 = __riscv_vand_vx_u8m1(d1, 0x0f, vl);

                    const vuint8m1_t packed0 = __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(w0, 4, vl), w2, vl);
                    const vuint8m1_t packed1 = __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(w1, 4, vl), w3, vl);
                    __riscv_vsse8_v_u8m1(buffer.data() + 2 * offset + 0, 2, packed0, vl);
                    __riscv_vsse8_v_u8m1(buffer.data() + 2 * offset + 1, 2, packed1, vl);

                    const vint16m2_t w0_16 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(w0, vl));
                    const vint16m2_t w1_16 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(w1, vl));
                    const vint16m2_t w2_16 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(w2, vl));
                    const vint16m2_t w3_16 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(w3, vl));

                    const vint32m4_t sum0 = __riscv_vadd_vv_i32m4(__riscv_vwcvt_x_x_v_i32m4(w0_16, vl),
                                                                  __riscv_vwcvt_x_x_v_i32m4(w1_16, vl), vl);
                    const vint32m4_t sum1 = __riscv_vadd_vv_i32m4(__riscv_vwcvt_x_x_v_i32m4(w2_16, vl),
                                                                  __riscv_vwcvt_x_x_v_i32m4(w3_16, vl), vl);

                    sumBase[h0] += static_cast<float>(_rvvReduceAddI32(sum0, vl));
                    sumBase[h1] += static_cast<float>(_rvvReduceAddI32(sum1, vl));

                    p += static_cast<int32_t>(vl);
                }
                j += chunk;
            }
            std::memcpy(dstBase, buffer.data(), static_cast<size_t>(inside));
        }
    }
}
