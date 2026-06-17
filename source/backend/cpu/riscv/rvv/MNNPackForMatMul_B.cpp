#include <riscv_vector.h>
#include <cstring>
#include <stdint.h>
#include <stddef.h>

#define RVV_MATMUL_LP 1
#define RVV_MATMUL_HP 4

void MNNPackForMatMul_B_RVV(float* destC, const float* sourceC, size_t h, size_t kernelsize, size_t ic,
                            bool transpose) {
    auto dest = (int32_t*)destC;
    auto source = (int32_t*)sourceC;

    int LP = RVV_MATMUL_LP;
    int HP = RVV_MATMUL_HP;
    auto l = kernelsize * ic;

    size_t dest_size = ROUND_UP(h, HP) * ROUND_UP(ic, LP) * kernelsize * 4;
    memset(dest, 0, dest_size);

    auto stride0 = kernelsize * ROUND_UP(ic, LP) * HP;
    auto stride1 = ROUND_UP(ic, LP) * HP;
    auto stride2 = HP * LP;

    auto srcStride0 = l;
    auto srcStride1 = 1;
    if (!transpose) {
        srcStride0 = 1;
        srcStride1 = h;
    }

    size_t h_blocks = ROUND_UP(h, HP) / HP;

    for (size_t yHu = 0; yHu < h_blocks; ++yHu) {
        size_t y_start = yHu * HP;
        size_t y_end = (y_start + HP < h) ? (y_start + HP) : h;
        size_t y_len = y_end - y_start;

        if (y_len == 0)
            break;

        for (size_t k = 0; k < kernelsize; ++k) {
            for (size_t x = 0; x < ic; ++x) {
                auto xLu = x / LP;
                auto xLp = x % LP;

                int32_t* dst_ptr = dest + yHu * stride0 + k * stride1 + xLu * stride2 + xLp;

                size_t l_idx = x + k * ic;
                const int32_t* src_ptr = source + y_start * srcStride0 + l_idx * srcStride1;

                size_t vl;
                for (size_t yHp = 0; yHp < y_len; yHp += vl) {
                    vl = __riscv_vsetvl_e32m1(y_len - yHp);

                    vint32m1_t v_src;
                    if (!transpose) {
                        v_src = __riscv_vle32_v_i32m1(src_ptr + yHp, vl);
                    } else {
                        v_src = __riscv_vlse32_v_i32m1(src_ptr + yHp * l, l * sizeof(int32_t), vl);
                    }

                    __riscv_vse32_v_i32m1(dst_ptr + yHp, v_src, vl);
                }
            }
        }
    }
}
