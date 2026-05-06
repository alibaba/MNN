#include "../../compute/Int8FunctionsOpt.h"
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void _MNNPackC4Int8ForMatMul_ASparse_RVV(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info,
                                         const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];

    for (int n = 0; n < number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto dest = destOrigin + lOffset * eDest + eOffset;
        auto source = sourceGroup[n];

        for (int y = 0; y < e; ++y) {
            auto yR = y % eDest;
            size_t vl;
            int x = 0;
            for (size_t l_left = l; l_left > 0; l_left -= vl, x += vl) {
                // 【修复】以 8 位数据为基准设置向量长度，数据占 1 个寄存器 (m1)
                vl = __riscv_vsetvl_e8m1(l_left);

                uint32_t src_idx[vl];
                uint32_t dst_idx[vl];
                for (size_t i = 0; i < vl; ++i) {
                    int cur_x = x + i;
                    int xR = cur_x % 4;
                    int xC = cur_x / 4;
                    src_idx[i] = xC * eReal * 4 + y * 4 * offset + xR;
                    dst_idx[i] = cur_x * eDest + yR;
                }

                // 【修复】32 位索引必须使用 m4 寄存器组装载 (8位 * 4 = 32位)
                vuint32m4_t v_src_idx = __riscv_vle32_v_u32m4(src_idx, vl);
                vuint32m4_t v_dst_idx = __riscv_vle32_v_u32m4(dst_idx, vl);

                // 索引(m4) 寻找 数据(m1)
                vint8m1_t v_data = __riscv_vloxei32_v_i8m1(source, v_src_idx, vl);
                __riscv_vsoxei32_v_i8m1(dest, v_dst_idx, v_data, vl);
            }
        }
    }
}