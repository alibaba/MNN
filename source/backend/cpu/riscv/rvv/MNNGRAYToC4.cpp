#include <riscv_vector.h>

void MNNGRAYToC4(const unsigned char* source, unsigned char* dest, size_t count) {
    size_t i = 0;

    while (i < count) {
        size_t vl = __riscv_vsetvl_e8m8(count - i);
        vuint8m8_t gray = __riscv_vle8_v_u8m8(source + i, vl);
        vuint8m8_t alpha = __riscv_vmv_v_x_u8m8(255, vl);
        __riscv_vsse8_v_u8m8(dest + i * 4 + 0, 4, gray, vl);
        __riscv_vsse8_v_u8m8(dest + i * 4 + 1, 4, gray, vl);
        __riscv_vsse8_v_u8m8(dest + i * 4 + 2, 4, gray, vl);
        __riscv_vsse8_v_u8m8(dest + i * 4 + 3, 4, alpha, vl);
        i += vl;
    }
}
