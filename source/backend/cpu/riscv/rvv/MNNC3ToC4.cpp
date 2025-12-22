#include <riscv_vector.h>

void MNNC3ToC4(const unsigned char* source, unsigned char* dest, size_t count) {
    size_t i = 0;
    while (i < count) {
        size_t vl = __riscv_vsetvl_e8m8(count - i);
        vuint8m8_t channel = __riscv_vlse8_v_u8m8(source + 3 * i + 0, 3, vl);
        __riscv_vsse8_v_u8m8(dest + 4 * i + 0, 4, channel, vl);

        channel = __riscv_vlse8_v_u8m8(source + 3 * i + 1, 3, vl);
        __riscv_vsse8_v_u8m8(dest + 4 * i + 1, 4, channel, vl);

        channel = __riscv_vlse8_v_u8m8(source + 3 * i + 2, 3, vl);
        __riscv_vsse8_v_u8m8(dest + 4 * i + 2, 4, channel, vl);

        vuint8m8_t alpha = __riscv_vmv_v_x_u8m8(255, vl);
        __riscv_vsse8_v_u8m8(dest + 4 * i + 3, 4, alpha, vl);
        i += vl;
    }
}
