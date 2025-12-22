#include <riscv_vector.h>

void MNNGRAYToC3(const unsigned char* source, unsigned char* dest, size_t count) {
    size_t i = 0;
    while (i < count) {
        size_t vl = __riscv_vsetvl_e8m8(count - i);
        vuint8m8_t gray = __riscv_vle8_v_u8m8(source + i, vl);
        __riscv_vsse8_v_u8m8(dest + i * 3 + 0, 3, gray, vl);
        __riscv_vsse8_v_u8m8(dest + i * 3 + 1, 3, gray, vl);
        __riscv_vsse8_v_u8m8(dest + i * 3 + 2, 3, gray, vl);
        i += vl;
    }
}
