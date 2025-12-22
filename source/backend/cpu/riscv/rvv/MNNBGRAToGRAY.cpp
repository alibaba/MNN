#include <riscv_vector.h>

void MNNBGRAToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    size_t i = 0;
    while (i < count) {
        size_t vl = __riscv_vsetvl_e8m4(count - i);
        vuint8m4_t channel = __riscv_vlse8_v_u8m4(source + 4 * i + 2, 4, vl);
        vuint16m8_t sum = __riscv_vwmulu_vx_u16m8(channel, 19, vl);
        
        channel = __riscv_vlse8_v_u8m4(source + 4 * i + 1, 4, vl);
        sum = __riscv_vwmaccu_vx_u16m8(sum, 38, channel, vl);
        
        channel = __riscv_vlse8_v_u8m4(source + 4 * i + 0, 4, vl);
        sum = __riscv_vwmaccu_vx_u16m8(sum, 7, channel, vl);
        
        channel = __riscv_vnsrl_wx_u8m4(sum, 6, vl);
        __riscv_vse8_v_u8m4(dest + i, channel, vl);
        i += vl;
    }
}
