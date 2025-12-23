#include <riscv_vector.h>

void MNNC3ToBGR565(const unsigned char* source, unsigned char* dest, 
                       size_t count, bool bgr) {
    unsigned short* dest16 = reinterpret_cast<unsigned short*>(dest);
    size_t i = 0;
    int rOffset = bgr ? 2 : 0;
    int bOffset = bgr ? 0 : 2;
    
    while (i < count) {
        size_t vl = __riscv_vsetvl_e8m4(count - i);
        
        vuint8m4_t channel = __riscv_vlse8_v_u8m4(source + 3 * i + bOffset, 3, vl);
        vuint8m4_t shifted = __riscv_vsrl_vx_u8m4(channel, 3, vl);
        vuint16m8_t result = __riscv_vzext_vf2_u16m8(shifted, vl);
        
        channel = __riscv_vlse8_v_u8m4(source + 3 * i + 1, 3, vl);
        vuint8m4_t masked = __riscv_vand_vx_u8m4(channel, 0xFC, vl);
        vuint16m8_t wide = __riscv_vzext_vf2_u16m8(masked, vl);
        wide = __riscv_vsll_vx_u16m8(wide, 3, vl);
        result = __riscv_vor_vv_u16m8(result, wide, vl);
        
        channel = __riscv_vlse8_v_u8m4(source + 3 * i + rOffset, 3, vl);
        masked = __riscv_vand_vx_u8m4(channel, 0xF8, vl);
        wide = __riscv_vzext_vf2_u16m8(masked, vl);
        wide = __riscv_vsll_vx_u16m8(wide, 8, vl);
        result = __riscv_vor_vv_u16m8(result, wide, vl);
        
        __riscv_vse16_v_u16m8(dest16 + i, result, vl);
        i += vl;
    }
}
