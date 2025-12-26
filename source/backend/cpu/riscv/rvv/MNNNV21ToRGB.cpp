#include <riscv_vector.h>
#include <algorithm>

void MNNNV21ToRGB(const unsigned char* source, unsigned char* dest, size_t count) {
    const unsigned char* y = source;
    const unsigned char* uv = source + count;
    size_t i = 0;
    
    while (i < count) {
        size_t vl = __riscv_vsetvl_e8m2(count - i);
        vl = vl & ~1UL;
        if (vl == 0) break;
        size_t vlHalf = vl / 2;
        vuint8m2_t dupIdx = __riscv_vsrl_vx_u8m2(__riscv_vid_v_u8m2(vl), 1, vl);
        
        vuint8m2_t channel8 = __riscv_vle8_v_u8m2(y + i, vl);
        vint16m4_t y16 = __riscv_vreinterpret_v_u16m4_i16m4(
            __riscv_vzext_vf2_u16m4(channel8, vl));
        
        vuint8m1_t half8 = __riscv_vlse8_v_u8m1(uv + (i / 2) * 2, 2, vlHalf);
        channel8 = __riscv_vrgather_vv_u8m2(
            __riscv_vlmul_ext_v_u8m1_u8m2(half8), dupIdx, vl);
        vint16m4_t v16 = __riscv_vsub_vx_i16m4(
            __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(channel8, vl)), 
            128, vl);
        
        half8 = __riscv_vlse8_v_u8m1(uv + (i / 2) * 2 + 1, 2, vlHalf);
        channel8 = __riscv_vrgather_vv_u8m2(
            __riscv_vlmul_ext_v_u8m1_u8m2(half8), dupIdx, vl);
        vint16m4_t u16 = __riscv_vsub_vx_i16m4(
            __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(channel8, vl)), 
            128, vl);
        
        vint32m8_t y32 = __riscv_vsll_vx_i32m8(
            __riscv_vwcvt_x_x_v_i32m8(y16, vl), 6, vl);
        vint32m8_t v32 = __riscv_vwcvt_x_x_v_i32m8(v16, vl);
        vint32m8_t u32 = __riscv_vwcvt_x_x_v_i32m8(u16, vl);
        
        vint32m8_t calc32 = __riscv_vmacc_vx_i32m8(y32, 73, v32, vl);
        calc32 = __riscv_vsra_vx_i32m8(calc32, 6, vl);
        calc32 = __riscv_vmax_vx_i32m8(calc32, 0, vl);
        calc32 = __riscv_vmin_vx_i32m8(calc32, 255, vl);
        vint16m4_t res16 = __riscv_vncvt_x_x_w_i16m4(calc32, vl);
        channel8 = __riscv_vncvt_x_x_w_u8m2(
            __riscv_vreinterpret_v_i16m4_u16m4(res16), vl);
        __riscv_vsse8_v_u8m2(dest + 3 * i + 0, 3, channel8, vl);
        
        calc32 = __riscv_vnmsac_vx_i32m8(y32, 25, u32, vl);
        calc32 = __riscv_vnmsac_vx_i32m8(calc32, 37, v32, vl);
        calc32 = __riscv_vsra_vx_i32m8(calc32, 6, vl);
        calc32 = __riscv_vmax_vx_i32m8(calc32, 0, vl);
        calc32 = __riscv_vmin_vx_i32m8(calc32, 255, vl);
        res16 = __riscv_vncvt_x_x_w_i16m4(calc32, vl);
        channel8 = __riscv_vncvt_x_x_w_u8m2(
            __riscv_vreinterpret_v_i16m4_u16m4(res16), vl);
        __riscv_vsse8_v_u8m2(dest + 3 * i + 1, 3, channel8, vl);
        
        calc32 = __riscv_vmacc_vx_i32m8(y32, 130, u32, vl);
        calc32 = __riscv_vsra_vx_i32m8(calc32, 6, vl);
        calc32 = __riscv_vmax_vx_i32m8(calc32, 0, vl);
        calc32 = __riscv_vmin_vx_i32m8(calc32, 255, vl);
        res16 = __riscv_vncvt_x_x_w_i16m4(calc32, vl);
        channel8 = __riscv_vncvt_x_x_w_u8m2(
            __riscv_vreinterpret_v_i16m4_u16m4(res16), vl);
        __riscv_vsse8_v_u8m2(dest + 3 * i + 2, 3, channel8, vl);
                
        i += vl;
    }
    
    for (; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;
        Y = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;
        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);
        dest[3 * i + 0] = (uint8_t)R;
        dest[3 * i + 1] = (uint8_t)G;
        dest[3 * i + 2] = (uint8_t)B;
    }
}
