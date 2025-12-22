#include <riscv_vector.h>

void MNNBilinearLineC8(int8_t* dst, const int16_t* A, const int16_t* B, 
                           const float* t, int8_t* zeroPoint, size_t number) {
    int offset = *zeroPoint;
    int8_t* dstPtr = dst;
    
    const int pack = 8;
    const int16_t df = (int16_t)((*t) * 128.0f);
    const int16_t sf = (int16_t)((1.0f - *t) * 128.0f);
    const size_t total = number * pack;
    const int32_t ROUND_HALF = 1 << 13;
    
    size_t vl;
    for (size_t i = 0; i < total; i += vl) {
        vl = __riscv_vsetvl_e16m4(total - i);
        vint16m4_t v16 = __riscv_vle16_v_i16m4(A + i, vl);
        vint32m8_t v32 = __riscv_vwmul_vx_i32m8(v16, sf, vl);
        v16 = __riscv_vle16_v_i16m4(B + i, vl);
        v32 = __riscv_vwmacc_vx_i32m8(v32, df, v16, vl);
        
        vbool4_t mask = __riscv_vmslt_vx_i32m8_b4(v32, 0, vl);
        vint32m8_t tmp = __riscv_vadd_vx_i32m8(v32, ROUND_HALF, vl);
        v32 = __riscv_vsub_vx_i32m8(v32, ROUND_HALF, vl);
        v32 = __riscv_vmerge_vvm_i32m8(tmp, v32, mask, vl);
        
        tmp = __riscv_vsra_vx_i32m8(v32, 14, vl);
                mask = __riscv_vmslt_vx_i32m8_b4(v32, 0, vl);
        v32 = __riscv_vand_vx_i32m8(v32, 0x3FFF, vl);
        vbool4_t hasRem = __riscv_vmsne_vx_i32m8_b4(v32, 0, vl);
        mask = __riscv_vmand_mm_b4(mask, hasRem, vl);
        
        v32 = __riscv_vadd_vx_i32m8_mu(mask, tmp, tmp, 1, vl);
                v32 = __riscv_vadd_vx_i32m8(v32, offset, vl);
                v16 = __riscv_vnsra_wx_i16m4(v32, 0, vl);
        vint8m2_t v8 = __riscv_vnsra_wx_i8m2(v16, 0, vl);
        
        __riscv_vse8_v_i8m2(dstPtr + i, v8, vl);
    }
}
