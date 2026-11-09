#include "dsp/int8_cnn.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __hexagon__
#include "dsp/hvx_utils.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static AEEResult requantize_i8_per_channel_32_scalar(
    uint8_t* dst, const int32_t* src,
    const DSPRequantParam* params) {
    if (dst == NULL || src == NULL || params == NULL) {
        return AEE_EBADPARM;
    }
    for (int lane = 0; lane < 32; ++lane) {
        const AEEResult status = requantize_i8_reference(
            dst + lane, src + lane, 1, params + lane);
        if (status != AEE_SUCCESS) {
            return status;
        }
    }
    return AEE_SUCCESS;
}

#ifdef __hexagon__

static HVX_Vector multiply_q31_round_sat(
    HVX_Vector value, HVX_Vector multiplier) {
    return Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(
        Q6_Vw_vmpye_VwVuh(value, multiplier), value, multiplier);
}

AEEResult htp_ops_prepare_requantize_i8_per_channel_32_hvx(
    DSPRequantTile32Hvx* tile, const DSPRequantParam* params) {
    if (tile == NULL || params == NULL) {
        return AEE_EBADPARM;
    }
    int32_t multipliers[32] __attribute__((aligned(128)));
    int32_t rightShifts[32] __attribute__((aligned(128)));
    int32_t outputZeros[32] __attribute__((aligned(128)));
    int32_t clampMins[32] __attribute__((aligned(128)));
    int32_t clampMaxs[32] __attribute__((aligned(128)));
    tile->useScalar = 0;
    for (int lane = 0; lane < 32; ++lane) {
        const DSPRequantParam* param = params + lane;
        if (param->clampMin < -127 || param->clampMax > 127 ||
            param->clampMin > param->clampMax) {
            return AEE_EBADPARM;
        }
        if (param->multiplier < 0 || param->shift > 0 ||
            param->shift < -31 || param->outputZero < -127 ||
            param->outputZero > 127) {
            tile->useScalar = 1;
        }
        multipliers[lane] = param->multiplier;
        rightShifts[lane] = -param->shift;
        outputZeros[lane] = param->outputZero;
        clampMins[lane] = param->clampMin;
        clampMaxs[lane] = param->clampMax;
    }
    tile->multiplier = vmem(multipliers);
    tile->rightShift = vmem(rightShifts);
    tile->outputZero = vmem(outputZeros);
    tile->clampMin = vmem(clampMins);
    tile->clampMax = vmem(clampMaxs);
    return AEE_SUCCESS;
}

AEEResult htp_ops_requantize_i8_per_channel_32_prepared_hvx(
    uint8_t* dst, HVX_Vector src, const DSPRequantParam* params,
    const DSPRequantTile32Hvx* tile) {
    if (dst == NULL || params == NULL || tile == NULL) {
        return AEE_EBADPARM;
    }
    if (tile->useScalar) {
        int32_t lanes[32] __attribute__((aligned(128)));
        vmem(lanes) = src;
        return requantize_i8_per_channel_32_scalar(
            dst, lanes, params);
    }

    const HVX_Vector shift = tile->rightShift;
    HVX_Vector scaled =
        multiply_q31_round_sat(src, tile->multiplier);
    const HVX_Vector allBits = Q6_V_vsplat_R(-1);
    const HVX_Vector one = Q6_V_vsplat_R(1);
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector mask =
        Q6_V_vnot_V(Q6_Vw_vasl_VwVw(allBits, shift));
    const HVX_Vector remainder = Q6_V_vand_VV(scaled, mask);
    const HVX_Vector half =
        Q6_Vw_vlsr_VwVw(Q6_Vw_vadd_VwVw(mask, one), one);
    const HVX_VectorPred exactTie = Q6_Q_and_QQ(
        Q6_Q_vcmp_eq_VwVw(remainder, half),
        Q6_Q_vcmp_gt_VwVw(shift, zero));
    HVX_Vector threshold = Q6_Vw_vlsr_VwVw(mask, one);
    threshold = Q6_Vw_condacc_QVwVw(
        Q6_Q_vcmp_gt_VwVw(zero, scaled), threshold, one);
    scaled = Q6_Vw_vasr_VwVw(scaled, shift);
    scaled = Q6_Vw_condacc_QVwVw(
        Q6_Q_vcmp_gt_VwVw(remainder, threshold), scaled, one);

    const HVX_Vector outputZero = tile->outputZero;
    scaled = Q6_Vw_vmax_VwVw(
        scaled, Q6_Vw_vsub_VwVw(tile->clampMin, outputZero));
    scaled = Q6_Vw_vmin_VwVw(
        scaled, Q6_Vw_vsub_VwVw(tile->clampMax, outputZero));
    scaled = Q6_Vw_vadd_VwVw(scaled, outputZero);

    const HVX_Vector packedHalf =
        Q6_Vh_vpack_VwVw_sat(scaled, scaled);
    const HVX_Vector bytes =
        Q6_Vb_vpack_VhVh_sat(packedHalf, packedHalf);
    vstu_variable(dst, 32, bytes);

    HVX_Vector tieLanes = Q6_V_vmux_QVV(exactTie, one, zero);
    HVX_Vector tieReduction =
        Q6_V_vor_VV(tieLanes, Q6_V_vror_VR(tieLanes, 64));
    tieReduction = Q6_V_vor_VV(
        tieReduction, Q6_V_vror_VR(tieReduction, 32));
    tieReduction = Q6_V_vor_VV(
        tieReduction, Q6_V_vror_VR(tieReduction, 16));
    tieReduction = Q6_V_vor_VV(
        tieReduction, Q6_V_vror_VR(tieReduction, 8));
    tieReduction = Q6_V_vor_VV(
        tieReduction, Q6_V_vror_VR(tieReduction, 4));
    if (Q6_R_vextract_VR(tieReduction, 0) != 0) {
        int32_t sourceLanes[32] __attribute__((aligned(128)));
        int32_t tieFlags[32] __attribute__((aligned(128)));
        vmem(sourceLanes) = src;
        vmem(tieFlags) = tieLanes;
        for (int lane = 0; lane < 32; ++lane) {
            if (tieFlags[lane] != 0) {
                const AEEResult status = requantize_i8_reference(
                    dst + lane, sourceLanes + lane, 1, params + lane);
                if (status != AEE_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return AEE_SUCCESS;
}

#endif

AEEResult htp_ops_requantize_i8_per_channel_32(
    uint8_t* dst, const int32_t* src,
    const DSPRequantParam* params) {
#ifdef __hexagon__
    if (src == NULL) {
        return AEE_EBADPARM;
    }
    DSPRequantTile32Hvx tile;
    const AEEResult status =
        htp_ops_prepare_requantize_i8_per_channel_32_hvx(
            &tile, params);
    if (status != AEE_SUCCESS) {
        return status;
    }
    return htp_ops_requantize_i8_per_channel_32_prepared_hvx(
        dst, vmemu(src), params, &tile);
#else
    return requantize_i8_per_channel_32_scalar(dst, src, params);
#endif
}

#ifdef __cplusplus
} // extern "C"
#endif
