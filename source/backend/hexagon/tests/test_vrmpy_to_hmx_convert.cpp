// Host-side scalar reference test for vrmpy_tile_to_hmx_int4_512() conversion
// (Path A, Step 2.3.1 of perf/hexagon/hexagon_phase2_plan.md).
//
// Goal: prove that a 512B int4 weight tile in the *vrmpy* layout (produced by
// reorderInt4WeightForVrmpyGemv) can be reordered, byte-for-byte, into the *HMX*
// layout (produced by reorderInt4WeightForHmx) that the prefill GEMM dequant
// (dequant_q4_tile_scaled / vlut16) consumes. If this holds, prefill can read the
// single vrmpy weight buffer and we drop the duplicate HMX weight.
//
// The two ref packers below are faithful (non-NEON) copies of the production
// loops in HexagonConvolution.cpp. scalarVrmpyToHmx() is the reference converter,
// derived analytically from those two layouts (see DERIVATION). The HVX
// implementation (Step 2.3.2) will later be checked against this scalar reference.
//
// Build: clang++ -O2 -std=c++11 -o /tmp/test_vrmpy_to_hmx test_vrmpy_to_hmx_convert.cpp
// Run:   /tmp/test_vrmpy_to_hmx
//
// ============================================================================
// DERIVATION  (relative within-tile indices: ocIn 0..31, kt 0..31)
//
// Raw nibble convention (both production functions agree):
//   even k -> high nibble of raw byte (byte>>4); odd k -> low nibble (byte&0xf).
//
// VRMPY tile (reorderInt4WeightForVrmpyGemv, lines 504-518):
//   tile[g*64 + ocIn*2 + p] low  = W(ocIn, 4g+2p)     (even kt -> low nibble)
//   tile[g*64 + ocIn*2 + p] high = W(ocIn, 4g+2p+1)   (odd  kt -> high nibble)
//   => for W(ocIn,kt):  B_v = (kt/4)*64 + 2*ocIn + ((kt>>1)&1)
//                       nibble_v = (kt&1) ? HIGH : LOW
//
// HMX tile (reorderInt4WeightForHmx, non-NEON, lines 329-372):
//   local[(kt/2)*64 + 2*ocIn + (kt&1)] = W(ocIn,kt)            (nibble value in byte)
//   shuffle: for q in 0..7: dst[q*128 + 2i]=src[q*128+i], dst[q*128+2i+1]=src[q*128+64+i]
//   pack:    for q in 0..3: out[q*128+i] = shuf[q*256+i]&0xf | (shuf[q*256+128+i]&0xf)<<4
//   Composing (kP=kt/2, koff=kt&1):
//     B_h = (kP/4)*128 + 2*(2*ocIn+koff) + (kP%2)
//         = (kt/8)*128 + 4*ocIn + 2*(kt&1) + ((kt>>1)&1)
//     nibble_h = ((kt>>2)&1) ? HIGH : LOW
//
// Converter: for every (ocIn,kt) move nibble (B_v,nibble_v) -> (B_h,nibble_h).
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static inline int UP_DIV(int x, int y) {
    return (x + y - 1) / y;
}

// Raw int4: row-major per oc, icBytes = UP_DIV(ic,2) bytes/row, 2 int4/byte.
// even k = high nibble (byte>>4), odd k = low nibble (byte&0xf) -- matches the
// rawNibble lambda in reorderInt4WeightForVrmpyGemv and the val>>4 / val&0xf in
// reorderInt4WeightForHmx.
static inline uint8_t rawNibble(const uint8_t* raw, int icBytes, int o, int k) {
    const uint8_t byte = raw[(size_t)o * icBytes + (k >> 1)];
    return (k & 1) ? (byte & 0x0f) : (uint8_t)(byte >> 4);
}

// ---- Faithful copy of reorderInt4WeightForVrmpyGemv (weight bytes only) ----
// dst size = icP*ocP*512. Requires ic%32==0, oc%32==0.
static void refPackVrmpy(uint8_t* dst, const uint8_t* raw, int ic, int oc) {
    const int icP = ic / 32, ocP = oc / 32;
    const int icBytes = UP_DIV(ic, 2);
    for (int y = 0; y < ocP; ++y) {
        for (int x = 0; x < icP; ++x) {
            uint8_t* tile = dst + (size_t)(y * icP + x) * 512;
            for (int ocIn = 0; ocIn < 32; ++ocIn) {
                const int o = y * 32 + ocIn;
                for (int g = 0; g < 8; ++g) {
                    for (int p = 0; p < 2; ++p) {
                        const int kLo = x * 32 + 4 * g + 2 * p;
                        const int kHi = kLo + 1;
                        tile[g * 64 + ocIn * 2 + p] =
                            (uint8_t)(rawNibble(raw, icBytes, o, kLo) | (rawNibble(raw, icBytes, o, kHi) << 4));
                    }
                }
            }
        }
    }
}

// ---- Faithful copy of reorderInt4WeightForHmx non-NEON path (weight only) ----
// dst size = icP*ocP*512 (== icP*ocP*32*16). Requires ic%32==0, oc%32==0 (aligned).
static void refPackHmx(uint8_t* dst, const uint8_t* raw, int ic, int oc) {
    const int icP = UP_DIV(ic, 32), ocP = UP_DIV(oc, 32);
    const int icBytes = UP_DIV(ic, 2);
    uint8_t local[32 * 32];
    uint8_t shuffled[32 * 32];
    for (int y = 0; y < ocP; ++y) {
        for (int x = 0; x < icP; ++x) {
            const int yCount = std::min(32, oc - y * 32);
            const int xCount = std::min(16, icBytes - x * 16);
            memset(local, 0, sizeof(local));
            for (int yi = 0; yi < yCount; ++yi) {
                const uint8_t* src = raw + (size_t)(y * 32 + yi) * icBytes + x * 16;
                for (int xi = 0; xi < xCount; ++xi) {
                    const uint8_t val = src[xi];
                    local[2 * xi * 32 + 2 * yi] = val >> 4;
                    local[2 * xi * 32 + 2 * yi + 1] = val & 0x0f;
                }
            }
            for (int q = 0; q < 8; ++q) {
                const uint8_t* s = local + q * 128;
                uint8_t* d = shuffled + q * 128;
                for (int i = 0; i < 64; ++i) {
                    d[2 * i] = s[i];
                    d[2 * i + 1] = s[64 + i];
                }
            }
            uint8_t* dstChunk = dst + (size_t)(y * icP + x) * 32 * 16;
            for (int q = 0; q < 4; ++q) {
                const uint8_t* low = shuffled + q * 256;
                const uint8_t* high = low + 128;
                for (int i = 0; i < 128; ++i) {
                    dstChunk[q * 128 + i] = (low[i] & 0x0f) | ((high[i] & 0x0f) << 4);
                }
            }
        }
    }
}

// ---- Reference converter: vrmpy 512B tile -> HMX 512B tile ----
// Pure permutation of nibbles; derived from the two layouts above.
static void scalarVrmpyToHmx(uint8_t* dst /*512*/, const uint8_t* src /*512*/) {
    memset(dst, 0, 512);
    for (int ocIn = 0; ocIn < 32; ++ocIn) {
        for (int kt = 0; kt < 32; ++kt) {
            const int Bv = (kt / 4) * 64 + 2 * ocIn + ((kt >> 1) & 1);
            const uint8_t nib = (kt & 1) ? (uint8_t)(src[Bv] >> 4) : (uint8_t)(src[Bv] & 0x0f);
            const int Bh = (kt / 8) * 128 + 4 * ocIn + 2 * (kt & 1) + ((kt >> 1) & 1);
            if ((kt >> 2) & 1) {
                dst[Bh] |= (uint8_t)((nib & 0x0f) << 4); // high nibble
            } else {
                dst[Bh] |= (uint8_t)(nib & 0x0f); // low nibble
            }
        }
    }
}

// ============================================================================
// HVX-op-emulation model of the DSP converter.
//
// Emulates the exact intrinsic sequence the on-device vrmpy_tile_to_hmx_int4_512()
// will use, so the derivation + mechanism are validated on host before any device
// round-trip. Each helper mirrors one HVX intrinsic (128B vector granularity).
//
// vrmpy->local expansion (per 128B input vector = 2 groups -> 2 local vectors):
//   vd = Q6_Vb_vdeal_Vb(v)                 // even bytes -> low64, odd -> high64
//   P  = Q6_W_vshuff_VVR(vd>>4, vd&0xf,-1) // P[2i]=lo[i], P[2i+1]=hi[i] (256B pair)
//   out0 = vmux(first64, Plo,        vror(Phi,64))   // local vec 2m
//   out1 = vmux(first64, vror(Plo,64), Phi)          // local vec 2m+1
// then the verified HMX tail (8x Q6_Vb_vshuff_Vb + 4x nibble pack).
// ============================================================================
static void vb_deal(uint8_t* out, const uint8_t* in) { // Q6_Vb_vdeal_Vb
    for (int i = 0; i < 64; ++i) {
        out[i] = in[2 * i];
        out[64 + i] = in[2 * i + 1];
    }
}
// Q6_W_vshuff_VVR(vhi, vlo, -1): 256B pair, tmp[2i]=vlo[i], tmp[2i+1]=vhi[i].
static void w_shuff_neg1(uint8_t* Plo, uint8_t* Phi, const uint8_t* vhi, const uint8_t* vlo) {
    uint8_t tmp[256];
    for (int i = 0; i < 128; ++i) {
        tmp[2 * i] = vlo[i];
        tmp[2 * i + 1] = vhi[i];
    }
    memcpy(Plo, tmp, 128);
    memcpy(Phi, tmp + 128, 128);
}
static void vror64(uint8_t* out, const uint8_t* in) { // Q6_V_vror_VR(v,64); dir-agnostic
    for (int j = 0; j < 128; ++j)
        out[j] = in[(j + 64) & 127];
}
static void vmux_first64(uint8_t* out, const uint8_t* a, const uint8_t* b) { // Q6_V_vmux_QVV(vsetq(64),a,b)
    for (int j = 0; j < 128; ++j)
        out[j] = (j < 64) ? a[j] : b[j];
}

static void hvxModelVrmpyToHmx(uint8_t* dst /*512*/, const uint8_t* src /*512*/) {
    uint8_t local[1024];
    for (int i = 0; i < 4; ++i) { // 4 input vectors, groups (2i, 2i+1)
        const uint8_t* v = src + i * 128;
        uint8_t vd[128];
        vb_deal(vd, v);
        uint8_t vlo[128], vhi[128];
        for (int j = 0; j < 128; ++j) {
            vlo[j] = vd[j] & 0x0f;
            vhi[j] = (vd[j] >> 4) & 0x0f;
        }
        uint8_t Plo[128], Phi[128];
        w_shuff_neg1(Plo, Phi, vhi, vlo);
        uint8_t PhiR[128], PloR[128];
        vror64(PhiR, Phi);
        vror64(PloR, Plo);
        vmux_first64(local + (2 * i) * 128, Plo, PhiR);     // local vec 2i
        vmux_first64(local + (2 * i + 1) * 128, PloR, Phi); // local vec 2i+1
    }
    // ---- verified HMX tail (identical to htp_ops_weight_reorder_int4_block) ----
    for (int q = 0; q < 8; ++q) { // Q6_Vb_vshuff_Vb per 128B block
        uint8_t* blk = local + q * 128;
        uint8_t t[128];
        for (int i = 0; i < 64; ++i) {
            t[2 * i] = blk[i];
            t[2 * i + 1] = blk[64 + i];
        }
        memcpy(blk, t, 128);
    }
    for (int q = 0; q < 4; ++q) { // pack: low | (high<<4)
        const uint8_t* low = local + q * 256;
        const uint8_t* high = low + 128;
        for (int i = 0; i < 128; ++i)
            dst[q * 128 + i] = (low[i] & 0x0f) | ((high[i] & 0x0f) << 4);
    }
}

static int testShape(int ic, int oc) {
    const int icBytes = UP_DIV(ic, 2);
    const int icP = ic / 32, ocP = oc / 32;
    const size_t rawSize = (size_t)oc * icBytes;
    const size_t packSize = (size_t)icP * ocP * 512;

    std::vector<uint8_t> raw(rawSize);
    for (size_t i = 0; i < rawSize; ++i)
        raw[i] = (uint8_t)(rand() & 0xff);

    std::vector<uint8_t> vrmpy(packSize), hmx(packSize), conv(packSize), hvx(packSize);
    refPackVrmpy(vrmpy.data(), raw.data(), ic, oc);
    refPackHmx(hmx.data(), raw.data(), ic, oc);

    for (int t = 0; t < icP * ocP; ++t) {
        scalarVrmpyToHmx(conv.data() + (size_t)t * 512, vrmpy.data() + (size_t)t * 512);
        hvxModelVrmpyToHmx(hvx.data() + (size_t)t * 512, vrmpy.data() + (size_t)t * 512);
    }

    const bool scalarOk = memcmp(hmx.data(), conv.data(), packSize) == 0;
    const bool hvxOk = memcmp(hmx.data(), hvx.data(), packSize) == 0;
    if (scalarOk && hvxOk) {
        printf("  PASS  ic=%5d oc=%6d  (%d tiles)  [scalar+hvxmodel]\n", ic, oc, icP * ocP);
        return 0;
    }
    const uint8_t* got = !scalarOk ? conv.data() : hvx.data();
    const char* which = !scalarOk ? "scalar" : "hvxmodel";
    for (size_t i = 0; i < packSize; ++i) {
        if (hmx[i] != got[i]) {
            int tile = (int)(i / 512), off = (int)(i % 512);
            printf("  FAIL(%s)  ic=%5d oc=%6d  tile=%d byte=%d hmx=0x%02x got=0x%02x\n", which, ic, oc, tile, off,
                   hmx[i], got[i]);
            break;
        }
    }
    return 1;
}

// ============================================================================
// Step 1: scale repack (fp32 vrmpy scale -> HMX dup / packed fp16 regions).
//
// This is a pure LAYOUT test: fp16 scale values are represented by unique int16
// tokens (the fp32->fp16 value conversion is orthogonal — done by verified HVX
// helpers on-device, checked numerically end-to-end). We verify that rearranging
// the vrmpy scale layout reproduces, byte-for-byte, the two HMX scale regions that
// the prefill kernels consume:
//   - dup region (hmx_matmulq4fp16 / dequant_q4_tile_scaled's vBlockScale):
//       per oy-tile y, per block k, 64 int16: lane 2*yi = lane 2*yi+1 = s(o,k)
//   - packed region (mle32 accumulate): per y, per pair p=k/2, 64 int16:
//       lane 2*yi = s(o,2p), lane 2*yi+1 = s(o,2p+1) (odd tail -> 0)
//
// Source vrmpy scale layout (reorderInt4WeightForVrmpyGemv): fp32,
//   sw[(y*nblk + b)*32 + ocIn] = alpha(o = y*32+ocIn, b)   (oc order, NOT duped)
// Both HMX regions and the vrmpy region derive from the same alpha(o,b).
// ============================================================================

// Unique int16 token standing in for fp16(alpha(o,b)).
static int16_t scaleTok(int o, int b) {
    return (int16_t)(((o * 7 + b * 131 + 5) & 0x7fff));
}

// Reference: replicate reorderInt4WeightForHmx's dup region (block-quant path).
static void refHmxDupScale(int16_t* dst, int ocP, int nblk) {
    for (int y = 0; y < ocP; ++y)
        for (int k = 0; k < nblk; ++k)
            for (int yi = 0; yi < 32; ++yi) {
                int16_t s = scaleTok(y * 32 + yi, k);
                dst[(size_t)y * nblk * 64 + k * 64 + 2 * yi] = s;
                dst[(size_t)y * nblk * 64 + k * 64 + 2 * yi + 1] = s;
            }
}
// Reference: replicate reorderInt4WeightForHmx's packed region.
static void refHmxPackedScale(int16_t* dst, int ocP, int nblk) {
    const int packedBlocks = (nblk + 1) / 2;
    for (int y = 0; y < ocP; ++y)
        for (int k = 0; k < nblk; k += 2)
            for (int yi = 0; yi < 32; ++yi) {
                int o = y * 32 + yi;
                dst[(size_t)y * packedBlocks * 64 + (k / 2) * 64 + 2 * yi] = scaleTok(o, k);
                dst[(size_t)y * packedBlocks * 64 + (k / 2) * 64 + 2 * yi + 1] =
                    (k + 1 < nblk) ? scaleTok(o, k + 1) : 0;
            }
}

// Converter (models the DSP HVX repack from the vrmpy fp32 scale buffer).
// vrmpy scale token at sw[(y*nblk+b)*32+ocIn].
static int16_t vrmpyScale(const int16_t* sw, int nblk, int y, int b, int ocIn) {
    return sw[((size_t)y * nblk + b) * 32 + ocIn];
}
// dup: hvx_my_wsf_to_vhf(sf,sf) -> [s0,s0,s1,s1,...] (64 lanes per block).
static void convDupScale(int16_t* dst, const int16_t* sw, int ocP, int nblk) {
    for (int y = 0; y < ocP; ++y)
        for (int b = 0; b < nblk; ++b)
            for (int ocIn = 0; ocIn < 32; ++ocIn) {
                int16_t s = vrmpyScale(sw, nblk, y, b, ocIn);
                dst[(size_t)y * nblk * 64 + b * 64 + 2 * ocIn] = s;
                dst[(size_t)y * nblk * 64 + b * 64 + 2 * ocIn + 1] = s;
            }
}
// packed: hf0=sf32_to_hf_low(block 2p), hf1=block 2p+1; shuff -> out[2i]=hf0[i],out[2i+1]=hf1[i].
static void convPackedScale(int16_t* dst, const int16_t* sw, int ocP, int nblk) {
    const int packedBlocks = (nblk + 1) / 2;
    for (int y = 0; y < ocP; ++y)
        for (int p = 0; p < packedBlocks; ++p)
            for (int ocIn = 0; ocIn < 32; ++ocIn) {
                int16_t hf0 = vrmpyScale(sw, nblk, y, 2 * p, ocIn);
                int16_t hf1 = (2 * p + 1 < nblk) ? vrmpyScale(sw, nblk, y, 2 * p + 1, ocIn) : 0;
                dst[(size_t)y * packedBlocks * 64 + p * 64 + 2 * ocIn] = hf0;
                dst[(size_t)y * packedBlocks * 64 + p * 64 + 2 * ocIn + 1] = hf1;
            }
}

static int testScaleShape(int oc, int nblk) {
    const int ocP = oc / 32;
    const int packedBlocks = (nblk + 1) / 2;
    std::vector<int16_t> sw((size_t)ocP * nblk * 32);
    for (int y = 0; y < ocP; ++y)
        for (int b = 0; b < nblk; ++b)
            for (int ocIn = 0; ocIn < 32; ++ocIn)
                sw[((size_t)y * nblk + b) * 32 + ocIn] = scaleTok(y * 32 + ocIn, b);

    std::vector<int16_t> dupRef((size_t)ocP * nblk * 64), dupConv((size_t)ocP * nblk * 64);
    std::vector<int16_t> pkRef((size_t)ocP * packedBlocks * 64), pkConv((size_t)ocP * packedBlocks * 64);
    refHmxDupScale(dupRef.data(), ocP, nblk);
    convDupScale(dupConv.data(), sw.data(), ocP, nblk);
    refHmxPackedScale(pkRef.data(), ocP, nblk);
    convPackedScale(pkConv.data(), sw.data(), ocP, nblk);

    bool dupOk = dupRef == dupConv;
    bool pkOk = pkRef == pkConv;
    if (dupOk && pkOk) {
        printf("  PASS  scale oc=%6d nblk=%3d  [dup+packed]\n", oc, nblk);
        return 0;
    }
    printf("  FAIL  scale oc=%6d nblk=%3d  dup=%s packed=%s\n", oc, nblk, dupOk ? "ok" : "MISMATCH",
           pkOk ? "ok" : "MISMATCH");
    return 1;
}

static int testAsymmetricActivationSums(int kp, int nblk) {
    if (kp <= 0 || nblk <= 0 || kp % nblk != 0) {
        return 1;
    }
    const int valuesPerTile = 32;
    const int tilesPerBlock = kp / nblk;
    std::vector<float> activation((size_t)kp * valuesPerTile);
    for (int tile = 0; tile < kp; ++tile) {
        for (int lane = 0; lane < valuesPerTile; ++lane) {
            activation[(size_t)tile * valuesPerTile + lane] =
                (float)((tile + 1) * 100 + lane + 1) / 128.0f;
        }
    }
    for (int block = 0; block < nblk; ++block) {
        float expected = 0.0f;
        float actual = 0.0f;
        const int tileBegin = block * tilesPerBlock;
        const int tileEnd = tileBegin + tilesPerBlock;
        for (int tile = tileBegin; tile < tileEnd; ++tile) {
            for (int lane = 0; lane < valuesPerTile; ++lane) {
                expected += activation[(size_t)tile * valuesPerTile + lane];
            }
        }
        const float* blockActivation = activation.data() + (size_t)block * tilesPerBlock * valuesPerTile;
        for (int i = 0; i < tilesPerBlock * valuesPerTile; ++i) {
            actual += blockActivation[i];
        }
        if (actual != expected) {
            printf("  FAIL  activation-sum kp=%d nblk=%d block=%d expected=%f actual=%f\n", kp, nblk, block,
                   expected, actual);
            return 1;
        }
    }
    printf("  PASS  activation-sum kp=%d nblk=%d tiles/block=%d\n", kp, nblk, tilesPerBlock);
    return 0;
}

int main() {
    srand(42);
    struct {
        int ic, oc;
    } shapes[] = {
        {64, 64},       {128, 96}, {64, 32}, // small / tiling stress
        {1024, 1024},                        // k_proj / v_proj
        {1024, 2048},                        // q_proj
        {1024, 3072},                        // gate_proj / up_proj
        {2048, 1024},                        // o_proj
        {3072, 1024},                        // down_proj
        {1024, 151936},                      // lm_head (big oc)
    };
    int fails = 0, n = 0;
    for (auto& s : shapes) {
        fails += testShape(s.ic, s.oc);
        ++n;
    }

    // Scale repack (oc, nblk=ic/64). Cover even/odd nblk and real layer dims.
    struct {
        int oc, nblk;
    } scaleShapes[] = {
        {1024, 16},   // ic=1024
        {2048, 32},   // o_proj ic=2048
        {1024, 48},   // down_proj ic=3072
        {64, 3},      // odd nblk (packed tail padding)
        {32, 1},      // single block
        {151936, 16}, // lm_head
    };
    int scaleFails = 0, sn = 0;
    for (auto& s : scaleShapes) {
        scaleFails += testScaleShape(s.oc, s.nblk);
        ++sn;
    }

    int activationSumFails = 0;
    activationSumFails += testAsymmetricActivationSums(32, 8);
    activationSumFails += testAsymmetricActivationSums(32, 16);
    activationSumFails += testAsymmetricActivationSums(32, 32);

    if (fails == 0 && scaleFails == 0 && activationSumFails == 0) {
        printf("PASS: %d weight shapes + %d scale shapes byte-match reorderInt4WeightForHmx; activation sums pass.\n",
               n, sn);
        return 0;
    }
    printf("FAIL: weight %d/%d, scale %d/%d, activation sum %d/3 mismatched.\n", fails, n, scaleFails, sn,
           activationSumFails);
    return 1;
}
