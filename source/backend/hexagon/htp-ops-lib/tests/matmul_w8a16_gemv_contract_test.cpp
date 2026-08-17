// Lightweight host-side validation of the W8A16 M=1 GEMV layout contract
// (see ops/matmul_w8a16_gemv_i8.c). The DSP kernel consumes the EXISTING HMX
// int8 tile layout — byte (ix>>2)*128 + oy*4 + (ix&1)*2 + ((ix>>1)&1) holds
// w(oy, ix) for a 32x32 tile, i.e. each 4-k group is stored as [k0,k2,k1,k3] —
// and compensates by splatting the activation in the same permuted order
// (swap bytes 1 and 2 of each 4-byte word; the dot sum is unchanged). This
// test mirrors that math in portable C++ and compares against a naive CPU
// reference on the handoff §6 nonuniform inputs.
//
// Build/run (no MNN dependency):
//   g++ -std=c++11 -O2 -o /tmp/gemv_contract_test matmul_w8a16_gemv_contract_test.cpp
//   /tmp/gemv_contract_test

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kBlock = 64;  // quantization block size

// Replicate reorderInt8SymWeightForHmx's 32x32 tile byte order: returns the
// byte offset within the 1024-byte tile that holds w(oy, ix).
inline size_t hmx_tile_index(int oy, int ix) {
  const int ixPair = ix / 2;
  const int ixRem  = ix & 1;
  const int lane   = oy * 2 + ixRem;
  return static_cast<size_t>(ixPair / 2) * 128 + static_cast<size_t>(lane) * 2 + (ixPair & 1);
}

// Byte order of each 4-k group inside the HMX tile: byte p holds w(k = 4g+perm[p]).
const int kPerm[4] = { 0, 2, 1, 3 };

// The kernel's permuted activation word: [a(k0), a(k2), a(k1), a(k3)].
inline void permuted_splat(const int8_t *a, int k0, int8_t out[4]) {
  out[0] = a[k0 + kPerm[0]];
  out[1] = a[k0 + kPerm[1]];
  out[2] = a[k0 + kPerm[2]];
  out[3] = a[k0 + kPerm[3]];
}

void quantize_activation(const std::vector<float> &a, std::vector<int8_t> &aq, float &sa) {
  float absmax = 0.0f;
  for (float v : a) {
    if (std::fabs(v) > absmax) {
      absmax = std::fabs(v);
    }
  }
  if (absmax <= 0.0f) {
    absmax = 1.0f;
  }
  sa        = absmax / 127.0f;
  float inv = 127.0f / absmax;
  for (size_t i = 0; i < a.size(); ++i) {
    int v = static_cast<int>(std::lrintf(a[i] * inv));
    if (v > 127) {
      v = 127;
    }
    if (v < -127) {
      v = -127;
    }
    aq[i] = static_cast<int8_t>(v);
  }
}

// Portable GEMV over the HMX-layout blob, mirroring the DSP compute:
//   out[oc] = sa * sum_b scale[b][oc] * sum_{k in b} aq[k]*wq[oc][k] + bias[oc]
// with the per-4-k-group dot evaluated in the HMX weight order paired with the
// permuted activation splat (vrmpy semantics).
void gemv_portable(int K, int N, const std::vector<int8_t> &aq, float sa, const std::vector<int8_t> &blob,
                   const std::vector<float> &scales, const std::vector<float> &bias, int nblk,
                   std::vector<float> &out) {
  const int kp   = K / 32;
  const int np   = N / 32;
  const int ktpb = kBlock / 32;  // k-tiles per scale block (block 64 -> 2)
  out.assign(N, 0.0f);
  for (int y = 0; y < np; ++y) {
    for (int b = 0; b < nblk; ++b) {
      int32_t acc[32] = { 0 };
      for (int ktl = 0; ktl < ktpb; ++ktl) {
        const int     kx   = b * ktpb + ktl;
        const int8_t *tile = blob.data() + (static_cast<size_t>(y) * kp + kx) * 1024;
        for (int ocIn = 0; ocIn < 32; ++ocIn) {
          for (int g = 0; g < 8; ++g) {
            const int k0 = kx * 32 + 4 * g;
            int8_t    splat[4];
            permuted_splat(aq.data(), k0, splat);
            for (int p = 0; p < 4; ++p) {
              acc[ocIn] += static_cast<int32_t>(tile[g * 128 + ocIn * 4 + p]) * splat[p];
            }
          }
        }
      }
      for (int ocIn = 0; ocIn < 32; ++ocIn) {
        const int o = y * 32 + ocIn;
        out[o] += scales[(static_cast<size_t>(y) * nblk + b) * 32 + ocIn] * static_cast<float>(acc[ocIn]);
      }
    }
  }
  for (int o = 0; o < N; ++o) {
    out[o] = out[o] * sa + bias[o];
  }
}

// Naive reference: out[oc] = sa * sum_b scale[oc][b] * sum_{k in b} aq[k]*wq[oc][k] + bias[oc].
void gemv_naive(int K, int N, const std::vector<int8_t> &aq, float sa, const std::vector<int8_t> &wq,
                const std::vector<float> &scale, const std::vector<float> &bias, int nblk, std::vector<float> &out) {
  out.assign(N, 0.0f);
  for (int o = 0; o < N; ++o) {
    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b) {
      int32_t blockSum = 0;
      for (int k = b * kBlock; k < (b + 1) * kBlock; ++k) {
        blockSum += static_cast<int32_t>(aq[k]) * wq[o * K + k];
      }
      acc += scale[o * nblk + b] * static_cast<float>(blockSum);
    }
    out[o] = acc * sa + bias[o];
  }
}

int run_case(int K, int N) {
  const int nblk = K / kBlock;
  const int kp   = K / 32;
  const int np   = N / 32;

  // handoff §6 nonuniform data (M=1).
  std::vector<float> a(K);
  for (int k = 0; k < K; ++k) {
    a[k] = std::sin(static_cast<float>(k) * 17.0f * 0.013f) * 1.7f + (static_cast<float>(k % 7) - 3.0f) * 0.09f;
  }
  std::vector<int8_t> wq(N * K);
  std::vector<float>  scale(N * nblk);
  std::vector<float>  bias(N);
  for (int o = 0; o < N; ++o) {
    for (int k = 0; k < K; ++k) {
      wq[o * K + k] = static_cast<int8_t>((o * 29 + k * 11 + 7) % 255 - 127);
    }
    for (int b = 0; b < nblk; ++b) {
      scale[o * nblk + b] = 0.003f + static_cast<float>((o * 5 + b * 3) % 17) * 0.0007f;
    }
    bias[o] = (static_cast<float>(o % 19) - 9.0f) * 0.017f;
  }

  std::vector<int8_t> aq(K);
  float               sa = 0.0f;
  quantize_activation(a, aq, sa);

  // Build the HMX-layout weight blob: tile (y,kx) at (y*kp+kx)*1024; byte
  // (g*128 + ocIn*4 + p) = w(ocIn, kx*32 + 4g + perm[p]). Sanity-check the
  // tile-index formula against the group/byte derivation.
  std::vector<int8_t> blob((size_t) np * kp * 1024);
  for (int y = 0; y < np; ++y) {
    for (int kx = 0; kx < kp; ++kx) {
      int8_t *tile = blob.data() + ((size_t) y * kp + kx) * 1024;
      for (int oy = 0; oy < 32; ++oy) {
        for (int ix = 0; ix < 32; ++ix) {
          tile[hmx_tile_index(oy, ix)] = wq[(y * 32 + oy) * K + kx * 32 + ix];
        }
      }
    }
  }
  // Verify the group/byte derivation: byte (g*128 + ocIn*4 + p) == tile index of (ocIn, 4g+perm[p]).
  for (int g = 0; g < 8; ++g) {
    for (int ocIn = 0; ocIn < 32; ++ocIn) {
      for (int p = 0; p < 4; ++p) {
        const size_t byGroup = (size_t) g * 128 + ocIn * 4 + p;
        const size_t byIndex = hmx_tile_index(ocIn, 4 * g + kPerm[p]);
        if (byGroup != byIndex) {
          std::printf("K=%d N=%d: layout derivation mismatch g=%d ocIn=%d p=%d\n", K, N, g, ocIn, p);
          return 1;
        }
      }
    }
  }

  // fp32 scale blob (reorderInt8ScaleForGemv contract).
  std::vector<float> scales((size_t) np * nblk * 32, 0.0f);
  for (int y = 0; y < np; ++y) {
    for (int b = 0; b < nblk; ++b) {
      for (int ocIn = 0; ocIn < 32; ++ocIn) {
        scales[((size_t) y * nblk + b) * 32 + ocIn] = scale[(y * 32 + ocIn) * nblk + b];
      }
    }
  }

  std::vector<float> outP, outN;
  gemv_portable(K, N, aq, sa, blob, scales, bias, nblk, outP);
  gemv_naive(K, N, aq, sa, wq, scale, bias, nblk, outN);

  float maxErr = 0.0f;
  int   bad    = 0;
  for (int i = 0; i < N; ++i) {
    const float err = std::fabs(outP[i] - outN[i]);
    if (err > maxErr) {
      maxErr = err;
    }
    if (err > 1e-4f) {
      ++bad;
    }
  }
  const bool ok = (bad == 0);
  std::printf("K=%4d N=%4d block=%d: max_err=%.3e bad=%d %s\n", K, N, kBlock, maxErr, bad, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  int rc = 0;
  rc |= run_case(64, 32);
  rc |= run_case(64, 64);
  rc |= run_case(128, 32);
  rc |= run_case(128, 64);
  rc |= run_case(1024, 32);
  std::printf("%s\n", rc == 0 ? "ALL CASES PASS" : "FAILURES PRESENT");
  return rc;
}
