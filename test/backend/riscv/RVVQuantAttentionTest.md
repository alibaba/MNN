# RVV KV quantization regression

Build a standalone target executable from the repository root:

```sh
$CXX -std=c++11 -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fno-tree-vectorize \
  -DMNN_SUPPORT_TRANSFORMER_FUSE -DMNN_RVV_QUANT_TEST_MAIN \
  source/backend/cpu/riscv/rvv/MNNQuantAttentionKey.cpp \
  source/backend/cpu/riscv/rvv/MNNQuantAttentionValue.cpp \
  test/backend/riscv/RVVQuantAttentionTest.cpp -o rvv-kv-test
./rvv-kv-test
```

For cross compilation, supply the toolchain/sysroot and optionally `-static`, then run using a configured
`qemu-riscv64 -cpu max,vlen=128 ./rvv-kv-test`. Repeat with VLEN 256, 512 and 1024.
QEMU execution does not establish hardware performance.

The reference is copied from `CommonOptFunction.cpp` at `bef71b9756a2c77549eddbe33eb97290e3b16602`.
Tests compare packed bytes, scales/biases, cached maxima and dequantized sums. They cover multiple heads,
prefill/decode, append offsets crossing KV blocks, LP/HP layouts, constant values, clipping and signed half ties.
The scalar key reference divides by zero for constant transformed rows; that undefined NaN-to-integer conversion
is excluded from bitwise equivalence. The test instead checks the explicit RVV -128 representation with zero scale.
Scalar key cases use whole blocks; non-divisible key blocks, non-finite inputs, full models and board performance
remain additional validation requirements.

An initial GCC 15.2 build using the explicit RMM conversion left scalar dequantization/sums under RMM inside
the loop. The current implementation uses RTZ plus a fractional tie comparison and does not change `frm`.
Keep the packed-output and sum checks together: a byte-identical quantized output alone missed this defect.
