# RVV linear attention regression

Build and run with both `-ffp-contract=fast` and `-ffp-contract=off`:

```sh
$CXX -std=c++11 -O3 -march=rv64gcv -ffp-contract=fast \
  -DMNN_USE_RVV -DMNN_RVV_KERNEL_TEST_MAIN \
  source/backend/cpu/riscv/rvv/MNNLinearAttentionKernels.cpp \
  test/backend/cpu/RVVLinearAttentionTest.cpp -o rvv-linear-test
./rvv-linear-test
```

The reference explicitly specifies FMA ordering and requires exact equality. It covers normalized
and unnormalized keys, eight recurrent steps, zero dimensions, unaligned pointers, tails and guards.
The RISC-V scalar fallback uses the same arithmetic contract even when automatic contraction is off.
The rank-one product is rounded before fusing the decay term; the correction fuses the subtraction
before multiplying beta. Reversing which product is fused changes recurrent state and model logits.

Also run `backend/cpu/rvv/linear_attention` and `op/linear_attention` with one and four threads in
`run_test.out`. Validate fixed-history logits for short and long prompts against the scalar fallback
and the pre-change Release baseline. Token agreement alone is insufficient. QEMU VLEN 128, 256,
512 and 1024 checks tail behavior but does not establish hardware performance.
