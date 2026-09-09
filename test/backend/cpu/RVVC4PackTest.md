# RVV C4 packing regression

The tensor ABI stays C4 at every hardware VLEN. This change does not enable automatic wider
channel packs, change MatMul tiles, or replace the generic Vec4 representation.

```sh
$CXX -std=c++11 -O3 -march=rv64gcv -DMNN_RVV_PACK_TEST_MAIN \
 source/backend/cpu/riscv/rvv/MNNPackC4.cpp \
 source/backend/cpu/riscv/rvv/MNNUnpackC4.cpp \
 source/backend/cpu/riscv/rvv/MNNPackC4ForMatMul_A.cpp \
 test/backend/cpu/RVVC4PackTest.cpp -o c4-pack-test
./c4-pack-test
```

The test checks FP32/Int8/Int16 normal and transposed packing, padding, non-contiguous plane
strides, guards and MatMul-A offsets against independent scalar indexing. Test VLEN 128, 256,
512 and 1024 with QEMU for correctness; measure speed only on native hardware. Also build the
runtime and run resize, interpolation, MatMul, depthwise convolution and Attention tests with
one and four threads. Compare fixed-history model logits for short and long inputs.
