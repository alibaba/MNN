// Copyright 2018 Alibaba Group Holding Limited. All rights reserved.
#ifdef MNN_RVV_PACK_TEST_MAIN
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#define DECL(T, S)                                                           \
    void MNNPackCUnit##S##_RVV(T*, const T*, size_t, size_t, int*);          \
    void MNNUnpackCUnit##S##_RVV(T*, const T*, size_t, size_t, int*);        \
    void MNNPackCUnitTranspose##S##_RVV(T*, const T*, size_t, size_t, int*); \
    void MNNUnpackCUnitTranspose##S##_RVV(T*, const T*, size_t, size_t, int*);
DECL(float, )
DECL(int8_t, Int8)
DECL(int16_t, Int16)
void MNNPackC4ForMatMul_A_RVV(float*, float const**, const int32_t*, const int32_t*);
template <class T>
using Fn = void (*)(T*, const T*, size_t, size_t, int*);
template <class T>
bool check(Fn<T> pack, Fn<T> unpack, Fn<T> packT, Fn<T> unpackT) {
    for (int padding : {0, 1})
    for (int area : {0, 1, 3, 16, 17, 65, 257})
        for (int depth : {0, 1, 3, 4, 5, 16, 17, 33, 65}) {
            int rounded = (depth + 3) / 4 * 4;
            int off[2] = {area + padding * 3, area + padding * 5};
            std::vector<T> src(20000), actual(20000, T(93)), expected = actual;
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = T(int(i * 17 % 101) - 50);
            for (int c = 0; c < rounded; ++c)
                for (int x = 0; x < area; ++x)
                    expected[1 + (c / 4 * off[1] + x) * 4 + c % 4] = c < depth ? src[1 + c * off[0] + x] : T(0);
            pack(actual.data() + 1, src.data() + 1, area, depth, off);
            if (actual != expected) {
                std::printf("pack area=%d depth=%d bytes=%zu\n", area, depth, sizeof(T));
                return false;
            }
            std::fill(actual.begin(), actual.end(), T(93));
            expected = actual;
            for (int c = 0; c < depth; ++c)
                for (int x = 0; x < area; ++x)
                    expected[1 + c * off[1] + x] = src[1 + (c / 4 * off[0] + x) * 4 + c % 4];
            unpack(actual.data() + 1, src.data() + 1, area, depth, off);
            if (actual != expected) {
                std::printf("unpack area=%d depth=%d bytes=%zu\n", area, depth, sizeof(T));
                return false;
            }
            std::fill(actual.begin(), actual.end(), T(93));
            expected = actual;
            for (int c = 0; c < rounded; ++c)
                for (int x = 0; x < area; ++x)
                    expected[1 + (c / 4 * off[1] + x) * 4 + c % 4] = c < depth ? src[1 + x * depth + c] : T(0);
            packT(actual.data() + 1, src.data() + 1, area, depth, off);
            if (actual != expected) {
                std::printf("packT area=%d depth=%d bytes=%zu\n", area, depth, sizeof(T));
                return false;
            }
            off[1] = depth + padding * 5;
            std::fill(actual.begin(), actual.end(), T(93));
            expected = actual;
            for (int c = 0; c < depth; ++c)
                for (int x = 0; x < area; ++x)
                    expected[1 + x * off[1] + c] = src[1 + (c / 4 * off[0] + x) * 4 + c % 4];
            unpackT(actual.data() + 1, src.data() + 1, area, depth, off);
            if (actual != expected) {
                std::printf("unpackT area=%d depth=%d bytes=%zu\n", area, depth, sizeof(T));
                return false;
            }
        }
    return true;
}
int main() {
    if (!check<float>(MNNPackCUnit_RVV, MNNUnpackCUnit_RVV, MNNPackCUnitTranspose_RVV, MNNUnpackCUnitTranspose_RVV) ||
        !check<int8_t>(MNNPackCUnitInt8_RVV, MNNUnpackCUnitInt8_RVV, MNNPackCUnitTransposeInt8_RVV,
                       MNNUnpackCUnitTransposeInt8_RVV) ||
        !check<int16_t>(MNNPackCUnitInt16_RVV, MNNUnpackCUnitInt16_RVV, MNNPackCUnitTransposeInt16_RVV,
                        MNNUnpackCUnitTransposeInt16_RVV))
        return 1;
    size_t cases = 1512;
    for (int e : {0, 1, 3, 16, 17, 65})
        for (int l : {0, 1, 3, 4, 5, 17})
            for (int stride : {1, 2, 3}) {
                int ed = e + 5, er = (e + 1) * stride + 5;
                int32_t info[] = {1, er, ed, stride}, el[] = {e, l, 2, 1};
                std::vector<float> src(20000), actual(20000, 93), expected = actual;
                for (size_t i = 0; i < src.size(); ++i)
                    src[i] = float(int(i * 17 % 101) - 50);
                const float* ptr = src.data() + 1;
                for (int x = 0; x < l; ++x)
                    for (int y = 0; y < e; ++y)
                        expected[1 + ed + 2 + x * ed + y] = ptr[x / 4 * er * 4 + y * 4 * stride + x % 4];
                MNNPackC4ForMatMul_A_RVV(actual.data() + 1, &ptr, info, el);
                if (actual != expected) {
                    std::printf("packA e=%d l=%d stride=%d\n", e, l, stride);
                    return 2;
                }
                ++cases;
            }
    std::printf("C4 RVV packing: %zu scalar/guard comparisons passed\n", cases);
}
#endif
