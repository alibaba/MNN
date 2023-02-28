#ifndef MNNCUDADEFINE_HPP
#define MNNCUDADEFINE_HPP

#define PACK_NUMBER 8
#define INT8_PACK_NUMBER 16

#define MNN_CUDA_HALF2_MAX(a, b)                     \
    do {                                             \
        (a).x = __hgt((a).x, (b).x) ? (a).x : (b).x; \
        (a).y = __hgt((a).y, (b).y) ? (a).y : (b).y; \
    } while (0)

#define MNN_CUDA_HALF2_MIN(a, b)                     \
    do {                                             \
        (a).x = __hlt((a).x, (b).x) ? (a).x : (b).x; \
        (a).y = __hlt((a).y, (b).y) ? (a).y : (b).y; \
    } while (0)

#endif
