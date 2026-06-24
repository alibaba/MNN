// Copyright @ MNN
#pragma once

namespace MNN {

// Plane softmax (scalar T macro)
extern const char* gSoftmaxPlaneSrc;
// Plane softmax with simd_max/simd_sum (scalar T macro)
extern const char* gSoftmaxPlaneSgSrc;
// Plane softmax with enlarged local size (multi-simdgroup reduce)
extern const char* gSoftmaxPlaneSgTG;

// Plane softmax with simd reduce used by Attention (uses ftype and axis_align_length)
extern const char* gSoftmaxSgReduce;

}
