//
//  ResizeFunction.h
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ResizeFunction_h
#define ResizeFunction_h

#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

void MNNCubicSampleC4(const float* src, float* dst, int32_t* position, const float* factor, size_t number);
void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t,
                    size_t number);

#ifdef __cplusplus
}
#endif

#endif /* ResizeFunction_hpp */
