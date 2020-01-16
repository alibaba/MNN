//
//  mnnportraitnative.cpp
//  MNN
//
//  Created by MNN on 2019/01/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <android/bitmap.h>
#include <jni.h>
#include <string.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>

extern "C" JNIEXPORT jintArray JNICALL
Java_com_taobao_android_mnn_MNNPortraitNative_nativeConvertMaskToPixelsMultiChannels(JNIEnv *env, jclass jclazz,
                                                                                     jfloatArray jmaskarray,
                                                                                     jint length) {
    float *scores = (float *)env->GetFloatArrayElements(jmaskarray, 0);

    int dst32[length];

#if 0
    for (int l = 0; l < length; l++) {
        int* dst = dst32 + l;
        float* src = scores + l;
        float max = scores[l];
        float min = scores[l];
        for(int c = 0; c < 21; c++){
            float data = src[c*length];
            if(max < data){
                max = data;
            }
            if(min > data){
                min = data;
            }
        }

        unsigned data = src[15*length];
        float range = 255.0f / (exp(max) - exp(min));
        float result = (exp(data) - exp(min)) * range;

        unsigned result_uint8 = result > 255.0f ? 255 : result;

        unsigned a = result_uint8;
        unsigned r = a;
        unsigned g = a;
        unsigned b = a;
        // ARGB
        dst[0] = a << 24 | r << 16 | g << 8 | b;
    }
#else
    for (int l = 0; l < length; l++) {
        int *dst   = dst32 + l;
        float *src = scores + l;
        float max  = scores[l];
        for (int c = 0; c < 21; c++) {
            if (max < src[c * length]) {
                max = src[c * length];
            }
        }
        unsigned a = src[15 * length] == max ? 0 : 255;
        unsigned r = a;
        unsigned g = a;
        unsigned b = a;
        // ARGB
        dst[0] = a << 24 | r << 16 | g << 8 | b;
    }
#endif
    jintArray arr = env->NewIntArray(length);
    env->SetIntArrayRegion(arr, 0, length, dst32);

    env->ReleaseFloatArrayElements(jmaskarray, scores, 0);

    return arr;
}
