//
//  getPerformance.cpp
//  MNN
//
//  Created by MNN on 2019/03/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include <chrono>
#include <cstdint>
#include <vector>
#include <stdlib.h>
#include "MNNDefine.h"
#include "Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

class Timer {
private:
    std::chrono::high_resolution_clock::time_point inTime, outTime;

public:
    void startTimer() {
        inTime = std::chrono::high_resolution_clock::now();
    }

    // unit ms
    float getCostTimer() {
        outTime = std::chrono::high_resolution_clock::now();
        return (float)(std::chrono::duration_cast<std::chrono::microseconds>(outTime - inTime).count());
    }
};

int getCpuCounts() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (fp == nullptr) {
        MNN_PRINT("fopen error ! \n");
    }
    int cpuCounts = 0;
    char data[1024];
    while (!feof(fp)) {
        char* a = fgets(data, 1024, fp);

        if (a == nullptr) {
            break;
        }
        if (memcmp(data, "processor", 9) == 0) {
            cpuCounts++;
        }
    }

    fclose(fp);
    fp = nullptr;
    return cpuCounts;
}

// 0 max 1 min 2 cur
void getFreqKhz(int cpuid, std::vector<int>& freqVector) {
    char path[256];
    int freqKhz = -1;
    // max
    sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
    FILE* fp = fopen(path, "rb");
    if (nullptr == fp) {
        MNN_PRINT("cpuinfo_max_freq fopen error ! \n");
    } else {
        fscanf(fp, "%d", &freqKhz);
        fclose(fp);
        freqVector.push_back(freqKhz);
    }

    // min
    sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_min_freq", cpuid);
    fp = fopen(path, "rb");
    if (nullptr == fp) {
        MNN_PRINT("cpuinfo_min_freq fopen error ! \n");
    } else {
        freqKhz = -1;
        fscanf(fp, "%d", &freqKhz);
        fclose(fp);
        freqVector.push_back(freqKhz);
    }

    // cur
    // sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_cur_freq", cpuid);
    // fp = fopen(path, "rb");
    // if(nullptr == fp){
    //     MNN_PRINT("cpuinfo_cur_freq fopen error ! \n");
    // }else{
    //     freqKhz = -1;
    //     fscanf(fp, "%d", &freqKhz);
    //     fclose(fp);
    //     freqVector.push_back(freqKhz);
    // }
}

void cpuUint8MlaTest(uint64_t loopCounts) {
#ifdef MNN_USE_NEON
#ifndef __aarch64__
    uint8_t* sumPtr = (uint8_t*)malloc(8 * sizeof(uint8_t));
    uint8_t a       = 1;
    uint8_t b       = 2;
    uint8_t c       = 3;
    uint8_t d       = 4;
    uint8_t e       = 5;
    uint8_t f       = 6;
    
    __asm__ __volatile__(
         "vdup.16   d3, %3              \n"
         "vdup.16   d4, %4              \n"
         "vdup.16   d5, %5              \n"
         "vdup.16   d6, %6              \n"
         "vdup.16   d7, %7              \n"
         "vdup.16   d8, %8              \n"
         "vdup.32   q15, %3              \n"
         "vdup.32   q14, %3              \n"
         
         "0:                             \n"
         "vmlal.s16  q15, d28, d3        \n"
         "vmlal.s16  q14, d29, d4        \n"
         "vmlal.s16  q15, d30, d5        \n"
         "vmlal.s16  q14, d31, d6        \n"
         "vmlal.s16  q15, d28, d7        \n"
         "vmlal.s16  q14, d29, d8        \n"
         "subs       %1, %1, #1          \n"
         "bgt        0b                  \n"
         "vst1.32   {d28-d29}, [%0]!   \n"
         "vst1.32   {d30-d31}, [%0]   \n"
         : "+r"(sumPtr)
         : "r"(loopCounts), "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f)
         : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q14", "q15");
    
    MNN_PRINT("sum : %d, %d, %d, %d \n", sumPtr[0], sumPtr[1], sumPtr[2], sumPtr[3]);
    free(sumPtr);
    
#else
    
    int32x4_t sum0 = vdupq_n_s32(1);
    int32x4_t sum1 = vdupq_n_s32(1);
    int16x4_t a    = vdup_n_s16(3);
    int16x4_t b    = vdup_n_s16(4);
    int16x4_t c    = vdup_n_s16(5);
    int16x4_t d    = vdup_n_s16(6);
    int16x4_t e    = vdup_n_s16(7);
    int16x4_t f    = vdup_n_s16(8);
    
    for (uint64_t i = 0; i < loopCounts; i++) {
        sum0 = vmlal_s16(sum0, a, f);
        sum1 = vmlal_s16(sum1, b, e);
        sum0 = vmlal_s16(sum0, c, a);
        sum1 = vmlal_s16(sum1, d, b);
        sum0 = vmlal_s16(sum0, e, a);
        sum1 = vmlal_s16(sum1, f, b);
    }
    MNN_PRINT("sum0 : %d, %d, %d, %d \n", sum0[0], sum0[1], sum0[2], sum0[3]);
    MNN_PRINT("sum0 : %d, %d, %d, %d \n", sum1[0], sum1[1], sum1[2], sum1[3]);
    
#endif
#endif
}

void cpuFloatMlaTest(uint64_t loopCounts) {
#ifdef MNN_USE_NEON
#ifndef __aarch64__

    float* sumPtr = (float*)malloc(8 * sizeof(float));
    float a       = 1.0;
    float b       = 1.1;
    float c       = 1.2;
    float d       = 1.3;
    float e       = 1.4;
    float f       = 1.5;

    __asm__ __volatile__(
        "vdup.f32   q3, %3              \n"
        "vdup.f32   q4, %4              \n"
        "vdup.f32   q5, %5              \n"
        "vdup.f32   q6, %6              \n"
        "vdup.f32   q7, %7              \n"
        "vdup.f32   q7, %8              \n"
        "vdup.f32   q15, %3              \n"
        "vdup.f32   q14, %3              \n"

        "0:                             \n"
        "vmla.f32   q15, q15, q3        \n"
        "vmla.f32   q14, q14, q4        \n"
        "vmla.f32   q15, q15, q5        \n"
        "vmla.f32   q14, q14, q6        \n"
        "vmla.f32   q15, q15, q7        \n"
        "vmla.f32   q14, q14, q8        \n"
        "subs       %1, %1, #1          \n"
        "bgt        0b                  \n"
        "vst1.f32   {d28-d29}, [%0]!   \n"
        "vst1.f32   {d30-d31}, [%0]   \n"
        : "+r"(sumPtr)
        : "r"(loopCounts), "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f)
        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q14", "q15");

    MNN_PRINT("sum : %f, %f, %f, %f \n", sumPtr[0], sumPtr[1], sumPtr[2], sumPtr[3]);
    free(sumPtr);

#else

    float32x4_t sum0 = vdupq_n_f32(1.0);
    float32x4_t sum1 = vdupq_n_f32(1.0);
    float32x4_t a    = vdupq_n_f32(1.0);
    float32x4_t b    = vdupq_n_f32(1.1);
    float32x4_t c    = vdupq_n_f32(1.2);
    float32x4_t d    = vdupq_n_f32(1.3);
    float32x4_t e    = vdupq_n_f32(1.4);
    float32x4_t f    = vdupq_n_f32(1.5);

    for (uint64_t i = 0; i < loopCounts; i++) {
        sum0 = vmlaq_f32(sum0, a, sum0);
        sum1 = vmlaq_f32(sum1, b, sum1);
        sum0 = vmlaq_f32(sum0, c, sum0);
        sum1 = vmlaq_f32(sum1, d, sum1);
        sum0 = vmlaq_f32(sum0, e, sum0);
        sum1 = vmlaq_f32(sum1, f, sum1);
    }
    MNN_PRINT("sum0 : %f, %f, %f, %f \n", sum0[0], sum0[1], sum0[2], sum0[3]);
    MNN_PRINT("sum0 : %f, %f, %f, %f \n", sum1[0], sum1[1], sum1[2], sum1[3]);
#endif
#endif
}

void cpuFLOPSPerformance() {
    int loopCounts = 10000000;
    int threadCounts    = getCpuCounts();

    MNN_PRINT("CPU PERFORMANCE -> loopCounts : %d , threadCounts : %d \n", loopCounts, threadCounts);

    std::vector<int> freqVector;
    for (int i = 0; i < getCpuCounts(); i++) {
        getFreqKhz(i, freqVector);
        // MNN_PRINT("core %d : max : %d, min : %d \n",i, freqVector.at(0), freqVector.at(1));
    }

    // warm up
    cpuFloatMlaTest(loopCounts);

    Timer timeInstance;
    timeInstance.startTimer();
    cpuFloatMlaTest(loopCounts);
    float costTime_ms = timeInstance.getCostTimer();
    double costTime_s = (double)(costTime_ms) / 1000000.0f;
    // MNN_PRINT("cost time : %f \n", costTime_s);
    double mlaCounts_g = loopCounts * 6 * 4 / 1000000000.0f;
    float gflops       = mlaCounts_g / costTime_s;
    getFreqKhz(0, freqVector);
    MNN_PRINT(" ======================== float ===============================\n");
    MNN_PRINT("CPU float gflops : %f , max freq gkhz : %f \n", gflops, (float)freqVector.at(0) / 1000000.0f);
    
    
    cpuUint8MlaTest(loopCounts);
    timeInstance.startTimer();
    cpuUint8MlaTest(loopCounts);
    costTime_ms = timeInstance.getCostTimer();
    costTime_s = (double)(costTime_ms) / 1000000.0f;
    // MNN_PRINT("cost time : %f \n", costTime_s);
    mlaCounts_g = loopCounts * 6 * 4 / 1000000000.0f;
    gflops       = mlaCounts_g / costTime_s;
    MNN_PRINT(" ============================ uint8 ===========================\n");
    MNN_PRINT("CPU uint8 gflops : %f , max freq gkhz : %f \n", gflops, (float)freqVector.at(0) / 1000000.0f);

}

int main(int argc, const char* argv[]) {
    MNN_PRINT("Start PERFORMANCE !!! \n");

    cpuFLOPSPerformance();

    return 0;
}
