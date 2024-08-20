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
#include <thread>
#include <MNN/AutoTime.hpp>
#include <stdlib.h>
#include <MNN/MNNDefine.h>
#include "core/Macro.h"
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
        return 0;
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
        freqVector.emplace_back(0);
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
        freqVector.emplace_back(0);
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

void cpuFloatMlaTest(int32_t loopCounts) {
#ifdef MNN_USE_NEON
#ifndef __aarch64__

    __asm__ __volatile__(
        "mov r12, %0\n"
        "0:                             \n"
        "vmla.f32   q15, q15, d0[0]        \n"
        "vmla.f32   q14, q14, d0[1]        \n"
        "vmla.f32   q13, q13, d1[0]        \n"
        "vmla.f32   q12, q12, d1[1]        \n"
        "vmla.f32   q11, q11, d2[0]        \n"
        "vmla.f32   q10, q10, d2[1]        \n"
        "vmla.f32   q9, q9, d3[0]        \n"
        "vmla.f32   q8, q8, d3[1]        \n"
        "vmla.f32   q7, q7, d4[0]        \n"
        "vmla.f32   q6, q6, d4[1]        \n"
        "vmla.f32   q5, q5, d5[0]        \n"
        "vmla.f32   q4, q4, d5[1]        \n"
        "vmla.f32   q3, q3, d6[0]        \n"
        "subs       r12, r12, #1          \n"
        "bne        0b                  \n"
        :
        : "r"(loopCounts)
        : "cc", "memory", "r12", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q14", "q15"
    );

#else
    __asm__ __volatile__(
        "mov w9, %w0\n"
        "0:                           \n"
        "fmla v31.4s, v31.4s, v0.s[0]\n"
        "fmla v30.4s, v30.4s, v0.s[1]\n"
        "fmla v29.4s, v29.4s, v0.s[2]\n"
        "fmla v28.4s, v28.4s, v0.s[3]\n"
        "fmla v27.4s, v27.4s, v1.s[0]\n"
        "fmla v26.4s, v26.4s, v1.s[1]\n"
        "fmla v25.4s, v25.4s, v1.s[2]\n"
        "fmla v24.4s, v24.4s, v1.s[3]\n"
        "fmla v23.4s, v23.4s, v3.s[0]\n"
        "fmla v22.4s, v22.4s, v3.s[1]\n"
        "fmla v21.4s, v21.4s, v3.s[2]\n"
        "fmla v20.4s, v20.4s, v3.s[3]\n"
        "fmla v19.4s, v19.4s, v4.s[0]\n"
        "fmla v18.4s, v18.4s, v4.s[1]\n"
        "fmla v17.4s, v17.4s, v4.s[2]\n"
        "fmla v16.4s, v16.4s, v4.s[3]\n"
        "fmla v15.4s, v15.4s, v5.s[0]\n"
        "fmla v14.4s, v14.4s, v5.s[1]\n"
        "fmla v13.4s, v13.4s, v5.s[2]\n"
        "fmla v12.4s, v12.4s, v5.s[3]\n"
        "fmla v11.4s, v11.4s, v6.s[0]\n"
        "fmla v10.4s, v10.4s, v6.s[1]\n"
        "fmla v9.4s, v9.4s, v6.s[2]\n"
        "fmla v8.4s, v8.4s, v6.s[3]\n"
        "fmla v7.4s, v7.4s, v2.s[0]\n"
        "subs       w9, w9, #1          \n"
        "bne        0b                  \n"
        :
        : "r"(loopCounts)
        : "cc", "memory", "w9", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
    );
#endif
#endif
}

void cpuFLOPSPerformance() {
    int32_t loopCounts = 100000000;
    MNN_PRINT("CPU PERFORMANCE -> loopCounts : %d \n", loopCounts);

    std::vector<int> freqVector;
    for (int i = 0; i < getCpuCounts(); i++) {
        freqVector.clear();
        getFreqKhz(i, freqVector);
        MNN_PRINT("core %d : max : %d, min : %d \n",i, freqVector.at(0), freqVector.at(1));
    }

    // warm up
    cpuFloatMlaTest(loopCounts);

    Timer timeInstance;
    timeInstance.startTimer();
    cpuFloatMlaTest(loopCounts);
#ifdef MNN_USE_NEON
#ifndef __aarch64__
    auto number = (double)loopCounts * 13;
#else 
    auto number = (double)loopCounts * 25;
#endif
#else
    auto number = 0.0;
#endif
    //FUNC_PRINT(number);
    float costTime_ms = timeInstance.getCostTimer();
    double costTime_s = (double)(costTime_ms) / 1000000.0f;
    // MNN_PRINT("cost time : %f \n", costTime_s);
    double mlaCounts_g = number * 4 / 1000000000.0f;
    float gflops       = mlaCounts_g / costTime_s;
    MNN_PRINT(" ======================== float ===============================\n");
    MNN_PRINT("CPU float gflops : %f\n", gflops);
}

static void _testMemcpy() {
    int size = 1024 * 1024;
    int loop = 10000;
    std::vector<std::thread> threads;
    MNN::Timer _t;
    for (int i=0; i<2; ++i) {
        threads.emplace_back(std::thread([size, loop]() {
            std::vector<int8_t> tmp0(size);
            std::vector<int8_t> tmp1(size);
            auto t0 = tmp0.data();
            auto t1 = tmp1.data();
            for (int i=0; i<loop; ++i) {
                ::memcpy(t0, t1, size);
                auto s = t0;
                t0 = t1;
                t1 = s;
            }
        }));
    }
    for (auto& t : threads) {
        t.join();
    }
    float timeInS = (float)_t.durationInUs() / 1000.0f / 1000.0f;
    float speed = (float)size * (float)threads.size() / 1024.0f / 1024.0f / 1024.0f * (float)loop / timeInS;
    MNN_PRINT("Memcpy speed: %f GB / s\n", speed);

}
int main(int argc, const char* argv[]) {
    MNN_PRINT("Start PERFORMANCE !!! \n");

    cpuFLOPSPerformance();
    _testMemcpy();
    
    return 0;
}
