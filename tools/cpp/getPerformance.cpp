//
//  getPerformance.cpp
//  MNN
//
//  Created by MNN on 2019/03/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <arm_neon.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <vector>
#include "MNNDefine.h"
#include "core/OpenCLBackend.hpp"
#include "core/OpenCLRunningUtils.hpp"

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

void cpuMlaTest(uint64_t loopCounts) {
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
}

void cpuFLOPSPerformance() {
    uint64_t loopCounts = 10000000;
    int threadCounts    = getCpuCounts();

    MNN_PRINT("CPU PERFORMANCE -> loopCounts : %lu , threadCounts : %d \n", loopCounts, threadCounts);

    std::vector<int> freqVector;
    for (int i = 0; i < getCpuCounts(); i++) {
        getFreqKhz(i, freqVector);
        // MNN_PRINT("core %d : max : %d, min : %d \n",i, freqVector.at(0), freqVector.at(1));
    }

    // warm up
    cpuMlaTest(loopCounts);

    Timer timeInstance;
    timeInstance.startTimer();

    cpuMlaTest(loopCounts);

    float costTime_ms = timeInstance.getCostTimer();
    double costTime_s = (double)(costTime_ms) / 1000000.0f;
    // MNN_PRINT("cost time : %f \n", costTime_s);
    double mlaCounts_g = loopCounts * 6 * 4 / 1000000000.0f;
    float gflops       = mlaCounts_g / costTime_s;
    getFreqKhz(0, freqVector);
    MNN_PRINT("CPU gflops : %f , max freq gkhz : %f \n", gflops, (float)freqVector.at(0) / 1000000.0f);
}

float getTimeFromEvent(cl::Event& event) {
    int64_t start    = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int64_t end      = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    int64_t costTime = start - end;
    MNN_PRINT("cost time : %ld \n", costTime);
    return (float)((int)costTime);
}

float run_kernel(cl::CommandQueue& queue, cl::Kernel& kernel, cl::NDRange& globalSize, cl::NDRange& localSize,
                 int loopCounts) {
    float costTime = 0;

    // warm up
    for (uint i = 0; i < loopCounts / 4; i++) {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    }

    queue.finish();
    Timer time;
    time.startTimer();
    cl_int error = CL_SUCCESS;
    cl::Event event;
    for (uint i = 0; i < loopCounts; i++) {
        error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &event);
    }
    event.wait();
    MNN_CHECK_CL_SUCCESS(error);
    costTime = time.getCostTimer();
    return costTime;
}

void gpuFLOPSPerformance() {
    uint64_t loopCounts = 10;
    MNN_PRINT("GPU PERFORMANCE -> loopCounts : %lu \n", loopCounts);
    using namespace MNN;
    // symbol load
    OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance();
    if (nullptr == OpenCLSymbolsOperator::getOpenclSymbolsPtr()) {
        MNN_PRINT("OpenCL init error , callback ...");
    }
    std::shared_ptr<OpenCLRuntime> openclBackendInstance;
    openclBackendInstance.reset(new OpenCLRuntime());
    if (false == openclBackendInstance.get()->isSupportedFP16()) {
        MNN_PRINT("OpenCL init error , callback ...");
    }
    std::shared_ptr<OpenCLBackend> openCLBackend;
    openCLBackend.reset(new OpenCLBackend());

    std::set<std::string> buildOptions;
    auto runtime = openCLBackend->getOpenCLRuntime();

    /////////////////////////////////////// float mad ///////////////////////////////////////
    {
        cl::Kernel kernel = runtime->buildKernel("performance", "float_precision", buildOptions);
        uint64_t wsPCU    = 8 * 4 * 256;
        // uint32_t max_wgs = runtime->getMaxWorkGroupSize(kernel);
        uint32_t max_wgs          = 128;
        int64_t global_size_int64 = runtime->deviceComputeUnits() * wsPCU * max_wgs;
        // MNN_PRINT("global_size_int64 : %lld , max_wgs : %d , cu : %d \n", global_size_int64, max_wgs,
        // runtime->deviceComputeUnits());
        int64_t local_size_int64 = max_wgs;

        cl::NDRange global_size = (global_size_int64 / local_size_int64) * local_size_int64;
        cl::NDRange local_size  = max_wgs;

        std::shared_ptr<cl::Buffer> outputBuffer;
        outputBuffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                          (global_size_int64 * sizeof(cl_float))));

        cl_float mul_value = 1.3f;
        kernel.setArg(0, *outputBuffer);
        kernel.setArg(1, mul_value);

        float cost_time = run_kernel(runtime->commandQueue(), kernel, global_size, local_size, loopCounts);

        float costTime_s   = cost_time / 1000000.0f;
        uint64_t madCounts = global_size_int64 * wsPCU / 1000000000;
        float gflops       = (float)madCounts / costTime_s;

        // MNN_PRINT("costTime_s : %f , madCounts : %lld G\n", costTime_s, madCounts);
        MNN_PRINT("GPU float mad gflops : %f , max freq : %f MHZ\n", gflops, (float)runtime->maxFreq() * 100);
    }

    /////////////////////////////////////// half4 mad ///////////////////////////////////////

    {
        cl::Kernel kernel = runtime->buildKernel("performance", "half4_precision", buildOptions);
        uint32_t wsPCU    = 8 * 2 * 256;
        // uint32_t max_wgs = runtime->getMaxWorkGroupSize(kernel);
        uint32_t max_wgs          = 128;
        int64_t global_size_int64 = runtime->deviceComputeUnits() * wsPCU * max_wgs;
        // MNN_PRINT("global_size_int64 : %lld , max_wgs : %d , cu : %d \n", global_size_int64, max_wgs,
        // runtime->deviceComputeUnits());
        int64_t local_size_int64 = max_wgs;

        cl::NDRange global_size = (global_size_int64 / local_size_int64) * local_size_int64;
        cl::NDRange local_size  = max_wgs;

        std::shared_ptr<cl::Buffer> outputBuffer;
        outputBuffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                          (global_size_int64 * sizeof(cl_half))));

        cl_float mul_value = 1.3f;
        kernel.setArg(0, *outputBuffer);
        kernel.setArg(1, mul_value);

        float cost_time = run_kernel(runtime->commandQueue(), kernel, global_size, local_size, loopCounts);

        float costTime_s   = cost_time / 1000000.0f;
        uint64_t madCounts = global_size_int64 * wsPCU * 4 / 1000000000;
        float gflops       = (float)madCounts / costTime_s;

        // MNN_PRINT("costTime_s : %f , madCounts : %lld G\n", costTime_s, madCounts);
        MNN_PRINT("GPU half4 mad gflops : %f , max freq : %f MHZ\n", gflops, (float)runtime->maxFreq() * 100);
    }
}

int main(int argc, const char* argv[]) {
    MNN_PRINT("Start PERFORMANCE !!! \n");

    // cpuFLOPSPerformance();
    gpuFLOPSPerformance();

    return 0;
}
