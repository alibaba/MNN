//
//  CPURuntime.cpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 Ref from https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp
 */
#ifdef __ANDROID__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if __APPLE__
#if TARGET_OS_IPHONE
#define __IOS__ 1
#endif
#endif
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#include <stdio.h>
#include <string.h>
#include <vector>
#include "CPURuntime.hpp"
#include "MNNDefine.h"

#ifdef __ANDROID__
static int getNumberOfCPU() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }
    int number = 0;
    char buffer[1024];
    while (!feof(fp)) {
        char* str = fgets(buffer, 1024, fp);
        if (!str) {
            break;
        }
        if (memcmp(buffer, "processor", 9) == 0) {
            number++;
        }
    }
    fclose(fp);
    if (number < 1) {
        number = 1;
    }
    return number;
}

static int getCPUMaxFreqKHz(int cpuID) {
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuID);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuID);
        fp = fopen(path, "rb");
        if (!fp) {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuID);
            fp = fopen(path, "rb");
            if (!fp) {
                return -1;
            }
            int maxfrequency = -1;
            fscanf(fp, "%d", &maxfrequency);
            fclose(fp);
            return maxfrequency;
        }
    }
    int maxfrequency = 0;
    while (!feof(fp)) {
        int frequency = 0;
        int history   = fscanf(fp, "%d %*d", &frequency);
        if (history != 1) {
            break;
        }
        if (frequency > maxfrequency) {
            maxfrequency = frequency;
        }
    }
    fclose(fp);
    return maxfrequency;
}

static int sortCPUIDByMaxFrequency(std::vector<int>& cpuIDs, int* littleClusterOffset) {
    const int cpuNumbers = cpuIDs.size();
    *littleClusterOffset = 0;
    if (cpuNumbers == 0) {
        return 0;
    }
    std::vector<int> cpusFrequency;
    cpusFrequency.resize(cpuNumbers);
    for (int i = 0; i < cpuNumbers; ++i) {
        int frequency    = getCPUMaxFreqKHz(i);
        cpuIDs[i]        = i;
        cpusFrequency[i] = frequency;
        // MNN_PRINT("cpu fre: %d, %d\n", i, frequency);
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        for (int j = i + 1; j < cpuNumbers; ++j) {
            if (cpusFrequency[i] < cpusFrequency[j]) {
                // id
                int temp  = cpuIDs[i];
                cpuIDs[i] = cpuIDs[j];
                cpuIDs[j] = temp;
                // frequency
                temp             = cpusFrequency[i];
                cpusFrequency[i] = cpusFrequency[j];
                cpusFrequency[j] = temp;
            }
        }
    }
    int midMaxFrequency = (cpusFrequency.front() + cpusFrequency.back()) / 2;
    if (midMaxFrequency == cpusFrequency.back()) {
        return 0;
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        if (cpusFrequency[i] < midMaxFrequency) {
            *littleClusterOffset = i;
            break;
        }
    }
    return 0;
}

static int setSchedAffinity(const std::vector<int>& cpuIDs) {
#define CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(unsigned long))
    typedef struct {
        unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

#define CPU_SET(cpu, cpusetp) ((cpusetp)->__bits[(cpu) / __NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
#ifdef __GLIBC__
    pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
    pid_t pid = getpid();
#else
    pid_t pid = gettid();
#endif
#endif
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 0; i < (int)cpuIDs.size(); i++) {
        CPU_SET(cpuIDs[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret) {
        MNN_PRINT("syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}

#endif // arch

int MNNSetCPUThreadsMode(MNNCPUThreadsMode mode) {
#ifdef __ANDROID__
    auto numberOfCPUs = getNumberOfCPU();
    if (mode == MNN_CPU_MODE_DEFAULT) {
        return 0;
    }
    static std::vector<int> sortedCPUIDs;
    static int littleClusterOffset = 0;
    if (sortedCPUIDs.empty()) {
        sortedCPUIDs.resize(numberOfCPUs);
        for (int i = 0; i < numberOfCPUs; ++i) {
            sortedCPUIDs[i] = i;
        }
        sortCPUIDByMaxFrequency(sortedCPUIDs, &littleClusterOffset);
    }

    if (littleClusterOffset == 0 && mode != MNN_CPU_MODE_POWER_FRI) {
        MNN_PRINT("This CPU Arch Do NOT support for setting cpu thread mode\n");
    }
    std::vector<int> cpuAttachIDs;
    switch (mode) {
        case MNN_CPU_MODE_POWER_FRI:
            cpuAttachIDs = sortedCPUIDs;
            break;
        case MNN_CPU_MODE_LITTLE:
            cpuAttachIDs = std::vector<int>(sortedCPUIDs.begin() + littleClusterOffset, sortedCPUIDs.end());
            break;
        case MNN_CPU_MODE_BIG:
            cpuAttachIDs = std::vector<int>(sortedCPUIDs.begin(), sortedCPUIDs.begin() + littleClusterOffset);
            break;
        default:
            cpuAttachIDs = sortedCPUIDs;
            break;
    }

#ifdef _OPENMP
    const int threadsNumber = cpuAttachIDs.size();
    omp_set_num_threads(threadsNumber);
    std::vector<int> result(threadsNumber, 0);
#pragma omp parallel for
    for (int i = 0; i < threadsNumber; ++i) {
        result[i] = setSchedAffinity(cpuAttachIDs);
    }
    for (int i = 0; i < threadsNumber; ++i) {
        if (result[i] != 0) {
            return -1;
        }
    }
#else
    int res   = setSchedAffinity(cpuAttachIDs);
    if (res != 0) {
        return -1;
    }
#endif // _OPENMP
    return 0;
#elif __IOS__
    return -1;
#else
    return -1;
#endif // arch
}
