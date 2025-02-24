#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include "evaluation/evaluation.hpp"
#include "evaluation/MemMonitor.hpp"


#if defined(__ANDROID__) || defined(linux) || defined(__APPLE__) || defined(__MACOSX)

int readMemInfo(MemoryInfo *mem_info) {
    FILE *file;
    char line[BUFFER_SIZE];
    char key[BUFFER_SIZE];
    long value;

    // Read /proc/meminfo
    file = fopen(MEMINFO_FILE, "r");
    if (!file) {
        perror("fopen /proc/meminfo");
        return -1;
    }

    while (fgets(line, BUFFER_SIZE, file)) {
        if (sscanf(line, "%s %ld", key, &value) == 2) {
            if (!strcmp(key, "MemTotal:")) mem_info->total_phys_mem = (float)value*KILLO_TO_MEGA;
            else if (!strcmp(key, "MemFree:")) mem_info->free_phys_mem = (float)value*KILLO_TO_MEGA;
            else if (!strcmp(key, "SwapTotal:")) mem_info->total_swap = (float)value*KILLO_TO_MEGA;
            else if (!strcmp(key, "SwapFree:")) mem_info->free_swap = (float)value*KILLO_TO_MEGA;
        }
    }
    fclose(file);

    return 0;
}

int readProcStatus(MemoryInfo *mem_info) {
    FILE *file;
    char line[BUFFER_SIZE];
    char key[BUFFER_SIZE];
    long value;


    file = fopen(SELF_FILE, "r");
    if (!file) {
        perror("fopen /proc/self/status");
        return -1;
    }

    while (fgets(line, BUFFER_SIZE, file)) {
        if (sscanf(line, "%s %ld", key, &value) == 2) {
            if (!strcmp(key, "VmRSS:")) mem_info->process_resident_set_size = (float)value*KILLO_TO_MEGA;
            else if (!strcmp(key, "VmSwap:")) mem_info->process_swap = (float)value*KILLO_TO_MEGA;
            else if (!strcmp(key, "VmSize:")) mem_info->process_virtual_mem_total = (float)value*KILLO_TO_MEGA;
            else if (!strcmp(key, "VmHWM:")) mem_info->process_virtual_mem_used = (float)value*KILLO_TO_MEGA;
        }
    }
    fclose(file);

    return 0;
}

#else // linux

#if defined(_WIN32) || defined(_WIN64)

#include <processthreadsapi.h>
#include <psapi.h>

int readMemInfo(MemoryInfo *mem_info) {}

int readProcStatus(MemoryInfo *mem_info) {
    HANDLE hProcess = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;

    GetProcessMemoryInfo( hProcess, &pmc, sizeof(pmc));

    mem_info->process_resident_set_size = (float)pmc.WorkingSetSize*BYTE_TO_MEGA;
}

#else // windows

// do nothing
int readMemInfo(MemoryInfo *mem_info) {}
int readProcStatus(MemoryInfo *mem_info) {}

#endif 

#endif 

void printMemoryInfo(const MemoryInfo *mem_info) {
    printf("Total Physical Memory:     %f MB\n", mem_info->total_phys_mem);
    printf("Available Physical Memory: %f MB\n", mem_info->free_phys_mem);
    printf("Total Swap:                %f MB\n", mem_info->total_swap);
    printf("Available Swap:            %f MB\n", mem_info->free_swap);
    printf("Process Resident Set Size: %f MB\n", mem_info->process_resident_set_size);
    printf("Process Swap:              %f MB\n", mem_info->process_swap);
    printf("Process Total Virtual Mem: %f MB\n", mem_info->process_virtual_mem_total);
    printf("Process Used Virtual Mem:  %f MB\n", mem_info->process_virtual_mem_used);
}

float getSysMemInc(MemoryInfo* prev, MemoryInfo* now) {
    float prev_free = prev->free_phys_mem + prev->free_swap;
    float now_free = now->free_phys_mem + now->free_swap;
    return (prev_free - now_free);
}

float getProcMem(MemoryInfo* info) {
    return (info->process_resident_set_size + info->process_swap);
}