#ifndef MEMMONITOR_hpp
#define MEMMONITOR_hpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#define BUFFER_SIZE 256

struct MemoryInfo {
    // in MB
    float total_phys_mem;
    float free_phys_mem;
    float total_swap;
    float free_swap;
    float process_resident_set_size;
    float process_swap;
    float process_virtual_mem_total;
    float process_virtual_mem_used;
};


#if defined(__ANDROID__) || defined(linux) || defined(__APPLE__) || defined(__MACOSX)
#define SELF_FILE "/proc/self/status"
#define MEMINFO_FILE "/proc/meminfo"
#endif // linux

int readMemInfo(MemoryInfo *mem_info);

int readProcStatus(MemoryInfo *mem_info);

void printMemoryInfo(const MemoryInfo *mem_info);

float getSysMemInc(MemoryInfo* prev, MemoryInfo* now);

float getProcMem(MemoryInfo* info);

#endif