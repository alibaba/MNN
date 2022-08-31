//
//  CPURuntime.cpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 Ref from:
 https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp
 https://github.com/pytorch/cpuinfo
 */
#ifdef __ANDROID__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "core/Macro.h"
#ifdef MNN_USE_ARMV82

#ifdef __ANDROID__
#include <fcntl.h>
#include <sys/auxv.h>
#include <sys/system_properties.h>
#endif // __ANDROID__

#endif // MNN_USE_ARMV82

#if __APPLE__
#include "TargetConditionals.h"
#if __aarch64__
#include <sys/sysctl.h>
#endif
#if TARGET_OS_IPHONE
#include <mach/machine.h>
#include <sys/types.h>
#define __IOS__ 1
#endif // TARGET_OS_IPHONE
#endif // __APPLE__

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#include <MNN/MNNDefine.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "backend/cpu/CPURuntime.hpp"

#if defined (__linux__) && defined (__aarch64__)
#include <sys/auxv.h>

#define CPUINFO_ARM_LINUX_FEATURE_FPHP       UINT32_C(0x00000200)
#define CPUINFO_ARM_LINUX_FEATURE_ASIMDHP    UINT32_C(0x00000400)
#define CPUINFO_ARM_LINUX_FEATURE_ASIMDDP  UINT32_C(0x00100000)
#endif /* __linux__ && __aarch64__ */

#ifdef __ANDROID__

/* As per include/sys/system_properties.h in Android NDK */
#define CPUINFO_HARDWARE_VALUE_MAX 64
#define CPUINFO_BUILD_PROP_VALUE_MAX 92

struct cpuinfo_android_properties {
    char proc_cpuinfo_hardware[CPUINFO_HARDWARE_VALUE_MAX];
    char ro_product_board[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_board_platform[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_mediatek_platform[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_arch[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_chipname[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_hardware_chipname[CPUINFO_BUILD_PROP_VALUE_MAX];
};

enum cpuinfo_android_chipset_property {
    cpuinfo_android_chipset_property_proc_cpuinfo_hardware = 0,
    cpuinfo_android_chipset_property_ro_product_board,
    cpuinfo_android_chipset_property_ro_board_platform,
    cpuinfo_android_chipset_property_ro_mediatek_platform,
    cpuinfo_android_chipset_property_ro_arch,
    cpuinfo_android_chipset_property_ro_chipname,
    cpuinfo_android_chipset_property_ro_hardware_chipname,
    cpuinfo_android_chipset_property_max,
};

enum cpuinfo_arm_chipset_vendor {
    cpuinfo_arm_chipset_vendor_unknown = 0,
    cpuinfo_arm_chipset_vendor_qualcomm,
    cpuinfo_arm_chipset_vendor_mediatek,
    cpuinfo_arm_chipset_vendor_samsung,
    cpuinfo_arm_chipset_vendor_hisilicon,
    cpuinfo_arm_chipset_vendor_actions,
    cpuinfo_arm_chipset_vendor_allwinner,
    cpuinfo_arm_chipset_vendor_amlogic,
    cpuinfo_arm_chipset_vendor_broadcom,
    cpuinfo_arm_chipset_vendor_lg,
    cpuinfo_arm_chipset_vendor_leadcore,
    cpuinfo_arm_chipset_vendor_marvell,
    cpuinfo_arm_chipset_vendor_mstar,
    cpuinfo_arm_chipset_vendor_novathor,
    cpuinfo_arm_chipset_vendor_nvidia,
    cpuinfo_arm_chipset_vendor_pinecone,
    cpuinfo_arm_chipset_vendor_renesas,
    cpuinfo_arm_chipset_vendor_rockchip,
    cpuinfo_arm_chipset_vendor_spreadtrum,
    cpuinfo_arm_chipset_vendor_telechips,
    cpuinfo_arm_chipset_vendor_texas_instruments,
    cpuinfo_arm_chipset_vendor_wondermedia,
    cpuinfo_arm_chipset_vendor_max,
};

enum cpuinfo_arm_chipset_series {
    cpuinfo_arm_chipset_series_unknown = 0,
    cpuinfo_arm_chipset_series_qualcomm_qsd,
    cpuinfo_arm_chipset_series_qualcomm_msm,
    cpuinfo_arm_chipset_series_qualcomm_apq,
    cpuinfo_arm_chipset_series_qualcomm_snapdragon,
    cpuinfo_arm_chipset_series_mediatek_mt,
    cpuinfo_arm_chipset_series_samsung_exynos,
    cpuinfo_arm_chipset_series_hisilicon_k3v,
    cpuinfo_arm_chipset_series_hisilicon_hi,
    cpuinfo_arm_chipset_series_hisilicon_kirin,
    cpuinfo_arm_chipset_series_actions_atm,
    cpuinfo_arm_chipset_series_allwinner_a,
    cpuinfo_arm_chipset_series_amlogic_aml,
    cpuinfo_arm_chipset_series_amlogic_s,
    cpuinfo_arm_chipset_series_broadcom_bcm,
    cpuinfo_arm_chipset_series_lg_nuclun,
    cpuinfo_arm_chipset_series_leadcore_lc,
    cpuinfo_arm_chipset_series_marvell_pxa,
    cpuinfo_arm_chipset_series_mstar_6a,
    cpuinfo_arm_chipset_series_novathor_u,
    cpuinfo_arm_chipset_series_nvidia_tegra_t,
    cpuinfo_arm_chipset_series_nvidia_tegra_ap,
    cpuinfo_arm_chipset_series_nvidia_tegra_sl,
    cpuinfo_arm_chipset_series_pinecone_surge_s,
    cpuinfo_arm_chipset_series_renesas_mp,
    cpuinfo_arm_chipset_series_rockchip_rk,
    cpuinfo_arm_chipset_series_spreadtrum_sc,
    cpuinfo_arm_chipset_series_telechips_tcc,
    cpuinfo_arm_chipset_series_texas_instruments_omap,
    cpuinfo_arm_chipset_series_wondermedia_wm,
    cpuinfo_arm_chipset_series_max,
};

struct cpuinfo_arm_chipset {
    enum cpuinfo_arm_chipset_vendor vendor;
    enum cpuinfo_arm_chipset_series series;
    uint32_t model;
    char suffix[8];
};

#define BUFFER_SIZE 1024

static uint32_t getNumberOfCPU() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }
    uint32_t number = 0;
    char buffer[BUFFER_SIZE];
    while (!feof(fp)) {
        char* str = fgets(buffer, BUFFER_SIZE, fp);
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
float MNNGetCPUFlops(uint32_t number) {
    float flops = 2048.0f;
#ifdef __ANDROID__
    auto numberOfCPUs = getNumberOfCPU();
    if (0 == numberOfCPUs) {
        return flops;
    }
    std::vector<int> freqs;
    freqs.resize(numberOfCPUs);
    for (int i = 0; i < numberOfCPUs; ++i) {
        freqs[i] = getCPUMaxFreqKHz(i);
    }
    std::sort(freqs.rbegin(), freqs.rend());
    number = std::min(number, numberOfCPUs);
    flops  = 0.0f;
    for (uint32_t i = 0; i < number; ++i) {
        flops += (float)freqs[i] / 1024.0f;
    }
#endif
    return flops;
}

// cpuinfo
// Reference from: https://github.com/pytorch/cpuinfo

#ifdef MNN_USE_ARMV82

#ifdef __ANDROID__

#define CPUINFO_ARM_MIDR_IMPLEMENTER_MASK UINT32_C(0xFF000000)
#define CPUINFO_ARM_MIDR_VARIANT_MASK UINT32_C(0x00F00000)
#define CPUINFO_ARM_MIDR_ARCHITECTURE_MASK UINT32_C(0x000F0000)
#define CPUINFO_ARM_MIDR_PART_MASK UINT32_C(0x0000FFF0)
#define CPUINFO_ARM_MIDR_REVISION_MASK UINT32_C(0x0000000F)

#define CPUINFO_ARM_LINUX_VALID_ARCHITECTURE UINT32_C(0x00010000)
#define CPUINFO_ARM_LINUX_VALID_IMPLEMENTER UINT32_C(0x00020000)
#define CPUINFO_ARM_LINUX_VALID_VARIANT UINT32_C(0x00040000)
#define CPUINFO_LINUX_FLAG_VALID UINT32_C(0x00001000)
#define CPUINFO_ARM_LINUX_VALID_MIDR UINT32_C(0x003F0000)
#define CPUINFO_ARM_LINUX_VALID_PART UINT32_C(0x00080000)
#define CPUINFO_ARM_LINUX_VALID_PROCESSOR UINT32_C(0x00200000)
#define CPUINFO_ARM_LINUX_VALID_REVISION UINT32_C(0x00100000)

#define CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET 24
#define CPUINFO_ARM_MIDR_VARIANT_OFFSET 20
#define CPUINFO_ARM_MIDR_ARCHITECTURE_OFFSET 16
#define CPUINFO_ARM_MIDR_PART_OFFSET 4
#define CPUINFO_ARM_MIDR_REVISION_OFFSET 0

#ifdef __aarch64__
#define CPUINFO_ARM_LINUX_FEATURE_FPHP UINT32_C(0x00000200)
#define CPUINFO_ARM_LINUX_FEATURE_ASIMDHP UINT32_C(0x00000400)
#define CPUINFO_ARM_LINUX_FEATURE_ASIMDDP UINT32_C(0x00100000)
#else
#define CPUINFO_ARM_LINUX_FEATURE_HALF     UINT32_C(0x00000002)
#define CPUINFO_ARM_LINUX_FEATURE_NEON     UINT32_C(0x00001000)
#endif

struct cpuinfo_arm_linux_processor {
    uint32_t architecture_version;
    // Main ID Register value
    uint32_t midr;

    uint32_t max_frequency;
    uint32_t min_frequency;

    uint32_t system_processor_id;
    uint32_t flags;
};

struct proc_cpuinfo_parser_state {
    char* hardware;
    uint32_t processor_index;
    uint32_t max_processors_count;
    struct cpuinfo_arm_linux_processor* processors;
    struct cpuinfo_arm_linux_processor dummy_processor;
};

typedef bool (*cpuinfo_line_callback)(const char*, const char*, void*, uint64_t);

inline static uint32_t midr_set_implementer(uint32_t midr, uint32_t implementer) {
    return (midr & ~CPUINFO_ARM_MIDR_IMPLEMENTER_MASK) |
           ((implementer << CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET) & CPUINFO_ARM_MIDR_IMPLEMENTER_MASK);
}

inline static uint32_t midr_set_architecture(uint32_t midr, uint32_t architecture) {
    return (midr & ~CPUINFO_ARM_MIDR_ARCHITECTURE_MASK) |
           ((architecture << CPUINFO_ARM_MIDR_ARCHITECTURE_OFFSET) & CPUINFO_ARM_MIDR_ARCHITECTURE_MASK);
}

inline static uint32_t midr_set_part(uint32_t midr, uint32_t part) {
    return (midr & ~CPUINFO_ARM_MIDR_PART_MASK) | ((part << CPUINFO_ARM_MIDR_PART_OFFSET) & CPUINFO_ARM_MIDR_PART_MASK);
}

inline static uint32_t midr_set_revision(uint32_t midr, uint32_t revision) {
    return (midr & ~CPUINFO_ARM_MIDR_REVISION_MASK) |
           ((revision << CPUINFO_ARM_MIDR_REVISION_OFFSET) & CPUINFO_ARM_MIDR_REVISION_MASK);
}

inline static uint32_t midr_set_variant(uint32_t midr, uint32_t variant) {
    return (midr & ~CPUINFO_ARM_MIDR_VARIANT_MASK) |
           ((variant << CPUINFO_ARM_MIDR_VARIANT_OFFSET) & CPUINFO_ARM_MIDR_VARIANT_MASK);
}

inline static uint32_t midr_get_variant(uint32_t midr) {
    return (midr & CPUINFO_ARM_MIDR_VARIANT_MASK) >> CPUINFO_ARM_MIDR_VARIANT_OFFSET;
}

static inline bool bitmask_all(uint32_t bitfield, uint32_t mask) {
    return (bitfield & mask) == mask;
}

static void parse_cpu_part(const char* cpu_part_start, const char* cpu_part_end,
                           struct cpuinfo_arm_linux_processor* processor) {
    const size_t cpu_part_length = (size_t)(cpu_part_end - cpu_part_start);

    /*
     * CPU part should contain hex prefix (0x) and one to three hex digits.
     * I have never seen less than three digits as a value of this field,
     * but I don't think it is impossible to see such values in future.
     * Value can not contain more than three hex digits since
     * Main ID Register (MIDR) assigns only a 12-bit value for CPU part.
     */
    if (cpu_part_length < 3 || cpu_part_length > 5) {
        MNN_PRINT("CPU part %.*s in /proc/cpuinfo is ignored due to unexpected length (%zu)\n", (int)cpu_part_length,
                  cpu_part_start, cpu_part_length);
        return;
    }

    /* Verify the presence of hex prefix */
    if (cpu_part_start[0] != '0' || cpu_part_start[1] != 'x') {
        MNN_PRINT("CPU part %.*s in /proc/cpuinfo is ignored due to lack of 0x prefix\n", (int)cpu_part_length,
                  cpu_part_start);
        return;
    }

    /* Verify that characters after hex prefix are hexadecimal digits and decode them */
    uint32_t cpu_part = 0;
    for (const char* digit_ptr = cpu_part_start + 2; digit_ptr != cpu_part_end; digit_ptr++) {
        const char digit_char = *digit_ptr;
        uint32_t digit;
        if (digit_char >= '0' && digit_char <= '9') {
            digit = digit_char - '0';
        } else if ((uint32_t)(digit_char - 'A') < 6) {
            digit = 10 + (digit_char - 'A');
        } else if ((uint32_t)(digit_char - 'a') < 6) {
            digit = 10 + (digit_char - 'a');
        } else {
            MNN_PRINT("CPU part %.*s in /proc/cpuinfo is ignored due to unexpected non-hex character %c at offset %zu\n",
                      (int)cpu_part_length, cpu_part_start, digit_char, (size_t)(digit_ptr - cpu_part_start));
            return;
        }
        cpu_part = cpu_part * 16 + digit;
    }

    processor->midr = midr_set_part(processor->midr, cpu_part);
    processor->flags |= CPUINFO_ARM_LINUX_VALID_PART | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
}

static void parse_cpu_revision(const char* cpu_revision_start, const char* cpu_revision_end,
                               struct cpuinfo_arm_linux_processor* processor) {
    uint32_t cpu_revision = 0;
    for (const char* digit_ptr = cpu_revision_start; digit_ptr != cpu_revision_end; digit_ptr++) {
        const uint32_t digit = (uint32_t)(*digit_ptr - '0');

        /* Verify that the character in CPU revision is a decimal digit */
        if (digit >= 10) {
            MNN_PRINT(
                "CPU revision %.*s in /proc/cpuinfo is ignored due to unexpected non-digit character '%c' at offset "
                "%zu\n",
                (int)(cpu_revision_end - cpu_revision_start), cpu_revision_start, *digit_ptr,
                (size_t)(digit_ptr - cpu_revision_start));
            return;
        }

        cpu_revision = cpu_revision * 10 + digit;
    }

    processor->midr = midr_set_revision(processor->midr, cpu_revision);
    processor->flags |= CPUINFO_ARM_LINUX_VALID_REVISION | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
}

static void parse_cpu_architecture(const char* cpu_architecture_start, const char* cpu_architecture_end,
                                   struct cpuinfo_arm_linux_processor* processor) {
    const size_t cpu_architecture_length = (size_t)(cpu_architecture_end - cpu_architecture_start);
    /* Early AArch64 kernels report "CPU architecture: AArch64" instead of a numeric value 8 */
    if (cpu_architecture_length == 7) {
        if (memcmp(cpu_architecture_start, "AArch64", cpu_architecture_length) == 0) {
            processor->midr                 = midr_set_architecture(processor->midr, UINT32_C(0xF));
            processor->architecture_version = 8;
            processor->flags |= CPUINFO_ARM_LINUX_VALID_ARCHITECTURE | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
            return;
        }
    }

    uint32_t architecture            = 0;
    const char* cpu_architecture_ptr = cpu_architecture_start;
    for (; cpu_architecture_ptr != cpu_architecture_end; cpu_architecture_ptr++) {
        const uint32_t digit = (*cpu_architecture_ptr) - '0';

        /* Verify that CPU architecture is a decimal number */
        if (digit >= 10) {
            break;
        }

        architecture = architecture * 10 + digit;
    }

    if (cpu_architecture_ptr == cpu_architecture_start) {
        MNN_PRINT("CPU architecture %.*s in /proc/cpuinfo is ignored due to non-digit at the beginning of the string\n",
                  (int)cpu_architecture_length, cpu_architecture_start);
    } else {
        if (architecture != 0) {
            processor->architecture_version = architecture;
            processor->flags |= CPUINFO_ARM_LINUX_VALID_ARCHITECTURE | CPUINFO_ARM_LINUX_VALID_PROCESSOR;

            for (; cpu_architecture_ptr != cpu_architecture_end; cpu_architecture_ptr++) {
                const char feature = *cpu_architecture_ptr;
                switch (feature) {
                    case ' ':
                    case '\t':
                        /* Ignore whitespace at the end */
                        break;
                    default:
                        MNN_PRINT("skipped unknown architectural feature '%c' for ARMv%u\n", feature, architecture);
                        break;
                }
            }
        } else {
            MNN_PRINT("CPU architecture %.*s in /proc/cpuinfo is ignored due to invalid value (0)\n",
                      (int)cpu_architecture_length, cpu_architecture_start);
        }
    }

    uint32_t midr_architecture = UINT32_C(0xF);
    processor->midr            = midr_set_architecture(processor->midr, midr_architecture);
}

static uint32_t parse_processor_number(const char* processor_start, const char* processor_end) {
    const size_t processor_length = (size_t)(processor_end - processor_start);

    if (processor_length == 0) {
        MNN_PRINT("Processor number in /proc/cpuinfo is ignored: string is empty\n");
        return 0;
    }

    uint32_t processor_number = 0;
    for (const char* digit_ptr = processor_start; digit_ptr != processor_end; digit_ptr++) {
        const uint32_t digit = (uint32_t)(*digit_ptr - '0');
        if (digit > 10) {
            MNN_PRINT("non-decimal suffix %.*s in /proc/cpuinfo processor number is ignored\n",
                      (int)(processor_end - digit_ptr), digit_ptr);
            break;
        }

        processor_number = processor_number * 10 + digit;
    }

    return processor_number;
}

static void parse_cpu_variant(const char* cpu_variant_start, const char* cpu_variant_end,
                              struct cpuinfo_arm_linux_processor* processor) {
    const size_t cpu_variant_length = cpu_variant_end - cpu_variant_start;

    /*
     * Value should contain hex prefix (0x) and one hex digit.
     * Value can not contain more than one hex digits since
     * Main ID Register (MIDR) assigns only a 4-bit value for CPU variant.
     */
    if (cpu_variant_length != 3) {
        MNN_PRINT("CPU variant %.*s in /proc/cpuinfo is ignored due to unexpected length (%zu)\n",
                  (int)cpu_variant_length, cpu_variant_start, cpu_variant_length);
        return;
    }

    /* Skip if there is no hex prefix (0x) */
    if (cpu_variant_start[0] != '0' || cpu_variant_start[1] != 'x') {
        MNN_PRINT("CPU variant %.*s in /proc/cpuinfo is ignored due to lack of 0x prefix\n", (int)cpu_variant_length,
                  cpu_variant_start);
        return;
    }

    /* Check if the value after hex prefix is indeed a hex digit and decode it. */
    const char digit_char = cpu_variant_start[2];
    uint32_t cpu_variant;
    if ((uint32_t)(digit_char - '0') < 10) {
        cpu_variant = (uint32_t)(digit_char - '0');
    } else if ((uint32_t)(digit_char - 'A') < 6) {
        cpu_variant = 10 + (uint32_t)(digit_char - 'A');
    } else if ((uint32_t)(digit_char - 'a') < 6) {
        cpu_variant = 10 + (uint32_t)(digit_char - 'a');
    } else {
        MNN_PRINT("CPU variant %.*s in /proc/cpuinfo is ignored due to unexpected non-hex character '%c'\n",
                  (int)cpu_variant_length, cpu_variant_start, digit_char);
        return;
    }

    processor->midr = midr_set_variant(processor->midr, cpu_variant);
    processor->flags |= CPUINFO_ARM_LINUX_VALID_VARIANT | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
}

static void parse_cpu_implementer(const char* cpu_implementer_start, const char* cpu_implementer_end,
                                  struct cpuinfo_arm_linux_processor* processor) {
    const size_t cpu_implementer_length = cpu_implementer_end - cpu_implementer_start;

    /*
     * Value should contain hex prefix (0x) and one or two hex digits.
     * I have never seen single hex digit as a value of this field,
     * but I don't think it is impossible in future.
     * Value can not contain more than two hex digits since
     * Main ID Register (MIDR) assigns only an 8-bit value for CPU implementer.
     */
    switch (cpu_implementer_length) {
        case 3:
        case 4:
            break;
        default:
            MNN_PRINT("CPU implementer %.*s in /proc/cpuinfo is ignored due to unexpected length (%zu)\n",
                      (int)cpu_implementer_length, cpu_implementer_start, cpu_implementer_length);
            return;
    }

    /* Verify the presence of hex prefix */
    if (cpu_implementer_start[0] != '0' || cpu_implementer_start[1] != 'x') {
        MNN_PRINT("CPU implementer %.*s in /proc/cpuinfo is ignored due to lack of 0x prefix\n",
                  (int)cpu_implementer_length, cpu_implementer_start);
        return;
    }

    /* Verify that characters after hex prefix are hexadecimal digits and decode them */
    uint32_t cpu_implementer = 0;
    for (const char* digit_ptr = cpu_implementer_start + 2; digit_ptr != cpu_implementer_end; digit_ptr++) {
        const char digit_char = *digit_ptr;
        uint32_t digit;
        if (digit_char >= '0' && digit_char <= '9') {
            digit = digit_char - '0';
        } else if ((uint32_t)(digit_char - 'A') < 6) {
            digit = 10 + (digit_char - 'A');
        } else if ((uint32_t)(digit_char - 'a') < 6) {
            digit = 10 + (digit_char - 'a');
        } else {
            MNN_PRINT(
                "CPU implementer %.*s in /proc/cpuinfo is ignored due to unexpected non-hex character '%c' at offset "
                "%zu\n",
                (int)cpu_implementer_length, cpu_implementer_start, digit_char,
                (size_t)(digit_ptr - cpu_implementer_start));
            return;
        }
        cpu_implementer = cpu_implementer * 16 + digit;
    }

    processor->midr = midr_set_implementer(processor->midr, cpu_implementer);
    processor->flags |= CPUINFO_ARM_LINUX_VALID_IMPLEMENTER | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
}

static bool parse_line(const char* line_start, const char* line_end, struct proc_cpuinfo_parser_state* state,
                       uint64_t line_number) {
    /* Empty line. Skip. */
    if (line_start == line_end) {
        return true;
    }

    /* Search for ':' on the line. */
    const char* separator = line_start;
    for (; separator != line_end; separator++) {
        if (*separator == ':') {
            break;
        }
    }
    /* Skip line if no ':' separator was found. */
    if (separator == line_end) {
        MNN_PRINT("Line %.*s in /proc/cpuinfo is ignored: key/value separator ':' not found\n",
                  (int)(line_end - line_start), line_start);
        return true;
    }

    /* Skip trailing spaces in key part. */
    const char* key_end = separator;
    for (; key_end != line_start; key_end--) {
        if (key_end[-1] != ' ' && key_end[-1] != '\t') {
            break;
        }
    }
    /* Skip line if key contains nothing but spaces. */
    if (key_end == line_start) {
        MNN_PRINT("Line %.*s in /proc/cpuinfo is ignored: key contains only spaces\n", (int)(line_end - line_start),
                  line_start);
        return true;
    }

    /* Skip leading spaces in value part. */
    const char* value_start = separator + 1;
    for (; value_start != line_end; value_start++) {
        if (*value_start != ' ') {
            break;
        }
    }
    /* Value part contains nothing but spaces. Skip line. */
    if (value_start == line_end) {
        MNN_PRINT("Line %.*s in /proc/cpuinfo is ignored: value contains only spaces\n", (int)(line_end - line_start),
                  line_start);
        return true;
    }

    /* Skip trailing spaces in value part (if any) */
    const char* value_end = line_end;
    for (; value_end != value_start; value_end--) {
        if (value_end[-1] != ' ') {
            break;
        }
    }

    const uint32_t processor_index                 = state->processor_index;
    const uint32_t max_processors_count            = state->max_processors_count;
    struct cpuinfo_arm_linux_processor* processors = state->processors;
    struct cpuinfo_arm_linux_processor* processor  = &state->dummy_processor;
    if (processor_index < max_processors_count) {
        processor = &processors[processor_index];
    }

    const size_t key_length = key_end - line_start;
    switch (key_length) {
        case 6:
            break;
        case 8:
            if (memcmp(line_start, "CPU part", key_length) == 0) {
                parse_cpu_part(value_start, value_end, processor);
            } else if (memcmp(line_start, "Features", key_length) == 0) {
                /* parse_features(value_start, value_end, processor); */
            } else if (memcmp(line_start, "BogoMIPS", key_length) == 0) {
                /* BogoMIPS is useless, don't parse */
            } else if (memcmp(line_start, "Hardware", key_length) == 0) {
                size_t value_length = value_end - value_start;
                if (value_length > CPUINFO_HARDWARE_VALUE_MAX) {
                    MNN_PRINT(
                        "length of Hardware value \"%.*s\" in /proc/cpuinfo exceeds limit (%d): truncating to the "
                        "limit\n",
                        (int)value_length, value_start, CPUINFO_HARDWARE_VALUE_MAX);
                    value_length = CPUINFO_HARDWARE_VALUE_MAX;
                } else {
                    state->hardware[value_length] = '\0';
                }
                memcpy(state->hardware, value_start, value_length);
                MNN_PRINT("parsed /proc/cpuinfo Hardware = \"%.*s\"\n", (int)value_length, value_start);
            } else if (memcmp(line_start, "Revision", key_length) == 0) {
                /* Board revision, no use for now */
            }
            break;
        case 9:
            if (memcmp(line_start, "processor", key_length) == 0) {
                const uint32_t new_processor_index = parse_processor_number(value_start, value_end);
                if (new_processor_index < processor_index) {
                    /* Strange: decreasing processor number */
                    MNN_PRINT("unexpectedly low processor number %u following processor %u in /proc/cpuinfo\n",
                              new_processor_index, processor_index);
                } else if (new_processor_index > processor_index + 1) {
                    /* Strange, but common: skipped processor $(processor_index + 1) */
                    MNN_PRINT("unexpectedly high processor number %u following processor %u in /proc/cpuinfo\n",
                              new_processor_index, processor_index);
                }
                if (new_processor_index < max_processors_count) {
                    /* Record that the processor was mentioned in /proc/cpuinfo */
                    processors[new_processor_index].flags |= CPUINFO_ARM_LINUX_VALID_PROCESSOR;
                } else {
                    /* Log and ignore processor */
                    MNN_PRINT("processor %u in /proc/cpuinfo is ignored: index exceeds system limit %u\n",
                              new_processor_index, max_processors_count - 1);
                }
                state->processor_index = new_processor_index;
                return true;
            } else if (memcmp(line_start, "Processor", key_length) == 0) {
                /* TODO: parse to fix misreported architecture, similar to Android's cpufeatures */
            }
            break;
        case 11:
            if (memcmp(line_start, "CPU variant", key_length) == 0) {
                parse_cpu_variant(value_start, value_end, processor);
            }
            break;
        case 12:
            if (memcmp(line_start, "CPU revision", key_length) == 0) {
                parse_cpu_revision(value_start, value_end, processor);
            }
            break;
        case 15:
            if (memcmp(line_start, "CPU implementer", key_length) == 0) {
                parse_cpu_implementer(value_start, value_end, processor);
            } else if (memcmp(line_start, "CPU implementor", key_length) == 0) {
                parse_cpu_implementer(value_start, value_end, processor);
            }
            break;
        case 16:
            if (memcmp(line_start, "CPU architecture", key_length) == 0) {
                parse_cpu_architecture(value_start, value_end, processor);
            }
            break;
        default:
            break;
    }
    return true;
}

bool cpuinfo_linux_parse_multiline_file(const char* filename, size_t buffer_size, cpuinfo_line_callback callback,
                                        void* context) {
#define RETIEMENT     \
    if (file != -1) { \
        close(file);  \
        file = -1;    \
    }                 \
    return false;

    int file     = -1;
    bool status  = false;
    char* buffer = (char*)alloca(buffer_size);

    file = open(filename, O_RDONLY);
    if (file == -1) {
        MNN_PRINT("failed to open %s\n", filename);
        RETIEMENT
    }

    /* Only used for error reporting */
    size_t position        = 0;
    uint64_t line_number   = 1;
    const char* buffer_end = &buffer[buffer_size];
    char* data_start       = buffer;
    ssize_t bytes_read;
    do {
        bytes_read = read(file, data_start, (size_t)(buffer_end - data_start));
        if (bytes_read < 0) {
            MNN_PRINT("failed to read file %s at position %zu\n", filename, position);
            RETIEMENT
        }

        position += (size_t)bytes_read;
        const char* data_end   = data_start + (size_t)bytes_read;
        const char* line_start = buffer;

        if (bytes_read == 0) {
            /* No more data in the file: process the remaining text in the buffer as a single entry */
            const char* line_end = data_end;
            if (!callback(line_start, line_end, context, line_number)) {
                RETIEMENT
            }
        } else {
            const char* line_end;
            do {
                /* Find the end of the entry, as indicated by newline character ('\n') */
                for (line_end = line_start; line_end != data_end; line_end++) {
                    if (*line_end == '\n') {
                        break;
                    }
                }

                /*
                 * If we located separator at the end of the entry, parse it.
                 * Otherwise, there may be more data at the end; read the file once again.
                 */
                if (line_end != data_end) {
                    if (!callback(line_start, line_end, context, line_number++)) {
                        RETIEMENT
                    }
                    line_start = line_end + 1;
                }
            } while (line_end != data_end);

            /* Move remaining partial line data at the end to the beginning of the buffer */
            const size_t line_length = (size_t)(line_end - line_start);
            memmove(buffer, line_start, line_length);
            data_start = &buffer[line_length];
        }
    } while (bytes_read != 0);

    /* Commit */
    status = true;

    if (file != -1) {
        close(file);
        file = -1;
    }
    return status;
}

bool cpuinfo_arm_linux_parse_proc_cpuinfo(char* hardware, uint32_t max_processors_count,
                                          struct cpuinfo_arm_linux_processor* processors) {
    struct proc_cpuinfo_parser_state state = {
        .hardware             = hardware,
        .processor_index      = 0,
        .max_processors_count = max_processors_count,
        .processors           = processors,
    };

    return cpuinfo_linux_parse_multiline_file("/proc/cpuinfo", BUFFER_SIZE, (cpuinfo_line_callback)parse_line, &state);
}

static inline int cpuinfo_android_property_get(const char* key, char* value) {
    return __system_property_get(key, value);
}

void cpuinfo_arm_android_parse_properties(struct cpuinfo_android_properties* properties) {
    cpuinfo_android_property_get("ro.product.board", properties->ro_product_board);
    cpuinfo_android_property_get("ro.board.platform", properties->ro_board_platform);
    cpuinfo_android_property_get("ro.mediatek.platform", properties->ro_mediatek_platform);
    cpuinfo_android_property_get("ro.arch", properties->ro_arch);
    cpuinfo_android_property_get("ro.chipname", properties->ro_chipname);
    cpuinfo_android_property_get("ro.hardware.chipname", properties->ro_hardware_chipname);
}

static inline uint16_t load_u16le(const void* ptr) {
    return *((const uint16_t*)ptr);
}

static inline uint32_t load_u32le(const void* ptr) {
    return *((const uint32_t*)ptr);
}

/**
 * Tries to match /Samsung Exynos\d{4}$/ signature (case-insensitive) for Samsung Exynos chipsets.
 * If match successful, extracts model information into \p chipset argument.
 *
 * @param start - start of the /proc/cpuinfo Hardware string to match.
 * @param end - end of the /proc/cpuinfo Hardware string to match.
 * @param[out] chipset - location where chipset information will be stored upon a successful match.
 *
 * @returns true if signature matched, false otherwise.
 */
static bool match_samsung_exynos(const char* start, const char* end, struct cpuinfo_arm_chipset* chipset) {
    /*
     * Expect at 18-19 symbols:
     * - "Samsung" (7 symbols) + space + "Exynos" (6 symbols) + optional space 4-digit model number
     */
    const size_t length = end - start;
    switch (length) {
        case 18:
        case 19:
            break;
        default:
            return false;
    }

    /*
     * Check that the string starts with "samsung exynos", case-insensitive.
     * Blocks of 4 characters are loaded and compared as little-endian 32-bit word.
     * Case-insensitive characters are binary ORed with 0x20 to convert them to lowercase.
     */
    const uint32_t expected_sams = UINT32_C(0x20202000) | load_u32le(start);
    if (expected_sams != UINT32_C(0x736D6153) /* "smaS" = reverse("Sams") */) {
        return false;
    }
    const uint32_t expected_ung = UINT32_C(0x00202020) | load_u32le(start + 4);
    if (expected_ung != UINT32_C(0x20676E75) /* " ung" = reverse("ung ") */) {
        return false;
    }
    const uint32_t expected_exyn = UINT32_C(0x20202000) | load_u32le(start + 8);
    if (expected_exyn != UINT32_C(0x6E797845) /* "nyxE" = reverse("Exyn") */) {
        return false;
    }
    const uint16_t expected_os = UINT16_C(0x2020) | load_u16le(start + 12);
    if (expected_os != UINT16_C(0x736F) /* "so" = reverse("os") */) {
        return false;
    }

    const char* pos = start + 14;

    /* There can be a space ' ' following the "Exynos" string */
    if (*pos == ' ') {
        pos++;

        /* If optional space if present, we expect exactly 19 characters */
        if (length != 19) {
            return false;
        }
    }

    /* Validate and parse 4-digit model number */
    uint32_t model = 0;
    for (uint32_t i = 0; i < 4; i++) {
        const uint32_t digit = (uint32_t)(uint8_t)(*pos++) - '0';
        if (digit >= 10) {
            /* Not really a digit */
            return false;
        }
        model = model * 10 + digit;
    }

    /* Return parsed chipset */
    *chipset = (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_samsung,
        .series = cpuinfo_arm_chipset_series_samsung_exynos,
        .model  = model,
    };
    return true;
}

/**
 * Tries to match /exynos\d{4}$/ signature for Samsung Exynos chipsets.
 * If match successful, extracts model information into \p chipset argument.
 *
 * @param start - start of the platform identifier (ro.board.platform or ro.chipname) to match.
 * @param end - end of the platform identifier (ro.board.platform or ro.chipname) to match.
 * @param[out] chipset - location where chipset information will be stored upon a successful match.
 *
 * @returns true if signature matched, false otherwise.
 */
static bool match_exynos(const char* start, const char* end, struct cpuinfo_arm_chipset* chipset) {
    /* Expect exactly 10 symbols: "exynos" (6 symbols) + 4-digit model number */
    if (start + 10 != end) {
        return false;
    }

    /* Load first 4 bytes as little endian 32-bit word */
    const uint32_t expected_exyn = load_u32le(start);
    if (expected_exyn != UINT32_C(0x6E797865) /* "nyxe" = reverse("exyn") */) {
        return false;
    }

    /* Load next 2 bytes as little endian 16-bit word */
    const uint16_t expected_os = load_u16le(start + 4);
    if (expected_os != UINT16_C(0x736F) /* "so" = reverse("os") */) {
        return false;
    }

    /* Check and parse 4-digit model number */
    uint32_t model = 0;
    for (uint32_t i = 6; i < 10; i++) {
        const uint32_t digit = (uint32_t)(uint8_t)start[i] - '0';
        if (digit >= 10) {
            /* Not really a digit */
            return false;
        }
        model = model * 10 + digit;
    }

    /* Return parsed chipset. */
    *chipset = (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_samsung,
        .series = cpuinfo_arm_chipset_series_samsung_exynos,
        .model  = model,
    };
    return true;
}

/**
 * Tries to match /universal\d{4}$/ signature for Samsung Exynos chipsets.
 * If match successful, extracts model information into \p chipset argument.
 *
 * @param start - start of the platform identifier (/proc/cpuinfo Hardware string, ro.product.board or ro.chipname)
 *                to match.
 * @param end - end of the platform identifier (/proc/cpuinfo Hardware string, ro.product.board or ro.chipname)
 *              to match.
 * @param[out] chipset - location where chipset information will be stored upon a successful match.
 *
 * @returns true if signature matched, false otherwise.
 */
static bool match_universal(const char* start, const char* end, struct cpuinfo_arm_chipset* chipset) {
    /* Expect exactly 13 symbols: "universal" (9 symbols) + 4-digit model number */
    if (start + 13 != end) {
        return false;
    }

    /*
     * Check that the string starts with "universal".
     * Blocks of 4 characters are loaded and compared as little-endian 32-bit word.
     * Case-insensitive characters are binary ORed with 0x20 to convert them to lowercase.
     */
    const uint8_t expected_u = UINT8_C(0x20) | (uint8_t)start[0];
    if (expected_u != UINT8_C(0x75) /* "u" */) {
        return false;
    }
    const uint32_t expected_nive = UINT32_C(0x20202020) | load_u32le(start + 1);
    if (expected_nive != UINT32_C(0x6576696E) /* "evin" = reverse("nive") */) {
        return false;
    }
    const uint32_t expected_ersa = UINT32_C(0x20202020) | load_u32le(start + 5);
    if (expected_ersa != UINT32_C(0x6C617372) /* "lasr" = reverse("rsal") */) {
        return false;
    }

    /* Validate and parse 4-digit model number */
    uint32_t model = 0;
    for (uint32_t i = 9; i < 13; i++) {
        const uint32_t digit = (uint32_t)(uint8_t)start[i] - '0';
        if (digit >= 10) {
            /* Not really a digit */
            return false;
        }
        model = model * 10 + digit;
    }

    /* Return parsed chipset. */
    *chipset = (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_samsung,
        .series = cpuinfo_arm_chipset_series_samsung_exynos,
        .model  = model,
    };
    return true;
}

struct cpuinfo_arm_chipset cpuinfo_arm_linux_decode_chipset_from_proc_cpuinfo_hardware(const char* hardware,
                                                                                       uint32_t cores,
                                                                                       uint32_t max_cpu_freq_max) {
    struct cpuinfo_arm_chipset chipset;
    const size_t hardware_length = strnlen(hardware, CPUINFO_HARDWARE_VALUE_MAX);
    const char* hardware_end     = hardware + hardware_length;

    if (match_samsung_exynos(hardware, hardware_end, &chipset)) {
        return chipset;
    }

    if (match_universal(hardware, hardware_end, &chipset)) {
        return chipset;
    }
    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_product_board(const char* ro_product_board,
                                                                                    uint32_t cores,
                                                                                    uint32_t max_cpu_freq_max) {
    struct cpuinfo_arm_chipset chipset;
    const char* board         = ro_product_board;
    const size_t board_length = strnlen(ro_product_board, CPUINFO_BUILD_PROP_VALUE_MAX);
    const char* board_end     = ro_product_board + board_length;

    if (match_universal(board, board_end, &chipset)) {
        return chipset;
    }

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_board_platform(const char* platform,
                                                                                     uint32_t cores,
                                                                                     uint32_t max_cpu_freq_max) {
    struct cpuinfo_arm_chipset chipset;
    const size_t platform_length = strnlen(platform, CPUINFO_BUILD_PROP_VALUE_MAX);
    const char* platform_end     = platform + platform_length;

    if (match_exynos(platform, platform_end, &chipset)) {
        return chipset;
    }

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_mediatek_platform(const char* platform) {
    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_arch(const char* arch) {
    struct cpuinfo_arm_chipset chipset;
    const char* arch_end = arch + strnlen(arch, CPUINFO_BUILD_PROP_VALUE_MAX);

    /* Check Samsung exynosXXXX signature */
    if (match_exynos(arch, arch_end, &chipset)) {
        return chipset;
    }

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset_from_ro_chipname(const char* chipname) {
    struct cpuinfo_arm_chipset chipset;
    const size_t chipname_length = strnlen(chipname, CPUINFO_BUILD_PROP_VALUE_MAX);
    const char* chipname_end     = chipname + chipname_length;

    if (match_exynos(chipname, chipname_end, &chipset)) {
        return chipset;
    }
    if (match_universal(chipname, chipname_end, &chipset)) {
        return chipset;
    }

    return (struct cpuinfo_arm_chipset){
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };
}

struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset(const struct cpuinfo_android_properties* properties,
                                                              uint32_t cores, uint32_t max_cpu_freq_max) {
    // this function is used to decode chipset, which is only used to detect Samsung Exynos chipsets
    // so chipesets now only have TWO classes, one is cpuinfo_arm_chipset_vendor_samsung, the other is
    // cpuinfo_arm_chipset_vendor_unknown

    struct cpuinfo_arm_chipset chipset = {
        .vendor = cpuinfo_arm_chipset_vendor_unknown,
        .series = cpuinfo_arm_chipset_series_unknown,
    };

    struct cpuinfo_arm_chipset chipsets[cpuinfo_android_chipset_property_max] = {
        [cpuinfo_android_chipset_property_proc_cpuinfo_hardware] =
            cpuinfo_arm_linux_decode_chipset_from_proc_cpuinfo_hardware(properties->proc_cpuinfo_hardware, cores,
                                                                        max_cpu_freq_max),
        [cpuinfo_android_chipset_property_ro_product_board] = cpuinfo_arm_android_decode_chipset_from_ro_product_board(
            properties->ro_product_board, cores, max_cpu_freq_max),
        [cpuinfo_android_chipset_property_ro_board_platform] =
            cpuinfo_arm_android_decode_chipset_from_ro_board_platform(properties->ro_board_platform, cores,
                                                                      max_cpu_freq_max),
        [cpuinfo_android_chipset_property_ro_mediatek_platform] =
            cpuinfo_arm_android_decode_chipset_from_ro_mediatek_platform(properties->ro_mediatek_platform),
        [cpuinfo_android_chipset_property_ro_arch] =
            cpuinfo_arm_android_decode_chipset_from_ro_arch(properties->ro_arch),
        [cpuinfo_android_chipset_property_ro_chipname] =
            cpuinfo_arm_android_decode_chipset_from_ro_chipname(properties->ro_chipname),
        [cpuinfo_android_chipset_property_ro_hardware_chipname] =
            cpuinfo_arm_android_decode_chipset_from_ro_chipname(properties->ro_hardware_chipname),
    };

    enum cpuinfo_arm_chipset_vendor vendor = cpuinfo_arm_chipset_vendor_unknown;
    for (size_t i = 0; i < cpuinfo_android_chipset_property_max; ++i) {
        const enum cpuinfo_arm_chipset_vendor decoded_vendor = chipsets[i].vendor;
        if (decoded_vendor != cpuinfo_arm_chipset_vendor_unknown) {
            if (vendor == cpuinfo_arm_chipset_vendor_unknown) {
                vendor = decoded_vendor;
            } else if (vendor != decoded_vendor) {
//                MNN_PRINT(
//                    "[MNN WARNING] chipset detection failed: different chipset vendors reported in different system "
//                    "properties\n");
                return chipset;
            }
        }
    }
    if (vendor == cpuinfo_arm_chipset_vendor_unknown) {
//        MNN_PRINT("[MNN WARNING] chipset detection failed: none of the system properties matched known signatures\n");
        return chipset;
    }

    for (size_t i = 0; i < cpuinfo_android_chipset_property_max; ++i) {
        if (chipsets[i].series != cpuinfo_arm_chipset_series_unknown) {
            chipset = chipsets[i];
            break;
        }
    }

    // MNN_PRINT("chipset vendor, series, model is: %d, %d, %d\n", chipset.vendor, chipset.series, chipset.model);
    return chipset;
}

#endif // __ANDROID__

#if defined(__APPLE__) && defined(__aarch64__)

static uint32_t get_sys_info_by_name(const char* type_specifier) {
    size_t size     = 0;
    uint32_t result = 0;
    if (sysctlbyname(type_specifier, NULL, &size, NULL, 0) != 0) {
        MNN_PRINT("sysctlbyname(\"%s\") failed\n", type_specifier);
    } else if (size == sizeof(uint32_t)) {
        sysctlbyname(type_specifier, &result, &size, NULL, 0);
        MNN_PRINT("%s: %u , size = %lu\n", type_specifier, result, size);
    } else {
        MNN_PRINT("sysctl does not support non-integer lookup for (\"%s\")\n", type_specifier);
    }
    return result;
}

#endif // iOS

void cpuinfo_arm_init(struct cpuinfo_arm_isa* cpuinfo_isa) {
    memset(cpuinfo_isa, 0, sizeof(struct cpuinfo_arm_isa));

    // android
#ifdef __ANDROID__
    struct cpuinfo_arm_linux_processor* arm_linux_processors = NULL;
    const uint32_t processors_count                          = getNumberOfCPU();

    char proc_cpuinfo_hardware[CPUINFO_HARDWARE_VALUE_MAX] = {0};

    arm_linux_processors = static_cast<struct cpuinfo_arm_linux_processor*>(
        calloc(processors_count, sizeof(struct cpuinfo_arm_linux_processor)));
    if (arm_linux_processors == NULL) {
        MNN_PRINT("failed to allocate %zu bytes for descriptions of %u ARM logical processors\n",
                  processors_count * sizeof(struct cpuinfo_arm_linux_processor), processors_count);
        return;
    }

    if (!cpuinfo_arm_linux_parse_proc_cpuinfo(proc_cpuinfo_hardware, processors_count, arm_linux_processors)) {
        MNN_PRINT("failed to parse processor information from /proc/cpuinfo\n");
        return;
    }

    uint32_t valid_processor_mask = 0;
    for (uint32_t i = 0; i < processors_count; i++) {
        if (bitmask_all(arm_linux_processors[i].flags, valid_processor_mask)) {
            arm_linux_processors[i].flags |= CPUINFO_LINUX_FLAG_VALID;
        }
    }

    uint32_t valid_processors = 0, last_midr = 0;
    for (uint32_t i = 0; i < processors_count; i++) {
        arm_linux_processors[i].system_processor_id = i;
        if (bitmask_all(arm_linux_processors[i].flags, CPUINFO_LINUX_FLAG_VALID)) {
            valid_processors += 1;
            if (bitmask_all(arm_linux_processors[i].flags, CPUINFO_ARM_LINUX_VALID_MIDR)) {
                last_midr = arm_linux_processors[i].midr;
            }
        }
    }

    uint32_t isa_features = 0;
#ifdef __aarch64__
    isa_features = (uint32_t)getauxval(AT_HWCAP);
#endif

    struct cpuinfo_android_properties android_properties;
    cpuinfo_arm_android_parse_properties(&android_properties);
    const struct cpuinfo_arm_chipset chipset =
        cpuinfo_arm_android_decode_chipset(&android_properties, valid_processors, 0);

    switch (last_midr & (CPUINFO_ARM_MIDR_IMPLEMENTER_MASK | CPUINFO_ARM_MIDR_PART_MASK)) {
        case UINT32_C(0x51008040): /* Kryo 485 Gold (Cortex-A76) */
            cpuinfo_isa->dot = true;
            break;
        default:
#ifdef __aarch64__
            if (isa_features & CPUINFO_ARM_LINUX_FEATURE_ASIMDDP) {
                cpuinfo_isa->dot = true;
            }
#endif
            // TODO, whitelist, ex: hisilicon_kirin 980...
            break;
    }
#ifdef __aarch64__
    const uint32_t fp16arith_mask = CPUINFO_ARM_LINUX_FEATURE_FPHP | CPUINFO_ARM_LINUX_FEATURE_ASIMDHP;
    if ((isa_features & fp16arith_mask) == fp16arith_mask) {
        if (chipset.series == cpuinfo_arm_chipset_series_samsung_exynos && chipset.model == 9810) {
            cpuinfo_isa->fp16arith = false;
        } else {
            cpuinfo_isa->fp16arith = true;
        }
    }
#else
    // pytorch/cpuinfo: src/arm/linux/aarch32-isa.c
    uint32_t architecture_version = 0;
    if (processors_count > 0) {
        architecture_version = arm_linux_processors[0].architecture_version;
    }
    if (architecture_version >= 8) {
        /*
         * NEON FP16 compute extension and VQRDMLAH/VQRDMLSH instructions are not indicated in /proc/cpuinfo.
         * Use a MIDR-based heuristic to whitelist processors known to support it:
         * - Processors with Cortex-A55 cores
         * - Processors with Cortex-A65 cores
         * - Processors with Cortex-A75 cores
         * - Processors with Cortex-A76 cores
         * - Processors with Cortex-A77 cores
         * - Processors with Exynos M4 cores
         * - Processors with Exynos M5 cores
         * - Neoverse N1 cores
         */
        if (chipset.series == cpuinfo_arm_chipset_series_samsung_exynos && chipset.model == 9810) {
            /* Only little cores of Exynos 9810 support FP16 & RDM */
            MNN_PRINT("FP16 arithmetics and RDM disabled: only little cores in Exynos 9810 support these extensions");
        } else {
            switch (last_midr & (CPUINFO_ARM_MIDR_IMPLEMENTER_MASK | CPUINFO_ARM_MIDR_PART_MASK)) {
                case UINT32_C(0x4100D050): /* Cortex-A55 */
                case UINT32_C(0x4100D060): /* Cortex-A65 */
                case UINT32_C(0x4100D0B0): /* Cortex-A76 */
                case UINT32_C(0x4100D0C0): /* Neoverse N1 */
                case UINT32_C(0x4100D0D0): /* Cortex-A77 */
                case UINT32_C(0x4100D0E0): /* Cortex-A76AE */
                case UINT32_C(0x4800D400): /* Cortex-A76 (HiSilicon) */
                case UINT32_C(0x51008020): /* Kryo 385 Gold (Cortex-A75) */
                case UINT32_C(0x51008030): /* Kryo 385 Silver (Cortex-A55) */
                case UINT32_C(0x51008040): /* Kryo 485 Gold (Cortex-A76) */
                case UINT32_C(0x51008050): /* Kryo 485 Silver (Cortex-A55) */
                case UINT32_C(0x53000030): /* Exynos M4 */
                case UINT32_C(0x53000040): /* Exynos M5 */
                    cpuinfo_isa->fp16arith = true;
                    break;
            }
        }
        /*
         * NEON VDOT instructions are not indicated in /proc/cpuinfo.
         * Use a MIDR-based heuristic to whitelist processors known to support it.
         */
        switch (last_midr & (CPUINFO_ARM_MIDR_IMPLEMENTER_MASK | CPUINFO_ARM_MIDR_PART_MASK)) {
            case UINT32_C(0x4100D0B0): /* Cortex-A76 */
            case UINT32_C(0x4100D0D0): /* Cortex-A77 */
            case UINT32_C(0x4100D0E0): /* Cortex-A76AE */
            case UINT32_C(0x4800D400): /* Cortex-A76 (HiSilicon) */
            case UINT32_C(0x51008040): /* Kryo 485 Gold (Cortex-A76) */
            case UINT32_C(0x51008050): /* Kryo 485 Silver (Cortex-A55) */
            case UINT32_C(0x53000030): /* Exynos-M4 */
            case UINT32_C(0x53000040): /* Exynos-M5 */
                cpuinfo_isa->dot = true;
                break;
            case UINT32_C(0x4100D050): /* Cortex A55: revision 1 or later only */
                cpuinfo_isa->dot = (midr_get_variant(last_midr) >= 1);
                break;
            case UINT32_C(0x4100D0A0): /* Cortex A75: revision 2 or later only */
                cpuinfo_isa->dot = (midr_get_variant(last_midr) >= 2);
                break;
        }
    }
#endif
    if (arm_linux_processors) {
        free(arm_linux_processors);
    }

#endif // #ifdef __ANDROID__

    // iOS
#if defined(__IOS__) && defined(__aarch64__)

// A11
#ifndef CPUFAMILY_ARM_MONSOON_MISTRAL
#define CPUFAMILY_ARM_MONSOON_MISTRAL 0xe81e7ef6
#endif
// A12
#ifndef CPUFAMILY_ARM_VORTEX_TEMPEST
#define CPUFAMILY_ARM_VORTEX_TEMPEST 0x07d34b9f
#endif
// A13
#ifndef CPUFAMILY_ARM_LIGHTNING_THUNDER
#define CPUFAMILY_ARM_LIGHTNING_THUNDER 0x462504d2
#endif
// A14
#ifndef CPUFAMILY_ARM_FIRESTORM_ICESTORM
#define CPUFAMILY_ARM_FIRESTORM_ICESTORM 0x1b588bb3
#endif
// A15
#ifndef CPUFAMILY_ARM_AVALANCHE_BLIZZARD
#define CPUFAMILY_ARM_AVALANCHE_BLIZZARD 0xda33d83d
#endif

    const uint32_t cpu_family = get_sys_info_by_name("hw.cpufamily");
    // const uint32_t cpu_type = get_sys_info_by_name("hw.cputype");
    // const uint32_t cpu_subtype = get_sys_info_by_name("hw.cpusubtype");

    cpuinfo_isa->fp16arith = cpu_family == CPUFAMILY_ARM_MONSOON_MISTRAL ||
                             cpu_family == CPUFAMILY_ARM_VORTEX_TEMPEST ||
                             cpu_family == CPUFAMILY_ARM_LIGHTNING_THUNDER ||
                             cpu_family == CPUFAMILY_ARM_FIRESTORM_ICESTORM ||
                             cpu_family == CPUFAMILY_ARM_AVALANCHE_BLIZZARD;

    cpuinfo_isa->dot = cpu_family == CPUFAMILY_ARM_LIGHTNING_THUNDER ||
                       cpu_family == CPUFAMILY_ARM_FIRESTORM_ICESTORM ||
                       cpu_family == CPUFAMILY_ARM_AVALANCHE_BLIZZARD;

#endif // iOS

// arm64-osx
#if defined(__APPLE__) && defined(__aarch64__) && !defined(__IOS__)   
#ifndef CPUFAMILY_AARCH64_FIRESTORM_ICESTORM
#define CPUFAMILY_AARCH64_FIRESTORM_ICESTORM 0x1b588bb3
#endif
    const uint32_t cpu_family = get_sys_info_by_name("hw.cpufamily");
    cpuinfo_isa->fp16arith = cpu_family == CPUFAMILY_AARCH64_FIRESTORM_ICESTORM;
    cpuinfo_isa->dot = cpu_family == CPUFAMILY_AARCH64_FIRESTORM_ICESTORM;
#endif

#ifndef __ANDROID__
#if defined (__linux__) && defined (__aarch64__)

    uint32_t isa_features = 0;
    isa_features = (uint32_t)getauxval(AT_HWCAP);


        if (isa_features & CPUINFO_ARM_LINUX_FEATURE_ASIMDDP) {
                cpuinfo_isa->dot = true;
        }

        const uint32_t fp16arith_mask = CPUINFO_ARM_LINUX_FEATURE_FPHP | CPUINFO_ARM_LINUX_FEATURE_ASIMDHP;
        if ((isa_features & fp16arith_mask) == fp16arith_mask) {
            cpuinfo_isa->fp16arith = true;
        }

#endif /* __linux__ && __aarch64__ */
#endif

    MNN_PRINT("The device support dot:%d, support fp16:%d\n", cpuinfo_isa->dot, cpuinfo_isa->fp16arith);
}

#endif // MNN_USE_ARMV82
