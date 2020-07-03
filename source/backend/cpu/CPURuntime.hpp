//
//  CPURuntime.hpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <stdint.h>
#ifndef CPURuntime_hpp
#define CPURuntime_hpp

#if defined(__aarch64__) && defined(ENABLE_ARMV82)
struct cpuinfo_arm_isa {
    bool fp16arith;
    bool dot;
};

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
#endif // __ANDROID__

#endif

/*
 CPU thread mode, only effective on HMP（Heterogeneous Multi-Processing）arch CPUs
 that have ARM big.LITTLE technology and on Android
 */
typedef enum {
    /* Compliance with Operating System Scheduling */
    MNN_CPU_MODE_DEFAULT = 0,
    /* Bind threads to CPU IDs according to CPU frequency, but this mode is power-friendly */
    MNN_CPU_MODE_POWER_FRI = 1,
    /* Bind threads to little CPUs */
    MNN_CPU_MODE_LITTLE = 2,
    /* Bind threads to big CPUs */
    MNN_CPU_MODE_BIG = 3
} MNNCPUThreadsMode;
int MNNSetCPUThreadsMode(MNNCPUThreadsMode mode);

//
float MNNGetCPUFlops(uint32_t number);

#if defined(__aarch64__) && defined(ENABLE_ARMV82)

void cpuinfo_arm_init(struct cpuinfo_arm_isa* cpuinfo_isa);

#endif // __aarch64__

#endif /* CPUInfo_hpp */
