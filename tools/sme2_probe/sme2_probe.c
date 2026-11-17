//
//  sme2_probe.c
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

// Standalone SME2 hardware probe for Apple Silicon. Independent of MNN, so it can
// run before/without the engine to answer: how many SME2 units does this SoC have?
//
// Run: ./MNNSme2Probe [bench_iters]
//
// What it does:
//   1. Dumps sysctl capability flags (FEAT_SME/SME2/...) and CPU topology.
//   2. Executes SME1 (smstart/zero{za}/fmopa) and SME2 (ldr zt0/luti4) blocks on
//      one thread per logical CPU; a SIGILL handler records which cores can run them.
//   3. Measures FMOPA throughput per capable core, then with k = 1..N threads, to
//      see whether the SME2 units are per-core or shared.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>

#define MAX_CPU 64
// 8 fmopa per loop iteration, each 16x16 FMA (vl16 pattern) = 4096 FLOP/iter
#define FLOP_PER_ITER 4096.0

static __thread sigjmp_buf* g_jmp = NULL;
static unsigned char g_ztbuf[4096] __attribute__((aligned(64)));

static void on_signal(int sig) {
    if (g_jmp) {
        siglongjmp(*g_jmp, sig);
    }
    _exit(128 + sig);
}

static int bind_cpu(int cpu) {
    thread_affinity_policy_data_t pol = { cpu };
    return thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY, (thread_policy_t)&pol,
                             THREAD_AFFINITY_POLICY_COUNT);
}

static int sys_int(const char* name, int def) {
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) {
        return def;
    }
    return v;
}

// SME1: streaming mode + ZA outer product (FMOPA is SME1)
static int sme1_block(void) {
    __asm__ volatile(
        ".inst 0xd503477f\n"  // smstart
        ".inst 0x2598e120\n"  // ptrue p0.s, vl16
        ".inst 0x2598e121\n"  // ptrue p1.s, vl16
        ".inst 0xc00800ff\n"  // zero {za}
        ".inst 0x25b8c000\n"  // mov z0.s, #0
        ".inst 0x25b8c001\n"  // mov z1.s, #0
        ".inst 0x25b8c002\n"  // mov z2.s, #0
        ".inst 0x25b8c003\n"  // mov z3.s, #0
        "mov x13, %0\n"
        ".inst 0xa540a5be\n"  // ld1w {z30.s}, p1/z, [x13]  (streaming-mode load)
        ".inst 0x80800020\n"  // fmopa za0.s, p0/m, p0/m, z1.s, z0.s
        ".inst 0x80820061\n"  // fmopa za1.s, p0/m, p0/m, z3.s, z2.s
        ".inst 0x808400a2\n"  // fmopa za2.s, p0/m, p0/m, z5.s, z4.s
        ".inst 0x808600e3\n"  // fmopa za3.s, p0/m, p0/m, z7.s, z6.s
        ".inst 0xd503467f\n"  // smstop
        :
        : "r"(g_ztbuf)
        : "memory", "x13");
    return 1;
}

// SME2: ZT0 lookup-table instructions (ldr zt0 / luti4 are SME2-only)
static int sme2_block(void) {
    __asm__ volatile(
        ".inst 0xd503477f\n"  // smstart
        "mov x8, %0\n"
        ".inst 0xe11f8100\n"  // ldr zt0, [x8]
        ".inst 0x25b8c000\n"  // mov z0.s, #0  (luti4 index source)
        ".inst 0xc08a4004\n"  // luti4 {z4.b-z5.b}, zt0, z0[0]
        ".inst 0xd503467f\n"  // smstop
        :
        : "r"(g_ztbuf)
        : "memory", "x8");
    return 1;
}

struct cap_arg {
    int cpu;
    int sme1;
    int sme2;
    int bind_rc;
};

struct bench_arg {
    int cpu;
    long iters;
    double secs;
    int ok;
};

static void* cap_worker(void* p) {
    struct cap_arg* a = (struct cap_arg*)p;
    a->bind_rc = bind_cpu(a->cpu);
    a->sme1 = 0;
    a->sme2 = 0;
    sigjmp_buf jb;
    g_jmp = &jb;
    if (sigsetjmp(jb, 1) == 0) {
        a->sme1 = sme1_block();
    }
    if (sigsetjmp(jb, 1) == 0) {
        a->sme2 = sme2_block();
    }
    g_jmp = NULL;
    return NULL;
}

// NEON cross-check: proves thread placement (P vs E cores differ for NEON, so
// identical SME2 rates across all cores cannot be a binding artifact).
#define NEON_FLOP_PER_ITER 64.0
static void* neon_bench_worker(void* p) {
    struct bench_arg* a = (struct bench_arg*)p;
    bind_cpu(a->cpu);
    a->secs = -1.0;
    a->ok = 0;
    long w = 2000;
    __asm__ volatile(
        "1: fmla v0.4s, v1.4s, v2.4s\n"
        "fmla v3.4s, v4.4s, v5.4s\n"
        "fmla v6.4s, v7.4s, v16.4s\n"
        "fmla v17.4s, v18.4s, v19.4s\n"
        "fmla v20.4s, v21.4s, v22.4s\n"
        "fmla v23.4s, v24.4s, v25.4s\n"
        "fmla v26.4s, v27.4s, v28.4s\n"
        "fmla v29.4s, v30.4s, v31.4s\n"
        "subs %0, %0, #1\n"
        "bne 1b\n"
        : "+r"(w)
        :
        : "cc", "memory");
    uint64_t t0 = mach_absolute_time();
    long it = a->iters;
    __asm__ volatile(
        "1: fmla v0.4s, v1.4s, v2.4s\n"
        "fmla v3.4s, v4.4s, v5.4s\n"
        "fmla v6.4s, v7.4s, v16.4s\n"
        "fmla v17.4s, v18.4s, v19.4s\n"
        "fmla v20.4s, v21.4s, v22.4s\n"
        "fmla v23.4s, v24.4s, v25.4s\n"
        "fmla v26.4s, v27.4s, v28.4s\n"
        "fmla v29.4s, v30.4s, v31.4s\n"
        "subs %0, %0, #1\n"
        "bne 1b\n"
        : "+r"(it)
        :
        : "cc", "memory");
    uint64_t t1 = mach_absolute_time();
    static mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    a->secs = (double)(t1 - t0) * (double)tb.numer / (double)tb.denom / 1e9;
    a->ok = 1;
    return NULL;
}

// E-core placement via occupancy: 4 spin threads saturate the P-cores, so the
// probe thread must land on an E-core. Its NEON rate confirms the placement.
static volatile int g_spin_stop = 0;

static void* spin_worker(void* p) {
    (void)p;
    long x = 1;
    while (!g_spin_stop) {
        x = x * 3 + 1;
    }
    return (void*)x;
}

struct eprobe_arg {
    double neon_gflops;
    int sme1;
    int sme2;
    double sme_gflops;
    long iters;
};

static void* eprobe_worker(void* p) {
    struct eprobe_arg* a = (struct eprobe_arg*)p;
    a->neon_gflops = -1;
    a->sme1 = -1;
    a->sme2 = -1;
    a->sme_gflops = -1;

    // NEON self-check first (placement evidence)
    {
        long w = 2000;
        __asm__ volatile(
            "1: fmla v0.4s, v1.4s, v2.4s\n"
            "fmla v3.4s, v4.4s, v5.4s\n"
            "fmla v6.4s, v7.4s, v16.4s\n"
            "fmla v17.4s, v18.4s, v19.4s\n"
            "fmla v20.4s, v21.4s, v22.4s\n"
            "fmla v23.4s, v24.4s, v25.4s\n"
            "fmla v26.4s, v27.4s, v28.4s\n"
            "fmla v29.4s, v30.4s, v31.4s\n"
            "subs %0, %0, #1\n"
            "bne 1b\n"
            : "+r"(w)
            :
            : "cc", "memory");
        uint64_t t0 = mach_absolute_time();
        long it = a->iters;
        __asm__ volatile(
            "1: fmla v0.4s, v1.4s, v2.4s\n"
            "fmla v3.4s, v4.4s, v5.4s\n"
            "fmla v6.4s, v7.4s, v16.4s\n"
            "fmla v17.4s, v18.4s, v19.4s\n"
            "fmla v20.4s, v21.4s, v22.4s\n"
            "fmla v23.4s, v24.4s, v25.4s\n"
            "fmla v26.4s, v27.4s, v28.4s\n"
            "fmla v29.4s, v30.4s, v31.4s\n"
            "subs %0, %0, #1\n"
            "bne 1b\n"
            : "+r"(it)
            :
            : "cc", "memory");
        uint64_t t1 = mach_absolute_time();
        static mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        double secs = (double)(t1 - t0) * (double)tb.numer / (double)tb.denom / 1e9;
        a->neon_gflops = a->iters * NEON_FLOP_PER_ITER / secs / 1e9;
    }

    // capability: fresh sigjmp_buf per section so a trap is always caught cleanly
    sigjmp_buf jb1;
    g_jmp = &jb1;
    if (sigsetjmp(jb1, 1) == 0) {
        a->sme1 = sme1_block();
    } else {
        a->sme1 = 0;
    }
    sigjmp_buf jb2;
    g_jmp = &jb2;
    if (sigsetjmp(jb2, 1) == 0) {
        a->sme2 = sme2_block();
    } else {
        a->sme2 = 0;
    }
    g_jmp = NULL;

    if (a->sme2 == 1) {
        sigjmp_buf jb3;
        g_jmp = &jb3;
        if (sigsetjmp(jb3, 1) == 0) {
            __asm__ volatile(
                ".inst 0xd503477f\n"  // smstart
                ".inst 0x2598e120\n"  // ptrue p0.s, vl16
                ".inst 0x2598e121\n"  // ptrue p1.s, vl16
                ".inst 0xc00800ff\n"  // zero {za}
                ".inst 0x25b8c000\n"
                ".inst 0x25b8c001\n"
                ".inst 0x25b8c002\n"
                ".inst 0x25b8c003\n"
                ".inst 0x25b8c004\n"
                ".inst 0x25b8c005\n"
                ".inst 0x25b8c006\n"
                ".inst 0x25b8c007\n"
                :
                :
                : "memory");
            long w = 2000;
            __asm__ volatile(
                "1: .inst 0x80800020\n"
                ".inst 0x80820061\n"
                ".inst 0x808400a2\n"
                ".inst 0x808600e3\n"
                ".inst 0x80800020\n"
                ".inst 0x80820061\n"
                ".inst 0x808400a2\n"
                ".inst 0x808600e3\n"
                "subs %0, %0, #1\n"
                "bne 1b\n"
                : "+r"(w)
                :
                : "cc", "memory");
            uint64_t t0 = mach_absolute_time();
            long it = a->iters;
            __asm__ volatile(
                "1: .inst 0x80800020\n"
                ".inst 0x80820061\n"
                ".inst 0x808400a2\n"
                ".inst 0x808600e3\n"
                ".inst 0x80800020\n"
                ".inst 0x80820061\n"
                ".inst 0x808400a2\n"
                ".inst 0x808600e3\n"
                "subs %0, %0, #1\n"
                "bne 1b\n"
                : "+r"(it)
                :
                : "cc", "memory");
            uint64_t t1 = mach_absolute_time();
            __asm__ volatile(".inst 0xd503467f\n" ::: "memory");  // smstop
            static mach_timebase_info_data_t tb;
            mach_timebase_info(&tb);
            double secs = (double)(t1 - t0) * (double)tb.numer / (double)tb.denom / 1e9;
            a->sme_gflops = a->iters * FLOP_PER_ITER / secs / 1e9;
        }
        g_jmp = NULL;
    }
    return NULL;
}

static void* bench_worker(void* p) {
    struct bench_arg* a = (struct bench_arg*)p;
    bind_cpu(a->cpu);
    a->secs = -1.0;
    a->ok = 0;
    sigjmp_buf jb;
    g_jmp = &jb;
    if (sigsetjmp(jb, 1) != 0) {
        g_jmp = NULL;
        return NULL;
    }

    __asm__ volatile(
        ".inst 0xd503477f\n"  // smstart
        ".inst 0x2598e120\n"  // ptrue p0.s, vl16
        ".inst 0x2598e121\n"  // ptrue p1.s, vl16
        ".inst 0xc00800ff\n"  // zero {za}
        ".inst 0x25b8c000\n"  // mov z0.s, #0
        ".inst 0x25b8c001\n"  // mov z1.s, #0
        ".inst 0x25b8c002\n"  // mov z2.s, #0
        ".inst 0x25b8c003\n"  // mov z3.s, #0
        ".inst 0x25b8c004\n"  // mov z4.s, #0
        ".inst 0x25b8c005\n"  // mov z5.s, #0
        ".inst 0x25b8c006\n"  // mov z6.s, #0
        ".inst 0x25b8c007\n"  // mov z7.s, #0
        :
        :
        : "memory");

    // warmup
    long w = 2000;
    __asm__ volatile(
        "1: .inst 0x80800020\n"  // fmopa za0.s, p0/m, p0/m, z1.s, z0.s
        ".inst 0x80820061\n"     // fmopa za1.s, p0/m, p0/m, z3.s, z2.s
        ".inst 0x808400a2\n"     // fmopa za2.s, p0/m, p0/m, z5.s, z4.s
        ".inst 0x808600e3\n"     // fmopa za3.s, p0/m, p0/m, z7.s, z6.s
        ".inst 0x80800020\n"
        ".inst 0x80820061\n"
        ".inst 0x808400a2\n"
        ".inst 0x808600e3\n"
        "subs %0, %0, #1\n"
        "bne 1b\n"
        : "+r"(w)
        :
        : "cc", "memory");

    uint64_t t0 = mach_absolute_time();
    long it = a->iters;
    __asm__ volatile(
        "1: .inst 0x80800020\n"
        ".inst 0x80820061\n"
        ".inst 0x808400a2\n"
        ".inst 0x808600e3\n"
        ".inst 0x80800020\n"
        ".inst 0x80820061\n"
        ".inst 0x808400a2\n"
        ".inst 0x808600e3\n"
        "subs %0, %0, #1\n"
        "bne 1b\n"
        : "+r"(it)
        :
        : "cc", "memory");
    uint64_t t1 = mach_absolute_time();

    __asm__ volatile(".inst 0xd503467f\n" ::: "memory");  // smstop
    g_jmp = NULL;

    static mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    a->secs = (double)(t1 - t0) * (double)tb.numer / (double)tb.denom / 1e9;
    a->ok = 1;
    return NULL;
}

static void print_optional_int(const char* label, const char* name) {
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, NULL, 0) == 0) {
        printf("  %-42s %d\n", label, v);
    } else {
        printf("  %-42s n/a\n", label);
    }
}

int main(int argc, char** argv) {
    long iters = 50000000;
    if (argc > 1) {
        iters = atol(argv[1]);
        if (iters <= 0) {
            iters = 50000000;
        }
    }

    signal(SIGILL, on_signal);
    signal(SIGSEGV, on_signal);
    signal(SIGBUS, on_signal);

    int ncpu = sys_int("hw.logicalcpu", sys_int("hw.ncpu", 8));
    if (ncpu > MAX_CPU) {
        ncpu = MAX_CPU;
    }

    printf("=== machine info (sysctl) ===\n");
    {
        char buf[256];
        size_t sz = sizeof(buf);
        if (sysctlbyname("machdep.cpu.brand_string", buf, &sz, NULL, 0) == 0) {
            printf("  %-42s %s\n", "machdep.cpu.brand_string", buf);
        }
        sz = sizeof(buf);
        if (sysctlbyname("hw.model", buf, &sz, NULL, 0) == 0) {
            printf("  %-42s %s\n", "hw.model", buf);
        }
    }
    printf("  %-42s %d\n", "hw.ncpu", sys_int("hw.ncpu", -1));
    printf("  %-42s %d\n", "hw.logicalcpu", sys_int("hw.logicalcpu", -1));
    printf("  %-42s %d\n", "hw.physicalcpu", sys_int("hw.physicalcpu", -1));
    printf("  %-42s %d\n", "hw.perflevel0.logicalcpu (P cores)", sys_int("hw.perflevel0.logicalcpu", -1));
    printf("  %-42s %d\n", "hw.perflevel1.logicalcpu (E cores)", sys_int("hw.perflevel1.logicalcpu", -1));
    print_optional_int("hw.optional.arm.FEAT_SME", "hw.optional.arm.FEAT_SME");
    print_optional_int("hw.optional.arm.FEAT_SME2", "hw.optional.arm.FEAT_SME2");
    print_optional_int("hw.optional.arm.FEAT_SME_F16F16", "hw.optional.arm.FEAT_SME_F16F16");
    print_optional_int("hw.optional.arm.FEAT_SME_F64F64", "hw.optional.arm.FEAT_SME_F64F64");
    print_optional_int("hw.optional.arm.FEAT_SME_I16I64", "hw.optional.arm.FEAT_SME_I16I64");
    print_optional_int("hw.optional.arm.FEAT_SVE", "hw.optional.arm.FEAT_SVE");
    print_optional_int("hw.optional.arm.FEAT_SSVE", "hw.optional.arm.FEAT_SSVE");
    printf("\n");

    // Capability test. One thread at a time; binding is refused by macOS
    // (rc=46), so each runnable thread lands on a P-core.
    printf("=== SME capability test (SIGILL-trapped) ===\n");
    printf("  run# | SME1 (smstart/za/fmopa) | SME2 (zt0/luti4) | bind rc\n");
    struct cap_arg caps[MAX_CPU];
    pthread_t th[MAX_CPU];
    for (int i = 0; i < ncpu; ++i) {
        caps[i].cpu = i;
        pthread_create(&th[i], NULL, cap_worker, &caps[i]);
    }
    for (int i = 0; i < ncpu; ++i) {
        pthread_join(th[i], NULL);
        printf("  %4d | %-23s | %-15s | %d\n", i, caps[i].sme1 ? "YES" : "no", caps[i].sme2 ? "YES" : "no",
               caps[i].bind_rc);
    }
    printf("\n");

    int capable[MAX_CPU];
    int ncap = 0;
    for (int i = 0; i < ncpu; ++i) {
        if (caps[i].sme1 && caps[i].sme2) {
            capable[ncap++] = i;
        }
    }
    if (ncap == 0) {
        printf("No SME2-capable core found. Aborting bench.\n");
        return 0;
    }

    // Single-thread throughput: one thread at a time, unpinned (lands on a P-core).
    printf("=== single-thread, %ld iters/run ===\n", iters);
    printf("  run# | SME2 fmopa GFLOPS | NEON fmla GFLOPS   (NEON rate identifies P vs E)\n");
    double best_single = 0.0;
    for (int j = 0; j < ncpu; ++j) {
        struct bench_arg ba;
        pthread_t t;
        double sme = -1, neon = -1;
        ba.cpu = j;
        ba.iters = iters;
        ba.secs = -1;
        pthread_create(&t, NULL, bench_worker, &ba);
        pthread_join(t, NULL);
        if (ba.ok) {
            sme = iters * FLOP_PER_ITER / ba.secs / 1e9;
            if (sme > best_single) {
                best_single = sme;
            }
        }
        ba.secs = -1;
        pthread_create(&t, NULL, neon_bench_worker, &ba);
        pthread_join(t, NULL);
        if (ba.ok) {
            neon = iters * NEON_FLOP_PER_ITER / ba.secs / 1e9;
        }
        printf("  %4d | %16.1f | %16.1f\n", j, sme, neon);
    }
    printf("\n");

    // Multi-thread scaling. NOTE: thread_policy_set(THREAD_AFFINITY_POLICY)
    // returns KERN_NOT_SUPPORTED (46) on macOS, so threads are NOT pinned; the
    // scheduler spreads runnable threads over the idle cores.
    printf("=== scaling: k threads, aggregate SME2 fmopa GFLOPS (unpinned) ===\n");
    double scale_ref = 0.0, peak_agg = 0.0;
    int peak_k = 0;
    {
        for (int k = 1; k <= ncpu; ++k) {
            struct bench_arg ba[MAX_CPU];
            pthread_t tt[MAX_CPU];
            for (int j = 0; j < k; ++j) {
                ba[j].cpu = -1;
                ba[j].iters = iters;
                ba[j].secs = -1;
                pthread_create(&tt[j], NULL, bench_worker, &ba[j]);
            }
            double tmax = 0.0;
            int allok = 1;
            for (int j = 0; j < k; ++j) {
                pthread_join(tt[j], NULL);
                if (!ba[j].ok) {
                    allok = 0;
                }
                if (ba[j].secs > tmax) {
                    tmax = ba[j].secs;
                }
            }
            if (!allok || tmax <= 0) {
                printf("  k=%d: trapped\n", k);
                break;
            }
            double agg = k * iters * FLOP_PER_ITER / tmax / 1e9;
            if (k == 1) {
                scale_ref = agg;
            }
            if (agg > peak_agg) {
                peak_agg = agg;
                peak_k = k;
            }
            printf("  k=%2d: %8.1f GFLOPS   speedup %.2fx (vs 1T)\n", k, agg, scale_ref > 0 ? agg / scale_ref : 0.0);
        }
    }
    printf("\n");

    // E-core placement via occupancy: 4 spin threads saturate the P-cores,
    // forcing the probe thread onto an E-core. NEON rate confirms placement.
    printf("=== E-core check: P-cores saturated by spins, probe forced onto E-core ===\n");
    for (int trial = 0; trial < 2; ++trial) {
        g_spin_stop = 0;
        pthread_t spins[4];
        for (int j = 0; j < 4; ++j) {
            pthread_create(&spins[j], NULL, spin_worker, NULL);
        }
        usleep(300000);
        struct eprobe_arg ea;
        ea.iters = iters;
        pthread_t t;
        pthread_create(&t, NULL, eprobe_worker, &ea);
        pthread_join(t, NULL);
        g_spin_stop = 1;
        for (int j = 0; j < 4; ++j) {
            pthread_join(spins[j], NULL);
        }
        const char* where = ea.neon_gflops >= 0 && ea.neon_gflops < 65 ? "E-CORE" : "P-core?";
        printf("  trial %d: NEON %.1f GFLOPS -> %s | SME1 %s | SME2 %s | SME2 fmopa %.1f GFLOPS\n", trial,
               ea.neon_gflops, where, ea.sme1 == 1 ? "YES" : "no", ea.sme2 == 1 ? "YES" : "no", ea.sme_gflops);
    }
    printf("\n");

    // E-only scaling: P-cores saturated, k probes all land on E-cores.
    printf("=== E-core scaling: k probe threads, P-cores occupied by spins ===\n");
    {
        double ref = 0.0;
        for (int k = 1; k <= 6; ++k) {
            g_spin_stop = 0;
            pthread_t spins[4];
            for (int j = 0; j < 4; ++j) {
                pthread_create(&spins[j], NULL, spin_worker, NULL);
            }
            usleep(300000);
            struct eprobe_arg ea[6];
            pthread_t tt[6];
            for (int j = 0; j < k; ++j) {
                ea[j].iters = iters;
                pthread_create(&tt[j], NULL, eprobe_worker, &ea[j]);
            }
            double tmax = 0.0;
            double neon_min = 1e9;
            for (int j = 0; j < k; ++j) {
                pthread_join(tt[j], NULL);
                double secs = ea[j].sme_gflops > 0 ? iters * FLOP_PER_ITER / ea[j].sme_gflops / 1e9 : 1e9;
                if (secs > tmax) {
                    tmax = secs;
                }
                if (ea[j].neon_gflops >= 0 && ea[j].neon_gflops < neon_min) {
                    neon_min = ea[j].neon_gflops;
                }
            }
            g_spin_stop = 1;
            for (int j = 0; j < 4; ++j) {
                pthread_join(spins[j], NULL);
            }
            double agg = k * iters * FLOP_PER_ITER / tmax / 1e9;
            if (k == 1) {
                ref = agg;
            }
            printf("  k=%d: %8.1f GFLOPS  speedup %.2fx (min NEON %.1f => placement %s)\n", k, agg,
                   ref > 0 ? agg / ref : 0.0, neon_min, neon_min < 65 ? "E" : "P?");
        }
    }
    printf("\n");

    printf("=== conclusion ===\n");
    printf("1) SME2 instructions execute on %d of %d logical cores (no SIGILL).\n", ncap, ncpu);
    printf("2) Single-core FMOPA peak: %.0f GFLOPS; aggregate saturates at %.0f GFLOPS (k=%d, %.2fx).\n",
           best_single, peak_agg, peak_k, scale_ref > 0 ? peak_agg / scale_ref : 0.0);
    printf("3) If the aggregate stays flat while k grows, the SoC shares one SMCU and MNN should\n");
    printf("   report smeCoreNumber=1: extra cores only help through the NEON branch.\n");
    return 0;
}
