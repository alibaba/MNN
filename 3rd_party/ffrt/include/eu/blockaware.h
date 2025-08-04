/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef BLOCKAWARE_H
#define BLOCKAWARE_H

#include <securec.h>
#include <cstdio>
#include <cstring>
#include <sys/prctl.h>
#include <cerrno>
#include "dfx/log/ffrt_log_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKAWARE_DOMAIN_ID_MAX     15
#define HM_PR_SILK_BLOCKAWARE_OPS    0x534b4241
#define BLOCKAWARE_SUBOPS_INIT       0x1
#define BLOCKAWARE_SUBOPS_REG        0x2
#define BLOCKAWARE_SUBOPS_UNREG      0x3
#define BLOCKAWARE_SUBOPS_WAIT       0x4
#define BLOCKAWARE_SUBOPS_WAKE       0x5
#define BLOCKAWARE_SUBOPS_MONITORFD  0X6

struct BlockawareDomainInfo {
    unsigned int nrRunning;
    unsigned int nrSleeping;
    unsigned int nrBlocked;
};
struct BlockawareDomainInfoArea {
    struct BlockawareDomainInfo localinfo[BLOCKAWARE_DOMAIN_ID_MAX + 1];
    struct BlockawareDomainInfo globalinfo;
};
struct BlockawareWatermark {
    unsigned int low;
    unsigned int high;
};
struct BlockawareWakeupCond {
    struct BlockawareWatermark local[BLOCKAWARE_DOMAIN_ID_MAX + 1];
    struct BlockawareWatermark global;
    bool check_ahead;
};
struct BlockawareKinfoPageS {
    uint32_t seq;
    struct BlockawareDomainInfoArea infoArea;
};

static inline int BlockawareInit(unsigned long *keyPtr);
static inline int BlockawareRegister(unsigned int domain);
static inline int BlockawareUnregister(void);
static int BlockawareLoadSnapshot(unsigned long key, struct BlockawareDomainInfoArea *infoArea);
static inline int BlockawareEnterSleeping(void);
static inline int BlockawareLeaveSleeping(void);
static inline int BlockawareWaitCond(struct BlockawareWakeupCond *cond);
static inline int BlockawareWake(void);
static inline int BlockawareMonitorfd(int fd, struct BlockawareWakeupCond *cond);

#ifdef __aarch64__
static inline void CpuRelax(void)
{
    asm volatile("yield" ::: "memory");
}

static inline void SmpRmb(void)
{
    asm volatile("dmb ishld" ::: "memory");
}

static inline unsigned long GetTlsPtr(void)
{
    unsigned long tls = 0;
    asm volatile ("mrs %0, tpidr_el0\n" : "=r" (tls));
    return tls;
}

static inline unsigned long *curr_thread_tls_blockaware_slot_of(void)
{
    unsigned long tls = GetTlsPtr();
    unsigned long slot_addr = tls - sizeof (unsigned long) * (2UL + 5UL);
    return reinterpret_cast<unsigned long *>(slot_addr);
}

static inline int BlockawareEnterSleeping(void)
{
    unsigned long *slot_ptr = curr_thread_tls_blockaware_slot_of();
    *slot_ptr += 1;
    return 0;
}

static inline int BlockawareLeaveSleeping(void)
{
    unsigned long *slot_ptr = curr_thread_tls_blockaware_slot_of();
    int err = 0;

    if (*slot_ptr == 0) {
        err = -EINVAL;
    } else {
        *slot_ptr -= 1;
    }

    return err;
}
#elif defined(__arm__)

static inline void CpuRelax(void)
{
    asm volatile("yield" ::: "memory");
}

static inline void SmpRmb(void)
{
    asm volatile("dmb ish" ::: "memory");
}

static inline unsigned long GetTlsPtr(void)
{
    unsigned long tpid = 0;
    asm volatile("mrc p15, 0, %0, c13, c0, 3" : "=r"(tpid));
    return tpid;
}

static inline unsigned long *curr_thread_tls_blockaware_slot_of(void)
{
    unsigned long tls = GetTlsPtr();
    unsigned long slot_addr = tls - sizeof (unsigned long) * (2UL + 5UL);
    return (unsigned long *)slot_addr;
}

static inline int BlockawareEnterSleeping(void)
{
    unsigned long *slot_ptr = curr_thread_tls_blockaware_slot_of();
    *slot_ptr += 1;
    return 0;
}

static inline int BlockawareLeaveSleeping(void)
{
    unsigned long *slot_ptr = curr_thread_tls_blockaware_slot_of();
    int err = 0;

    if (*slot_ptr == 0) {
        err = -EINVAL;
    } else {
        *slot_ptr -= 1;
    }

    return err;
}
#else
static inline void CpuRelax(void)
{
}

static inline void SmpRmb(void)
{
}

static inline unsigned long GetTlsPtr(void)
{
    return 0;
}

static inline int BlockawareEnterSleeping(void)
{
    return 0;
}

static inline int BlockawareLeaveSleeping(void)
{
    return 0;
}
#endif

static inline int BlockawareInit(unsigned long *keyPtr)
{
    int rc = prctl(HM_PR_SILK_BLOCKAWARE_OPS, BLOCKAWARE_SUBOPS_INIT, reinterpret_cast<unsigned long>(keyPtr));
    return (rc == 0) ? 0 : errno;
}

static inline int BlockawareRegister(unsigned int domain)
{
    /* Mention that it is kernel's responsibility to init tls slot to 0 */
    int rc = prctl(HM_PR_SILK_BLOCKAWARE_OPS, BLOCKAWARE_SUBOPS_REG, static_cast<unsigned long>(domain));
    return (rc == 0) ? 0 : errno;
}

static inline int BlockawareUnregister(void)
{
    /* Mention that it is kernel's responsibility to reset tls slot to 0 */
    int rc = prctl(HM_PR_SILK_BLOCKAWARE_OPS, BLOCKAWARE_SUBOPS_UNREG);
    return (rc == 0) ? 0 : errno;
}

static inline uint32_t seqlock_start_read(const uint32_t *seq_ptr)
{
    uint32_t seq;
    do {
        seq = *reinterpret_cast<const volatile uint32_t *>(seq_ptr);
        if ((seq & 1U) == 0U) {
            break;
        }
        CpuRelax();
    } while (true);
    SmpRmb();
    return seq;
}

static inline bool seqlock_check(const uint32_t *seq_ptr, uint32_t seq_prev)
{
    SmpRmb();
    return (*seq_ptr == seq_prev);
}

static int BlockawareLoadSnapshot(unsigned long key, struct BlockawareDomainInfoArea *infoArea)
{
    struct BlockawareKinfoPageS *kinfoPage = reinterpret_cast<struct BlockawareKinfoPageS *>(key);
    uint32_t seq;
    int ret = 0;
    do {
        seq = seqlock_start_read(&kinfoPage->seq);
        ret = memcpy_s(infoArea, sizeof(BlockawareDomainInfoArea),
            &kinfoPage->infoArea, sizeof(BlockawareDomainInfoArea));
    } while (!seqlock_check(&kinfoPage->seq, seq));
    if (ret != EOK) {
        FFRT_SYSEVENT_LOGE("The memcpy operation failed for the infoArea.");
    }
    return ret;
}

static inline unsigned int BlockawareLoadSnapshotNrRunningFast(unsigned long key, int domainId)
{
    BlockawareKinfoPageS* kinfoPage = reinterpret_cast<BlockawareKinfoPageS*>(key);
    return kinfoPage->infoArea.localinfo[domainId].nrRunning;
}

static inline unsigned int BlockawareLoadSnapshotNrBlockedFast(unsigned long key, int domainId)
{
    BlockawareKinfoPageS* kinfoPage = reinterpret_cast<BlockawareKinfoPageS*>(key);
    return kinfoPage->infoArea.localinfo[domainId].nrBlocked;
}

static inline int BlockawareWaitCond(struct BlockawareWakeupCond *cond)
{
    int rc = prctl(HM_PR_SILK_BLOCKAWARE_OPS, BLOCKAWARE_SUBOPS_WAIT, reinterpret_cast<unsigned long>(cond));
    return (rc == 0) ? 0 : errno;
}

static inline int BlockawareWake(void)
{
    int rc = prctl(HM_PR_SILK_BLOCKAWARE_OPS, BLOCKAWARE_SUBOPS_WAKE);
    return (rc == 0) ? 0 : errno;
}

static inline int BlockawareMonitorfd(int fd, struct BlockawareWakeupCond *cond)
{
    int rc = prctl(HM_PR_SILK_BLOCKAWARE_OPS, BLOCKAWARE_SUBOPS_MONITORFD,
        static_cast<unsigned long>(fd), reinterpret_cast<unsigned long>(cond));
    return (rc >= 0) ? rc : -errno;
}

#ifdef __cplusplus
}
#endif
#endif /* BLOCKAWARE_H */