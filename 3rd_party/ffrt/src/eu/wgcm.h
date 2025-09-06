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

#ifndef _UAPI_LINUX_WGCM_H
#define _UAPI_LINUX_WGCM_H

#include <linux/types.h>

constexpr int WGCM_TASK_ALIGN = 64;

constexpr int WGCM_ACTIVELY_WAKE = 0x10;
constexpr int PR_WGCM_CTL = 59;
/*
 * struct wgcm_workergrp_data: controls the state of WGCM tasks.
 *
 * The struct is aligned at 64 bytes to ensure that it fits into
 * a single cache line.
 */
struct wgcm_workergrp_data {
    /* wgcm workergroup's id */
    __u32   gid;

    /* server's thread id */
    __u32   server_tid;

    /*
     * min_concur_workers & max_workers_sum:
     * These two paras are used to detemine wether to wake up the server.
     *
     * When (workers_sum - blk_workers_sum < min_concur_workers) &&
     * (workers_sum < max_workers_sum), wake up server.
     */
    __u32   min_concur_workers;
    __u32   max_workers_sum;

    /* count the number of workers which is bound with server */
    __u32   workers_sum;

    /* count the number of block workers */
    __u32   blk_workers_sum;

    /* indicates whether the server task is actively woken up */
    __u32   woken_flag;

    __u32   reserved;
} __attribute__((packed, aligned(WGCM_TASK_ALIGN)));

/**
 * enum wgcm_ctl_flag - flags to pass to wgcm_ctl()
 * @WGCM_CTL_SERVER_REG: register the current task as a WGCM server
 * @WGCM_CTL_WORKER_REG: register the current task as a WGCM worker
 * @WGCM_CTL_UNREGISTER: unregister the current task as a WGCM task
 * @WGCM_CTL_GET: get infomation about workergroup
 * @WGCM_CTL_SET_GRP: set min_concur_workers & max_workers_sum to workergroup
 * @WGCM_CTL_WAIT: server thread enter the hibernation state
 * @WGCM_CTL_WAKE: actively wakes up WGCM server
 */
enum wgcm_ctl_flag {
    WGCM_CTL_SERVER_REG = 1,
    WGCM_CTL_WORKER_REG,
    WGCM_CTL_SET_GRP,
    WGCM_CTL_UNREGISTER,
    WGCM_CTL_GET = 5,
    WGCM_CTL_WAIT,
    WGCM_CTL_WAKE,
    WGCM_CTL_MAX_NR,
};

#endif /* _UAPI_LINUX_WGCM_H */
