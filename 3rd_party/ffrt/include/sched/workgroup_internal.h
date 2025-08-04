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

#ifndef WORKGROUP_INCLUDE
#define WORKGROUP_INCLUDE

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <cstdbool>
#include <chrono>
#include <fcntl.h>
#include <string>

constexpr int RS_UID = 1003;
constexpr int MAX_WG_THREADS = 32;
#define MAX_FRAME_BUFFER 6
#define gettid() syscall(SYS_gettid)
namespace ffrt {
enum WgType {
    TYPE_DEFAULT = 0,
    TYPE_RS = 1,
    TYPE_MAX
};

struct WorkGroup {
    bool started;
    int rtgId;
    int tids[MAX_WG_THREADS];
    uint64_t interval;
    int qos;
    WgType type;
};

#if (defined(QOS_FRAME_RTG))

struct WorkGroup* WorkgroupCreate(uint64_t interval, int qos);
int WorkgroupClear(struct WorkGroup* wg);
bool JoinWG(int tid, int qos);
bool LeaveWG(int tid, int qos);

#else

#ifdef QOS_WORKER_FRAME_RTG
    WorkGroup* CreateRSWorkGroup(uint64_t interval, int qos);
    bool JoinRSWorkGroup(int tid, int qos);
    bool LeaveRSWorkGroup(int tid, int qos);
    bool DestoryRSWorkGroup(int qos);
#endif

inline struct WorkGroup* WorkgroupCreate(uint64_t interval __attribute__((unused)), int qos)
{
#ifdef QOS_WORKER_FRAME_RTG
    int uid = getuid();
    if (uid == RS_UID) {
        return CreateRSWorkGroup(interval, qos);
    }
#endif
    struct WorkGroup* wg = new (std::nothrow) struct WorkGroup();
    if (wg == nullptr) {
        return nullptr;
    }
    return wg;
}

inline int WorkgroupClear(struct WorkGroup* wg)
{
#ifdef QOS_WORKER_FRAME_RTG
    int uid = getuid();
    if (uid == RS_UID) {
        return DestoryRSWorkGroup(wg->qos);
    }
#endif
    delete wg;
    wg = nullptr;
    return 0;
}

inline bool JoinWG(int tid, int qos)
{
#ifdef QOS_WORKER_FRAME_RTG
    int uid = getuid();
    if (uid == RS_UID) {
        return JoinRSWorkGroup(tid, qos);
    }
#endif
    (void)tid;
    return true;
}

inline bool LeaveWG(int tid, int qos)
{
#ifdef QOS_WORKER_FRAME_RTG
    int uid = getuid();
    if (uid == RS_UID) {
        return LeaveRSWorkGroup(tid, qos);
    }
#endif
    (void)tid;
    return true;
}

#endif

#if defined(QOS_FRAME_RTG)

void WorkgroupStartInterval(struct WorkGroup* wg);
void WorkgroupStopInterval(struct WorkGroup* wg);
void WorkgroupJoin(struct WorkGroup* wg, int tid);

#else /* !QOS_FRAME_RTG */

inline void WorkgroupStartInterval(struct WorkGroup* wg)
{
    if (wg->started) {
        return;
    }
    wg->started = true;
}

inline void WorkgroupStopInterval(struct WorkGroup* wg)
{
    if (!wg->started) {
        return;
    }
    wg->started = false;
}

inline void WorkgroupJoin(struct WorkGroup* wg, int tid)
{
    (void)wg;
    (void)tid;
}

#endif /* QOS_FRAME_RTG */
}
#endif
