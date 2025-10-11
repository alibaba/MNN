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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include "sched/workgroup_internal.h"
#include "dfx/log/ffrt_log_api.h"
#include "task_client_adapter.h"


#if (defined(QOS_WORKER_FRAME_RTG) || defined(QOS_FRAME_RTG))
constexpr int HWC_UID = 3039;
constexpr int ROOT_UID = 0;
constexpr int RS_RTG_ID = 10;

namespace ffrt {
static int wgId = -1;
static WorkGroup* rsWorkGroup = nullptr;
static int wgCount = 0;
static std::mutex wgLock;

#if (defined(QOS_WORKER_FRAME_RTG))

void WorkgroupInit(struct WorkGroup* wg, uint64_t interval, int rtgId, int qos)
{
    wg->started = false;
    wg->interval = interval;
    wg->rtgId = rtgId;
    wg->qos = qos;
    wgId = rtgId;

    for (int i = 0; i < MAX_WG_THREADS; i++) {
        wg->tids[i] = -1;
    }
}

int FindThreadInWorkGroup(WorkGroup *workGroup, int tid)
{
    if (workGroup == nullptr) {
        FFRT_SYSEVENT_LOGE("[RSWorkGroup] find thread %{public}d in workGroup failed, workGroup is null", tid);
        return -1;
    }
    for (int i = 0;i < MAX_WG_THREADS; i++) {
        if (workGroup->tids[i] == tid) {
            return i;
        }
    }
    return -1;
}

bool InsertThreadInWorkGroup(WorkGroup *workGroup, int tid)
{
    if (workGroup == nullptr) {
        FFRT_SYSEVENT_LOGE("[RSWorkGroup] join thread %{public}d into workGroup failed, workGroup is null", tid);
        return false;
    }
    for (int i = 0; i < MAX_WG_THREADS; i++) {
        if (workGroup->tids[i] == -1) {
            workGroup->tids[i] = tid;
            return true;
        }
    }
    return false;
}

WorkGroup* CreateRSWorkGroup(uint64_t interval, int qos)
{
    IntervalReply rs;
    rs.rtgId = -1;
    rs.tid = -1;
    {
        std::lock_guard<std::mutex> lck(wgLock);
        if (rsWorkGroup == nullptr) {
            CTC_QUERY_INTERVAL(QUERY_RENDER_SERVICE_RENDER, rs);
            if (rs.rtgId > 0) {
                rsWorkGroup = new struct WorkGroup();
                if (rsWorkGroup == nullptr) {
                    FFRT_SYSEVENT_LOGE("[RSWorkGroup] rsWorkGroup malloc failed!");
                    return nullptr;
                }
                WorkgroupInit(rsWorkGroup, interval, rs.rtgId, qos);
                wgCount++;
            }
        }
    }
    return rsWorkGroup;
}

bool LeaveRSWorkGroup(int tid, int qos)
{
    std::lock_guard<std::mutex> lck(wgLock);
    if (rsWorkGroup == nullptr || rsWorkGroup->qos != qos) {
        FFRT_LOGI("[RSWorkGroup] LeaveRSWorkGroup rsWorkGroup is null ,tid:%{public}d", tid);
        return false;
    }
    int existIndex = FindThreadInWorkGroup(rsWorkGroup, tid);
    if (existIndex != -1) {
        rsWorkGroup->tids[existIndex] = -1;
    }
    FFRT_LOGI("[RSWorkGroup] LeaveRSWorkGroup ,tid: %{public}d, existIndex: %{public}d", tid, existIndex);
    return true;
}

bool JoinRSWorkGroup(int tid, int qos)
{
    std::lock_guard<std::mutex> lck(wgLock);
    if (rsWorkGroup == nullptr || rsWorkGroup->qos != qos) {
        FFRT_SYSEVENT_LOGE("[RSWorkGroup] join thread %{public}d into RSWorkGroup failed; Create RSWorkGroup first",
            tid);
        return false;
    }
    int existIndex = FindThreadInWorkGroup(rsWorkGroup, tid);
    if (existIndex == -1) {
        IntervalReply rs;
        rs.rtgId = -1;
        rs.tid = tid;
        CTC_QUERY_INTERVAL(QUERY_RENDER_SERVICE, rs);
        if (rs.rtgId > 0) {
            bool success = InsertThreadInWorkGroup(rsWorkGroup, tid);
            if (!success) {
                return false;
            }
        }
    }
    FFRT_LOGI("[RSWorkGroup] update thread %{public}d success", tid);
    return true;
}

bool DestoryRSWorkGroup(int qos)
{
    std::lock_guard<std::mutex> lck(wgLock);
    if (rsWorkGroup != nullptr && rsWorkGroup->qos == qos) {
        delete rsWorkGroup;
        rsWorkGroup = nullptr;
        wgId = -1;
        return true;
    }
    return false;
}
#endif

#if defined(QOS_FRAME_RTG)
bool JoinWG(int tid, int qos)
{
    if (wgId < 0) {
        if (wgCount > 0) {
            FFRT_SYSEVENT_LOGE("[WorkGroup] interval is unavailable");
        }
        return false;
    }
    int uid = getuid();
    if (uid == RS_UID) {
        return JoinRSWorkGroup(tid, qos);
    }
    int addRet = AddThreadToRtgAdapter(tid, wgId, 0);
    if (addRet == 0) {
        FFRT_LOGI("[WorkGroup] update thread %{public}d success", tid);
    } else {
        FFRT_SYSEVENT_LOGE("[WorkGroup] update thread %{public}d failed, return %{public}d", tid, addRet);
    }
    return true;
}

bool LeaveWG(int tid, int qos)
{
    int uid = getuid();
    if (uid == RS_UID) {
        return LeaveRSWorkGroup(tid, qos);
    }
    return false;
}

struct WorkGroup* WorkgroupCreate(uint64_t interval, int qos)
{
    int rtgId = -1;
    int uid = getuid();
    int num = 0;
    
    if (uid == RS_UID) {
        CreateRSWorkGroup(interval);
        return rsWorkGroup;
    }

    if (rtgId < 0) {
        FFRT_SYSEVENT_LOGE("[WorkGroup] create rtg group %d failed", rtgId);
        return nullptr;
    }
    FFRT_LOGI("[WorkGroup] create rtg group %d success", rtgId);

    WorkGroup* wg = nullptr;
    wg = new struct WorkGroup();
    if (wg == nullptr) {
        FFRT_SYSEVENT_LOGE("[WorkGroup] workgroup malloc failed!");
        return nullptr;
    }
    WorkgroupInit(wg, interval, rtgId);
    {
        std::lock_guard<std::mutex> lck(wgLock);
        wgCount++;
    }
    return wg;
}

int WorkgroupClear(struct WorkGroup* wg)
{
    if (wg == nullptr) {
        FFRT_SYSEVENT_LOGE("[WorkGroup] input workgroup is null");
        return 0;
    }
    int uid = getuid();
    if (uid == RS_UID) {
        return DestoryRSWorkGroup(wg->qos);
    }
    int ret = -1;
    if (uid != RS_UID) {
        ret = DestroyRtgGrpAdapter(wg->rtgId);
        if (ret != 0) {
            FFRT_SYSEVENT_LOGE("[WorkGroup] destroy rtg group failed");
        } else {
            std::lock_guard<std::mutex> lck(wgLock);
            wgCount--;
        }
    }
    delete wg;
    wg = nullptr;
    return ret;
}

void WorkgroupStartInterval(struct WorkGroup* wg)
{
    if (wg == nullptr) {
        FFRT_SYSEVENT_LOGE("[WorkGroup] input workgroup is null");
        return;
    }

    if (wg->started) {
        FFRT_LOGW("[WorkGroup] already start");
        return;
    }

    if (BeginFrameFreqAdapter(0) == 0) {
        wg->started = true;
    } else {
        FFRT_LOGE("[WorkGroup] start rtg(%d) work interval failed", wg->rtgId);
    }
}

void WorkgroupStopInterval(struct WorkGroup* wg)
{
    if (wg == nullptr) {
        FFRT_LOGE("[WorkGroup] input workgroup is null");
        return;
    }

    if (!wg->started) {
        FFRT_LOGW("[WorkGroup] already stop");
        return;
    }

    int ret = EndFrameFreqAdapter(0);
    if (ret == 0) {
        wg->started = false;
    } else {
        FFRT_LOGE("[WorkGroup] stop rtg(%d) work interval failed", wg->rtgId);
    }
}

void WorkgroupJoin(struct WorkGroup* wg, int tid)
{
    if (wg == nullptr) {
        FFRT_LOGE("[WorkGroup] input workgroup is null");
        return;
    }
    int uid = getuid();
    FFRT_LOGI("[WorkGroup] %s uid = %d rtgid = %d", __func__, uid, wg->rtgId);
    if (uid == RS_UID) {
        IntervalReply rs;
        rs.tid = tid;
        CTC_QUERY_INTERVAL(QUERY_RENDER_SERVICE, rs);
        FFRT_LOGI("[WorkGroup] join thread %{public}ld", tid);
        return;
    }
    int addRet = AddThreadToRtgAdapter(tid, wg->rtgId, 0);
    if (addRet == 0) {
        FFRT_LOGI("[WorkGroup] join thread %{public}ld success", tid);
    } else {
        FFRT_LOGE("[WorkGroup] join fail with %{public}d threads for %{public}d", addRet, tid);
    }
}

#endif /* QOS_FRAME_RTG */
}

#endif
