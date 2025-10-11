/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
#include <algorithm>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include "eu/cpu_worker.h"
#include "eu/osattr_manager.h"
#include "eu/qos_interface.h"
#include "util/name_manager.h"
#include "util/ffrt_facade.h"
#include "internal_inc/osal.h"
#include "util/white_list.h"

namespace {
constexpr int CFS_PRIO_MIN = 1;
constexpr int CFS_PRIO_DEFAULT = 20;
constexpr int CFS_PRIO_MAX = 40;
constexpr int VIP_PRIO_MIN = 41;
constexpr int VIP_PRIO_MAX = 50;
constexpr int RT_PRIO_MIN = 51;
constexpr int RT_PRIO_MAX = 139;
}

namespace ffrt {
void CPUWorker::WorkerSetup()
{
    static std::atomic<int> threadIndex[QoS::MaxNum()] = {0};
    std::string qosStr = std::to_string(qos());
    int index = threadIndex[qos()];
    std::string threadName = std::string(WORKER_THREAD_NAME_PREFIX) + qosStr +
        std::string(WORKER_THREAD_SYMBOL) + std::to_string(index);
    if (qosStr == "") {
        FFRT_SYSEVENT_LOGE("ffrt threadName qos[%d] index[%d]", qos(), index);
    }
    threadIndex[qos()].fetch_add(1, std::memory_order_relaxed);
    pthread_setname_np(pthread_self(), threadName.c_str());
}

int SetCpuAffinity(unsigned long affinity, int tid)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (unsigned long i = 0; i < sizeof(affinity) * 8; i++) {
        if ((affinity & (static_cast<unsigned long>(1) << i)) != 0) {
            CPU_SET(i, &mask);
        }
    }
    int ret = syscall(__NR_sched_setaffinity, tid, sizeof(mask), &mask);
    if (ret < 0) {
        FFRT_SYSEVENT_LOGW("set qos affinity failed for tid %d\n", tid);
    }
    return ret;
}

void SetDefaultThreadAttr(CPUWorker* thread, const QoS& qos)
{
    if (qos() <= qos_max) {
        FFRTQosApplyForOther(qos(), thread->Id());
        FFRT_LOGD("qos apply tid[%d] level[%d]\n", thread->Id(), qos());
    }
}

void CPUWorker::SetThreadAttr(const QoS& newQos)
{
    SetDefaultThreadAttr(this, newQos);
}
}