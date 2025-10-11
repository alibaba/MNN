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
#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
#include <dlfcn.h>
#include <sstream>
#include "unwinder.h"
#include "backtrace_local.h"
#endif
#include <securec.h>
#include "dfx/bbox/bbox.h"
#include "internal_inc/osal.h"
#include "dfx/log/ffrt_log_api.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "util/common_const.h"
#include "dump.h"

#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
using namespace OHOS::HiviewDFX;
#endif
namespace {
ffrt::HiTraceIdStruct traceid;
ffrt_task_timeout_cb timeoutCb = nullptr;

void HitraceChainCbWrap(uint64_t gid, const char *msg, uint32_t size)
{
#ifdef FFRT_ENABLE_HITRACE_CHAIN
    if (traceid.valid == ffrt::HITRACE_ID_VALID) {
        ffrt::TraceChainAdapter::Instance().HiTraceChainRestoreId(&traceid);
    }
    timeoutCb(gid, msg, size);
    ffrt::TraceChainAdapter::Instance().HiTraceChainClearId();
#endif
}
}
namespace ffrt {
constexpr uint32_t DEFAULT_TIMEOUT_MS = 30000;
struct TimeoutCfg {
    static inline TimeoutCfg* Instance()
    {
        static TimeoutCfg inst;
        return &inst;
    }

    uint32_t timeout = DEFAULT_TIMEOUT_MS;
    ffrt_task_timeout_cb callback = nullptr;
};

#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
void DumpTask(CoTask* task, std::string& stackInfo, uint8_t flag)
{
    ucontext_t ctx;

    if (ExecuteCtx::Cur()->task == task || task == nullptr) {
        if (flag == 0) {
            OHOS::HiviewDFX::PrintTrace(-1);
        } else {
            OHOS::HiviewDFX::GetBacktrace(stackInfo, false);
        }
        return;
    } else {
        memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
#if defined(__aarch64__)
        ctx.uc_mcontext.regs[REG_AARCH64_X29] = task->coRoutine->ctx.storage[10];
        ctx.uc_mcontext.sp = task->coRoutine->ctx.storage[13];
        ctx.uc_mcontext.pc = task->coRoutine->ctx.storage[11];
#elif defined(__x86_64__)
        ctx.uc_mcontext.gregs[REG_RBX] = task->coRoutine->ctx.storage[0];
        ctx.uc_mcontext.gregs[REG_RBP] = task->coRoutine->ctx.storage[1];
        ctx.uc_mcontext.gregs[REG_RSP] = task->coRoutine->ctx.storage[6];
        ctx.uc_mcontext.gregs[REG_RIP] = *(reinterpret_cast<greg_t *>(ctx.uc_mcontext.gregs[REG_RSP] - 8));
#elif defined(__arm__)
        ctx.uc_mcontext.arm_sp = task->coRoutine->ctx.storage[0]; /* sp */
        ctx.uc_mcontext.arm_pc = task->coRoutine->ctx.storage[1]; /* pc */
        ctx.uc_mcontext.arm_lr = task->coRoutine->ctx.storage[1]; /* lr */
        ctx.uc_mcontext.arm_fp = task->coRoutine->ctx.storage[10]; /* fp */
#endif
    }

    auto co = task->coRoutine;
    uintptr_t stackBottom = reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(co) + sizeof(CoRoutine) - 8);
    uintptr_t stackTop = static_cast<uintptr_t>(stackBottom + co->stkMem.size);
    auto unwinder = std::make_shared<Unwinder>();
    auto regs = DfxRegs::CreateFromUcontext(ctx);
    unwinder->SetRegs(regs);
    UnwindContext context;
    context.pid = UNWIND_TYPE_LOCAL;
    context.regs = regs;
    context.maps = unwinder->GetMaps();
    context.stackCheck = false;
    context.stackBottom = stackBottom;
    context.stackTop = stackTop;
    bool resFlag = unwinder->Unwind(&context);
    if (!resFlag) {
        FFRT_LOGE("Call Unwind failed");
        return;
    }
    std::ostringstream ss;
    auto frames = unwinder->GetFrames();
    if (flag != 0) {
        ss << Unwinder::GetFramesStr(frames);
        ss << std::endl;
        stackInfo = ss.str();
        return;
    }
    FFRT_LOGE("%s", Unwinder::GetFramesStr(frames).c_str());
}
#endif
}

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

API_ATTRIBUTE((visibility("default")))
int dump_info_all(char *buf, uint32_t len)
{
#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
    if (FFRTIsWork()) {
        std::string dumpInfo;
        dumpInfo += GetDumpPreface();
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        dumpInfo += SaveTaskCounterInfo();
#endif
        dumpInfo += SaveKeyInfo();
        dumpInfo += SaveWorkerStatusInfo();
        dumpInfo += SaveNormalTaskStatusInfo();
        dumpInfo += SaveQueueTaskStatusInfo();
        dumpInfo += SaveTimeoutTaskInfo();
        dumpInfo += SaveQueueTrafficRecordInfo();
        if (dumpInfo.length() > (len - 1)) {
            FFRT_LOGW("dumpInfo exceeds the buffer length, info length:%d, input len:%u", dumpInfo.length(), len);
        }
        return snprintf_s(buf, len, len - 1, "%s", dumpInfo.c_str());
    } else {
        return snprintf_s(buf, len, len - 1, "|-> FFRT has done all tasks, pid: %u \n", GetPid());
    }
#else
    return -1;
#endif
}

API_ATTRIBUTE((visibility("default")))
int ffrt_dump(ffrt_dump_cmd_t cmd, char *buf, uint32_t len)
{
    FFRT_COND_RETURN_ERROR(buf == nullptr || len < 1, -1, "buf is nullptr or len is less than 1, len %u", len);
    switch (cmd) {
        case DUMP_INFO_ALL: {
            return dump_info_all(buf, len);
        }
        case DUMP_TASK_STATISTIC_INFO: {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
            return ffrt::FFRTTraceRecord::StatisticInfoDump(buf, len);
#else
            return -1;
#endif
        }
        case DUMP_START_STAT: {
            return ffrt::FFRTTraceRecord::StatEnable(buf, len);
        }
        case DUMP_STOP_STAT: {
            return ffrt::FFRTTraceRecord::StatsDisable(buf);
        }
        default: {
            FFRT_LOGE("ffrt_dump unsupport cmd[%d]", cmd);
        }
    }
    return -1;
}

API_ATTRIBUTE((visibility("default")))
ffrt_task_timeout_cb ffrt_task_timeout_get_cb(void)
{
    return ffrt::TimeoutCfg::Instance()->callback;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_timeout_set_cb(ffrt_task_timeout_cb cb)
{
    FFRT_COND_DO_ERR((cb == nullptr), return, "input invalid, cb is nullptr");
#ifdef FFRT_ENABLE_HITRACE_CHAIN
    if (ffrt::TraceChainAdapter::Instance().HiTraceChainGetId().valid == ffrt::HITRACE_ID_VALID) {
        traceid = ffrt::TraceChainAdapter::Instance().HiTraceChainCreateSpan();
        timeoutCb = cb;
        ffrt::TimeoutCfg::Instance()->callback = HitraceChainCbWrap;
    } else {
        ffrt::TimeoutCfg::Instance()->callback = cb;
    }
#else
    ffrt::TimeoutCfg::Instance()->callback = cb;
#endif
}

API_ATTRIBUTE((visibility("default")))
uint32_t ffrt_task_timeout_get_threshold(void)
{
    return ffrt::TimeoutCfg::Instance()->timeout;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_timeout_set_threshold(uint32_t threshold_ms)
{
    if (threshold_ms < ONE_THOUSAND) {
        ffrt::TimeoutCfg::Instance()->timeout = ONE_THOUSAND;
        return;
    }
    ffrt::TimeoutCfg::Instance()->timeout = threshold_ms;
}
#ifdef __cplusplus
}
#endif
