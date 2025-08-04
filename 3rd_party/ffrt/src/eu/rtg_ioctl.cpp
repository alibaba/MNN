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

#include "eu/rtg_ioctl.h"

#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <climits>
#include <sys/syscall.h>
#include <sys/ioctl.h>

#include "dfx/trace/ffrt_trace.h"
#include "dfx/log/ffrt_log_api.h"

constexpr int RTG_FRAME_START = 1;
constexpr int RTG_FRAME_END = 2;

constexpr int RTG_IPC_MAGIC = 0XBC;
constexpr int RTG_IPC_CMDID = 0xCD;
#define RTG_IPC_CMD _IOWR(RTG_IPC_MAGIC, RTG_IPC_CMDID, class RTGMsg)

namespace ffrt {
enum RTGCtrlCmd {
    CMD_CREATE_RTG,
    CMD_RELEASE_RTG,
    CMD_ADD_RTG_THREAD,
    CMD_DEL_RTG_THREAD,
    CMD_SET_GROUP_UTIL,
    CMD_SET_GROUP_FREQ,
    CMD_GET_THREAD_LOAD,
    CMD_GET_GROUP_LOAD,
    CMD_SET_GROUP_WINDOW_SIZE,
    CMD_SET_GROUP_WINDOW_ROLLOVER,
    CMD_SET_PREFERRED_CLUSTER,
    CMD_SET_INVALID_INTERVAL,
    CMD_ID_MAX,
};

static const char* FromatRTGCtrlCmd(uint32_t cmd)
{
    static const char* str[] = {
        "CMD_CREATE_RTG",
        "CMD_RELEASE_RTG",
        "CMD_ADD_RTG_THREAD",
        "CMD_DEL_RTG_THREAD",
        "CMD_SET_GROUP_UTIL",
        "CMD_SET_GROUP_FREQ",
        "CMD_GET_THREAD_LOAD",
        "CMD_GET_GROUP_LOAD",
        "CMD_SET_GROUP_WINDOW_SIZE",
        "CMD_SET_GROUP_WINDOW_ROLLOVER",
        "CMD_SET_PREFERRED_CLUSTER",
        "CMD_SET_INVALID_INTERVAL",
    };

    if (cmd >= CMD_ID_MAX) {
        return "Unknown";
    }

    return str[cmd];
}

class RTGCtrl::RTGMsg {
public:
    static RTGMsg Build(uint32_t cmd = 0, int32_t tgid = 0, int64_t data = 0)
    {
        return RTGMsg(cmd, tgid, data);
    }

    RTGMsg& Cmd(uint32_t var)
    {
        this->cmd = var;
        return *this;
    }

    RTGMsg& TGid(int32_t var)
    {
        this->tgid = var;
        return *this;
    }

    RTGMsg& Data(int64_t var)
    {
        this->data = var;
        return *this;
    }

    RTGMsg& InSize(uint32_t var)
    {
        this->in_size = var;
        return *this;
    }

    RTGMsg& OutSize(uint32_t var)
    {
        this->out_size = var;
        return *this;
    }

    RTGMsg& In(void* var)
    {
        this->in = var;
        return *this;
    }

    RTGMsg& Out(void* var)
    {
        this->out = var;
        return *this;
    }

    std::string Format() const
    {
        std::stringstream ss;

        auto formatBuf = [&](const char* head, const char* buf, uint32_t size) {
            if (!buf || size == 0) {
                return;
            }

            ss << head;
            for (uint32_t i = 0; i < size; ++i) {
                ss << static_cast<int>(buf[i]) << " ";
            }
        };

        ss << "cmd: " << FromatRTGCtrlCmd(cmd);
        ss << " tgid: " << tgid;
        ss << " data: " << data;

        ss << std::hex << std::uppercase << std::setfill('0') << std::setw(2);

        formatBuf(" in data: ", static_cast<const char*>(in), in_size);
        formatBuf(" out data: ", static_cast<const char*>(out), out_size);

        return ss.str();
    }

private:
    RTGMsg(uint32_t cmd, int32_t tgid, int64_t data)
        : cmd(cmd), tgid(tgid), data(data), in_size(0), out_size(0), in(nullptr), out(nullptr)
    {
    }

    uint32_t cmd;
    int32_t tgid;
    int64_t data;
    uint32_t in_size;
    uint32_t out_size;
    void* in;
    void* out;
};

RTGCtrl::RTGCtrl()
{
    char filePath[PATH_MAX];

    std::string fileName = "/proc/self/ffrt";
    if (realpath(fileName.c_str(), filePath) == nullptr) {
        FFRT_SYSEVENT_LOGE("Invalid file Path %s", fileName.c_str());
        return;
    }

    fd = open(filePath, O_RDWR);
    if (fd < 0) {
        FFRT_SYSEVENT_LOGE("Failed to open RTG, Ret %d", fd);
    }
}

RTGCtrl::~RTGCtrl()
{
    if (fd < 0) {
        return;
    }

    close(fd);
}

int RTGCtrl::GetThreadGroup()
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, CREATE_RTG);

    int tgid = -1;
    RTGMsg msg = RTGMsg::Build(CMD_CREATE_RTG).Out(&tgid).OutSize(sizeof(tgid));

    return RTGIOCtrl(msg) ? tgid : -1;
}

bool RTGCtrl::PutThreadGroup(int tgid)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, RELEASE_RTG);

    RTGMsg msg = RTGMsg::Build(CMD_RELEASE_RTG, tgid);
    return RTGIOCtrl(msg);
}

bool RTGCtrl::JoinThread(int tgid, pid_t tid)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, ADD_RTG_THREAD);

    RTGMsg msg = RTGMsg::Build(CMD_ADD_RTG_THREAD, tgid, tid);
    return RTGIOCtrl(msg);
}

bool RTGCtrl::RemoveThread(int tgid, pid_t tid)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, DEL_RTG_THREAD);

    RTGMsg msg = RTGMsg::Build(CMD_DEL_RTG_THREAD, tgid, tid);
    return RTGIOCtrl(msg);
}

bool RTGCtrl::UpdatePerfUtil(int tgid, int util)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, SET_GROUP_UTIL);

    RTGMsg msg = RTGMsg::Build(CMD_SET_GROUP_UTIL, tgid, util);
    return RTGIOCtrl(msg);
}

bool RTGCtrl::UpdatePerfFreq(int tgid, int64_t freq)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, SET_GROUP_FREQ);

    RTGMsg msg = RTGMsg::Build(CMD_SET_GROUP_FREQ, tgid, freq);
    return RTGIOCtrl(msg);
}

RTGLoadInfo RTGCtrl::UpdateAndGetLoad(int tgid, pid_t tid)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, GET_THREAD_LOAD);

    RTGLoadInfo info {0};
    RTGMsg msg = RTGMsg::Build(CMD_GET_THREAD_LOAD, tgid, tid).Out(&info).OutSize(sizeof(info));

    RTGIOCtrl(msg);

    FFRT_LOGI("Get Thread Load %llu Runtime %llu", info.load, info.runtime);

    return info;
}

RTGLoadInfo RTGCtrl::UpdateAndGetLoad(int tgid)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, GET_SERIAL_LOAD);

    RTGLoadInfo info {0};
    RTGMsg msg = RTGMsg::Build(CMD_GET_GROUP_LOAD, tgid).Out(&info).OutSize(sizeof(info));

    RTGIOCtrl(msg);

    FFRT_LOGI("Get Serial Load %llu Runtime %llu", info.load, info.runtime);

    return info;
}

bool RTGCtrl::SetGroupWindowSize(int tgid, uint64_t size)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, SET_GROUP_WINDOW_SIZE);

    RTGMsg msg = RTGMsg::Build(CMD_SET_GROUP_WINDOW_SIZE, tgid, size);
    return RTGIOCtrl(msg);
}

bool RTGCtrl::SetInvalidInterval(int tgid, uint64_t interval)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, SET_INVALID_INTERVAL);

    RTGMsg msg = RTGMsg::Build(CMD_SET_INVALID_INTERVAL, tgid, interval);
    return RTGIOCtrl(msg);
}

bool RTGCtrl::Begin(int tgid)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, SET_GROUP_WINDOW_ROLLOVER_BEGIN);

    RTGMsg msg = RTGMsg::Build(CMD_SET_GROUP_WINDOW_ROLLOVER, tgid, RTG_FRAME_START);
    return RTGIOCtrl(msg);
}

bool RTGCtrl::End(int tgid)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, SET_GROUP_WINDOW_ROLLOVER_END);

    RTGMsg msg = RTGMsg::Build(CMD_SET_GROUP_WINDOW_ROLLOVER, tgid, RTG_FRAME_END);
    return RTGIOCtrl(msg);
}

bool RTGCtrl::SetPreferredCluster(int tgid, int clusterId)
{
    FFRT_TRACE_SCOPE(TRACE_LEVEL1, SET_PREFERRED_CLUSTER);

    RTGMsg msg = RTGMsg::Build(CMD_SET_PREFERRED_CLUSTER, tgid, clusterId);
    return RTGIOCtrl(msg);
}

pid_t RTGCtrl::GetTID()
{
    return syscall(SYS_gettid);
}

bool RTGCtrl::RTGIOCtrl(RTGMsg& msg)
{
    int ret = ioctl(fd, RTG_IPC_CMD, &msg);
    if (ret < 0) {
        FFRT_SYSEVENT_LOGE("RTG IOCtrl Failed Ret:%d, %s\n", ret, msg.Format().c_str());
        return false;
    }

    FFRT_LOGD("RTG IOCtrl Success %s\n", msg.Format().c_str());

    return true;
}
}; // namespace ffrt