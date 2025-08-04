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

#ifndef OS_ATTR_MANAGER_H
#define OS_ATTR_MANAGER_H
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include "ffrt_inner.h"
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "dfx/log/ffrt_log_api.h"

namespace ffrt {
const std::string cpuctlGroupIvePath = "/dev/cpuctl/cam2stage";
const std::string cpusetGroupIvePath = "/dev/cpuset/cam2stage";
const std::string cpuThreadNode = "/tasks";
const std::string cpuSharesNode = "/cpu.shares";
const std::string cpuMapNode = "/cpus";
#ifdef OHOS_STANDARD_SYSTEM
const std::string cpuUclampminNode = "/load.min";
const std::string cpuUclampmaxNode = "/load.max";
#else
const std::string cpuUclampminNode = "/cpu.uclamp.min";
const std::string cpuUclampmaxNode = "/cpu.uclamp.max";
const std::string cpuLatencyniceNode = "/cpu.latency.nice";
const std::string cpuvipprioNode = "/cpu.vip_prio";
#endif
constexpr int PATH_MAX_LENS = 4096;
constexpr int CGROUP_SHARES_MIN = 2;
constexpr int CGROUP_SHARES_MAX = 262144;
constexpr int CGROUP_LAT_NICE_MIN = -20;
constexpr int CGROUP_LAT_NICE_MAX = 19;
constexpr int CGROUP_UCLAMP_MIN = 0;
constexpr int CGROUP_UCLAMP_MAX = 100;
constexpr int CGROUP_VIPPRIO_MIN = 0;
constexpr int CGROUP_VIPPRIO_MAX = 10;
class OSAttrManager {
public:
    OSAttrManager() {}
    ~OSAttrManager() {}

    static inline OSAttrManager* Instance()
    {
        static OSAttrManager instance;
        return &instance;
    }

    bool CheckSchedAttrPara(const std::string &name, int min, int max, int paraValue);
    int UpdateSchedAttr(const QoS& qos, ffrt_os_sched_attr *attr);
    void SetCGroupCtlPara(const std::string &name, int32_t value);
    void SetCGroupSetPara(const std::string &name, const std::string &value);
    void SetTidToCGroup(int32_t pid);
    void SetTidToCGroupPrivate(const std::string &filename, int32_t pid);
    template <typename T>
    void SetCGroupPara(const std::string &filename, T& value)
    {
        char filePath[PATH_MAX_LENS] = {0};
        if (filename.empty()) {
            FFRT_LOGE("[cgroup_ctrl] invalid para, filename is empty");
            return;
        }

        if ((strlen(filename.c_str()) > PATH_MAX_LENS) || (realpath(filename.c_str(), filePath) == nullptr)) {
            FFRT_LOGE("[cgroup_ctrl] invalid file path:%s, error:%s\n", filename.c_str(), strerror(errno));
            return;
        }

        int32_t fd = open(filePath, O_RDWR);
        if (fd < 0) {
            FFRT_LOGE("[cgroup_ctrl] fail to open filePath:%s", filePath);
            return;
        }

        std::string valueStr;
        if constexpr (std::is_same<T, int32_t>::value) {
            valueStr = std::to_string(value);
        } else if constexpr (std::is_same<T, const std::string>::value) {
            valueStr = value;
        } else {
            FFRT_LOGE("[cgroup_ctrl] invalid value type\n");
            close(fd);
            return;
        }

        int32_t ret = write(fd, valueStr.c_str(), valueStr.size());
        if (ret < 0) {
            FFRT_LOGE("[cgroup_ctrl] fail to write path:%s valueStr:%s to fd:%d, errno:%d",
                filePath, valueStr.c_str(), fd, errno);
            close(fd);
            return;
        }

        const uint32_t bufferLen = 20;
        std::array<char, bufferLen> buffer {};
        int32_t count = read(fd, buffer.data(), bufferLen);
        if (count <= 0) {
            FFRT_LOGE("[cgroup_ctrl] fail to read value:%s to fd:%d, errno:%d", buffer.data(), fd, errno);
        } else {
            FFRT_LOGI("[cgroup_ctrl] success to read %s buffer:%s", filePath, buffer.data());
        }
        close(fd);
    }
};
} // namespace ffrt

#endif /* OS_ATTR_MANAGER_H */