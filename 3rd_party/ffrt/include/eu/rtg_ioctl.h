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

#ifndef FFRT_RTG_IOCTL_H
#define FFRT_RTG_IOCTL_H

#include <cstdint>

#include <sched.h>

namespace ffrt {
struct RTGLoadInfo {
    uint64_t load = 0;
    uint64_t runtime = 0;
};

class RTGCtrl {
    class RTGMsg;

public:
    static RTGCtrl& Instance()
    {
        static RTGCtrl ctrl;
        return ctrl;
    }

    bool Enabled() const
    {
        return fd >= 0;
    }

    int GetThreadGroup();
    bool PutThreadGroup(int tgid);
    bool JoinThread(int tgid, pid_t tid);
    bool RemoveThread(int tgid, pid_t tid);
    bool UpdatePerfUtil(int tgid, int util);
    bool UpdatePerfFreq(int tgid, int64_t freq);
    RTGLoadInfo UpdateAndGetLoad(int tgid, pid_t tid);
    RTGLoadInfo UpdateAndGetLoad(int tgid);
    bool SetGroupWindowSize(int tgid, uint64_t size);
    bool SetInvalidInterval(int tgid, uint64_t interval);

    bool Begin(int tgid);
    bool End(int tgid);

    bool SetPreferredCluster(int tgid, int clusterId);

    static pid_t GetTID();

private:
    RTGCtrl();
    ~RTGCtrl();

    bool RTGIOCtrl(RTGMsg& msg);

    int fd = -1;
};
}; // namespace ffrt

#endif