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

#ifndef FFRT_THREAD_GROUP_H
#define FFRT_THREAD_GROUP_H

#include "eu/rtg_ioctl.h"

namespace ffrt {
class ThreadGroup {
public:
    int Id() const
    {
        return tgid;
    }

    bool Enabled() const
    {
        return tgid >= 0 && RTGCtrl::Instance().Enabled();
    }

    bool Init()
    {
        if (Enabled()) {
            return true;
        }

        tgid = RTGCtrl::Instance().GetThreadGroup();
        return tgid >= 0;
    }

    bool Release()
    {
        if (!Enabled()) {
            return true;
        }

        if (!RTGCtrl::Instance().PutThreadGroup(tgid)) {
            return false;
        }

        tgid = -1;
        return true;
    }

    void Begin()
    {
        if (!Enabled() || isBegin()) {
            return;
        }

        isbegin = true;
        RTGCtrl::Instance().Begin(tgid);
    }

    void End()
    {
        if (!Enabled() || !isBegin()) {
            return;
        }
        RTGCtrl::Instance().End(tgid);
        isbegin = false;
    }

    bool Join()
    {
        if (!Enabled()) {
            return false;
        }
        return RTGCtrl::Instance().JoinThread(tgid, RTGCtrl::GetTID());
    }

    bool Join(pid_t tid)
    {
        if (!Enabled()) {
            return false;
        }
        return RTGCtrl::Instance().JoinThread(tgid, tid);
    }

    bool Leave()
    {
        if (!Enabled()) {
            return false;
        }
        return RTGCtrl::Instance().RemoveThread(tgid, RTGCtrl::GetTID());
    }

    bool Leave(pid_t tid)
    {
        if (!Enabled()) {
            return false;
        }
        return RTGCtrl::Instance().RemoveThread(tgid, tid);
    }

    bool UpdateFreq(int64_t freq)
    {
        if (!Enabled()) {
            return false;
        }
        return RTGCtrl::Instance().UpdatePerfFreq(tgid, freq);
    }

    bool UpdateUitl(int64_t util)
    {
        if (!Enabled()) {
            return false;
        }
        return RTGCtrl::Instance().UpdatePerfUtil(tgid, util);
    }

    RTGLoadInfo GetLoad()
    {
        if (!Enabled()) {
            return RTGLoadInfo();
        }
        return RTGCtrl::Instance().UpdateAndGetLoad(tgid);
    }

    RTGLoadInfo GetLoad(pid_t tid)
    {
        if (!Enabled()) {
            return RTGLoadInfo();
        }
        return RTGCtrl::Instance().UpdateAndGetLoad(tgid, tid);
    }

    bool SetWindowSize(uint64_t size)
    {
        if (!Enabled()) {
            return false;
        }
        return RTGCtrl::Instance().SetGroupWindowSize(tgid, size);
    }

    bool SetInvalidInterval(uint64_t invalidMs)
    {
        if (!Enabled()) {
            return false;
        }
        return RTGCtrl::Instance().SetInvalidInterval(tgid, invalidMs);
    }

    static pid_t GetTID()
    {
        return RTGCtrl::GetTID();
    }

    bool isBegin()
    {
        return isbegin;
    }
private:
    int tgid = -1;
    bool isbegin = false;
};
}; // namespace ffrt

#endif