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

#ifndef _SCPU_TASK_H_
#define _SCPU_TASK_H_

#include "tm/cpu_task.h"

namespace ffrt {
class SCPUEUTask : public CPUEUTask {
public:
    SCPUEUTask(const task_attr_private *attr, CPUEUTask *parent, const uint64_t &id);
    std::unordered_set<VersionCtx*> ins;
    std::unordered_set<VersionCtx*> outs;

    Dependence dependenceStatus {Dependence::DEPENDENCE_INIT};

    union {
        std::atomic_uint64_t submitDep; // dependency refcnt during task submit
        std::atomic_uint64_t waitDep; // dependency refcnt during task execute when wait api called
    } dataRefCnt {0};
    std::atomic_uint64_t childRefCnt {0}; // unfinished children refcnt

    inline void IncDepRef()
    {
        ++dataRefCnt.submitDep;
    }
    void DecDepRef();

    inline void IncChildRef()
    {
        ++(static_cast<SCPUEUTask*>(parent)->childRefCnt);
    }
    void DecChildRef();

    inline void IncWaitDataRef()
    {
        ++dataRefCnt.waitDep;
    }
    void DecWaitDataRef();
    void MultiDependenceAdd(Dependence depType);

public:
    void Finish() override;
};

class RootTask : public SCPUEUTask {
public:
    RootTask(const task_attr_private* attr, SCPUEUTask* parent, const uint64_t& id) : SCPUEUTask(attr, parent, id)
    {
    }
public:
    bool thread_exit = false;
};

class RootTaskCtxWrapper {
public:
    RootTaskCtxWrapper()
    {
        task_attr_private task_attr;
        root = new RootTask{&task_attr, nullptr, 0};
    }
    ~RootTaskCtxWrapper()
    {
        std::unique_lock<decltype(root->mutex_)> lck(root->mutex_);
        if (root->childRefCnt == 0) {
            lck.unlock();
            delete root;
        } else {
            root->thread_exit = true;
        }
    }
    CPUEUTask* Root()
    {
        return root;
    }
private:
    RootTask *root = nullptr;
};
} /* namespace ffrt */
#endif
