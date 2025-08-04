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

#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "types.h"

namespace ffrt {
constexpr int DEFAULT_MINCONCURRENCY = 4;
constexpr int INTERACTIVE_MAXCONCURRENCY = USE_COROUTINE ? 8 : 40000;
constexpr int DEFAULT_MAXCONCURRENCY = USE_COROUTINE ? 8 : 80000;
constexpr int DEFAULT_HARDLIMIT = USE_COROUTINE ? 16 : 128;
constexpr int QOS_WORKER_MAXNUM = (8 * 16);

class GlobalConfig {
public:
    GlobalConfig(const GlobalConfig&) = delete;

    GlobalConfig& operator=(const GlobalConfig&) = delete;

    ~GlobalConfig() {}

    static inline GlobalConfig& Instance()
    {
        static GlobalConfig cfg;
        return cfg;
    }

    void setCpuWorkerNum(const QoS& qos, int num)
    {
        if ((num <= 0) || (num > DEFAULT_MAXCONCURRENCY)) {
            num = DEFAULT_MAXCONCURRENCY;
        }
        this->cpu_worker_num[qos()] = static_cast<size_t>(num);
    }

    size_t getCpuWorkerNum(const QoS& qos)
    {
        return this->cpu_worker_num[qos()];
    }

private:
    GlobalConfig()
    {
        for (auto qos = QoS::Min(); qos < QoS::Max(); ++qos) {
            if (qos == static_cast<int>(qos_user_interactive)) {
                this->cpu_worker_num[qos] = INTERACTIVE_MAXCONCURRENCY;
            } else {
                this->cpu_worker_num[qos] = DEFAULT_MAXCONCURRENCY;
            }
        }
    }

    size_t cpu_worker_num[QoS::MaxNum()];
};
}

#endif /* GLOBAL_CONFIG_H */
