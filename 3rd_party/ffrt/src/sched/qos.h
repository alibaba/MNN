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

#ifndef FFRT_QOS_H
#define FFRT_QOS_H

#include "ffrt_inner.h"

constexpr unsigned char NR_QOS = 6;

namespace ffrt {

typedef int (*FuncQosMap)(int qos);
void SetFuncQosMap(FuncQosMap func);
FuncQosMap GetFuncQosMap(void);

int QoSMap(int qos);

typedef int (*FuncQosMax)(void);
void SetFuncQosMax(FuncQosMax func);
FuncQosMax GetFuncQosMax(void);

int QoSMax(void);

class QoS {
public:
    QoS(int qos = qos_default)
    {
        if (qos < static_cast<int>(qos_inherit)) {
            qos = qos_inherit;
        } else if (qos > static_cast<int>(qos_max)) {
            qos = qos_max;
        }
        qos_ = qos;
    }

    QoS(const QoS& qos) : qos_(qos())
    {
    }

    int operator()() const
    {
        return qos_;
    }

    operator int() const
    {
        return qos_;
    }

    static constexpr int Min()
    {
        return qos_background;
    }

    static int Max()
    {
        return qos_max + 1;
    }

    static constexpr int MaxNum()
    {
        return qos_max + 1;
    }

private:
    int qos_;
};
}; // namespace ffrt
#endif