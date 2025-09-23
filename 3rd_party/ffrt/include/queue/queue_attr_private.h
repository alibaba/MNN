/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QUEUE_ATTR_PRIVATE_H
#define QUEUE_ATTR_PRIVATE_H
#include <string>
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif

namespace ffrt {
class queue_attr_private {
public:
    queue_attr_private()
        : qos_(qos_default)
    {
    }

    explicit queue_attr_private(const queue_attr attr)
        : qos_(attr.qos()),
          threadMode_(attr.thread_mode())
    {
    }

    int qos_;
    uint64_t timeout_ = 0;
    int maxConcurrency_ = 1;
    ffrt_function_header_t* timeoutCb_ = nullptr;
    bool threadMode_ = false;
};
}
#endif