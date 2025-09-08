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
#include "cpp/qos_convert.h"
#include "dfx/log/ffrt_log_api.h"
#include "eu/qos_interface.h"

constexpr int ERROR_NUM = -1;

namespace ffrt {
int GetStaticQos(qos &static_qos)
{
    struct QosCtrlData data;
    int ret = FFRTQosGet(data);
    if (ret < 0 || data.staticQos < 0) {
        FFRT_SYSEVENT_LOGE("get static qos failed");
        return ERROR_NUM;
    }
    static_qos = static_cast<qos>(data.staticQos);
    return ret;
}

int GetDynamicQos(qos &dynamic_qos)
{
    struct QosCtrlData data;
    int ret = FFRTQosGet(data);
    if (ret < 0 || data.dynamicQos < 0) {
        FFRT_SYSEVENT_LOGE("get dynamic qos failed");
        return ERROR_NUM;
    }
    dynamic_qos = static_cast<qos>(data.dynamicQos);
    return ret;
}
}; // namespace ffrt