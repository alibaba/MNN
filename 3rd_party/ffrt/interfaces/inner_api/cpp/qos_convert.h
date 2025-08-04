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
#ifndef FFRT_API_CPP_QOS_CONVERT_H
#define FFRT_API_CPP_QOS_CONVERT_H
#include "c/type_def.h"

namespace ffrt {
/**
 * @brief get current thread static qos level
 */
int GetStaticQos(qos &static_qos);

/**
 * @brief get current thread dynamic qos level
 */
int GetDynamicQos(qos &dynamic_qos);
}; // namespace ffrt

#endif // FFRT_API_CPP_QOS_CONVERT_H