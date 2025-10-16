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

#ifndef __FFRT_SYSEVENT_H__
#define __FFRT_SYSEVENT_H__
#include <string>
#include "dfx/trace_record/ffrt_trace_record.h"
namespace ffrt {
#ifdef FFRT_SEND_EVENT
bool IsBeta();
void TaskBlockInfoReport(const long long passed, const std::string& task_label, int qos, uint64_t freq);
void TaskTimeoutReport(std::stringstream& ss, const std::string& processName, const std::string& senarioName);
void TrafficOverloadReport(std::stringstream& ss, const std::string& senarioName);
void WorkerEscapeReport(const std::string& processName, int qos, size_t totalNum);
#endif
}
#endif