/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

#include "util/time_format.h"
#include <securec.h>
#include "dfx/log/ffrt_log_api.h"

namespace {
#if defined(__aarch64__)
const uint64_t tsc_base = ffrt::Arm64CntCt();
const std::chrono::steady_clock::time_point steady_base = std::chrono::steady_clock::now();
const uint64_t freq = ffrt::Arm64CntFrq();
#endif
}
namespace ffrt {
uint64_t ConvertTscToSteadyClockCount(uint64_t cntCt)
{
#if defined(__aarch64__)
    const uint64_t delta_tsc = cntCt - tsc_base;
    constexpr int ratio = 1000 * 1000;

    const uint64_t delta_micro = (delta_tsc * ratio) / freq;

    return static_cast<uint64_t>(std::chrono::time_point_cast<std::chrono::microseconds>(
        steady_base + std::chrono::microseconds(delta_micro)).time_since_epoch().count());
#else
    return cntCt;
#endif
}

uint64_t ConvertCntvctToUs(uint64_t cntCt)
{
    // 将获取到的CPU cycle数转成微秒数
#if defined(__aarch64__)
    uint64_t freq = Arm64CntFrq();
    constexpr int ratio = 1000 * 1000;

    return static_cast<uint64_t>((cntCt * ratio) / freq);
#else
    return cntCt;
#endif
}

uint64_t ConvertUsToCntvct(uint64_t time)
{
    // 将微秒数转成CPU cycle数
#if defined(__aarch64__)
    uint64_t freq = Arm64CntFrq();
    constexpr int ratio = 1000 * 1000;

    return static_cast<uint64_t>((time * freq) / ratio);
#else
    return time;
#endif
}

uint64_t TimeStampCntvct()
{
    // 在非arm环境下获取std的时间戳，单位微秒
    return static_cast<uint64_t>(std::chrono::time_point_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()).time_since_epoch().count());
}

std::string FormatDateToString(uint64_t timeStamp)
{
#if defined(__aarch64__)
    return FormatDateString4CntCt(timeStamp, MICROSECOND);
#else
    return FormatDateString4SteadyClock(timeStamp, MICROSECOND);
#endif
}

uint64_t Arm64CntFrq(void)
{
    uint64_t freq = 1;
#if defined(__aarch64__)
    asm volatile("mrs %0, cntfrq_el0" : "=r" (freq));
#endif
    return freq;
}

uint64_t Arm64CntCt(void)
{
    uint64_t tsc = 1;
#if defined(__aarch64__)
    asm volatile("mrs %0, cntvct_el0" : "=r" (tsc));
#endif
    return tsc;
}

std::string FormatDateString4SystemClock(const std::chrono::system_clock::time_point& timePoint,
    TimeUnitT timeUnit)
{
    constexpr int maxMsLength = 3;
    constexpr int msPerSecond = 1000;
    constexpr int datetimeStringLength = 80;
    constexpr int maxUsLength = 6;
    constexpr int usPerSecond = 1000 * 1000;

    std::string remainder;
    if (timeUnit == MICROSECOND) {
        auto tp = std::chrono::time_point_cast<std::chrono::microseconds>(timePoint);
        auto us = tp.time_since_epoch().count() % usPerSecond;
        remainder = std::to_string(us);
        if (remainder.length() < maxUsLength) {
            remainder = std::string(maxUsLength - remainder.length(), '0') + remainder;
        }
    } else {
        auto tp = std::chrono::time_point_cast<std::chrono::milliseconds>(timePoint);
        auto ms = tp.time_since_epoch().count() % msPerSecond;
        remainder = std::to_string(ms);
        if (remainder.length() < maxMsLength) {
            remainder = std::string(maxMsLength - remainder.length(), '0') + remainder;
        }
    }
    auto tt = std::chrono::system_clock::to_time_t(timePoint);
    struct tm curTime;
    if (memset_s(&curTime, sizeof(curTime), 0, sizeof(curTime)) != EOK) {
        FFRT_LOGE("Fail to memset");
        return "";
    }
    localtime_r(&tt, &curTime);
    char sysTime[datetimeStringLength];
    std::strftime(sysTime, sizeof(char) * datetimeStringLength, "%Y-%m-%d %H:%M:%S.", &curTime);
    return std::string(sysTime) + remainder;
}

std::string FormatDateString4SteadyClock(uint64_t steadyClockTimeStamp, TimeUnitT timeUnit)
{
    auto referenceTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    auto referenceTp = std::chrono::system_clock::now();

    std::chrono::microseconds us(static_cast<int64_t>(steadyClockTimeStamp - referenceTimeStamp));
    return FormatDateString4SystemClock(referenceTp + us, timeUnit);
}

std::string FormatDateString4CntCt(uint64_t cntCtTimeStamp, TimeUnitT timeUnit)
{
    constexpr int ratio = 1000 * 1000;

    int64_t referenceFreq = static_cast<int64_t>(Arm64CntFrq());
    if (referenceFreq == 0) {
        return "";
    }
    uint64_t referenceCntCt = Arm64CntCt();
    auto globalTp = std::chrono::system_clock::now();
    std::chrono::microseconds us(static_cast<int64_t>(cntCtTimeStamp - referenceCntCt) * ratio / referenceFreq);
    return FormatDateString4SystemClock(globalTp + us, timeUnit);
}
} // namespace ffrt
