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

#ifndef __OSAL_HPP__
#define __OSAL_HPP__

#include <string>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define API_ATTRIBUTE(attr) __attribute__(attr)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define likely(x)       __builtin_expect(!!(x), 1)

static inline unsigned int GetPid(void)
{
    return getpid();
}
static inline unsigned int GetTid(void)
{
    return syscall(SYS_gettid);
}
static inline std::string GetEnv(const char* name)
{
    char* val = std::getenv(name);
    if (val == nullptr) {
        return "";
    }
    return val;
}

static inline void GetProcessName(char* processName, int bufferLength)
{
    int fd = open("/proc/self/cmdline", O_RDONLY);
    if (fd != -1) {
        ssize_t ret = syscall(SYS_read, fd, processName, bufferLength - 1);
        if (ret != -1) {
            processName[ret] = 0;
        }

        syscall(SYS_close, fd);
    }
}
#endif