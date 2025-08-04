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
#ifndef FFRT_RING_BUFFER_H
#define FFRT_RING_BUFFER_H

#include <iostream>
#include <cstring>
#include <mutex>
#include <securec.h>

namespace ffrt {

class FFRTRingBuffer {
public:
    FFRTRingBuffer(char *buf, uint32_t len) : buf_(buf), bufferLen_(len), writtenSize_(0)
    {
        memset_s(buf_, bufferLen_, 0, bufferLen_);
    }

    template<typename T>
    int Write(T data)
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        if (writtenSize_ + sizeof(T) > bufferLen_) {
            writtenSize_ = sizeof(T);
            return memcpy_s(buf_, bufferLen_, &data, sizeof(T));
        } else {
            int ret = memcpy_s(buf_ + writtenSize_, bufferLen_ - writtenSize_, &data, sizeof(T));
            writtenSize_ += sizeof(T);
            return ret;
        }
    }

    char *GetBuffer()
    {
        return buf_;
    }

    uint32_t GetBufferSize()
    {
        return bufferLen_;
    }

private:
    char *buf_;
    uint32_t bufferLen_;
    uint32_t writtenSize_;
    std::mutex bufferMutex_;
};
}
#endif // FFRT_RING_BUFFER_H
