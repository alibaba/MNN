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

#ifndef _REF_FUNCTION_HEADER_H_
#define _REF_FUNCTION_HEADER_H_

#include <atomic>
#include "dfx/log/ffrt_log_api.h"
#include "c/type_def.h"

namespace ffrt {
enum FuncHeaderStatus {
    NOT_SUBMIT = 1,
    ALREADY_SUBMIT = 2,
};

void DestroyFunctionWrapper(ffrt_function_header_t* f, ffrt_function_kind_t kind);

class RefFunctionHeader {
public:
    RefFunctionHeader(ffrt_function_header_t* header) : functionHeader_(header) {}
    bool IncDeleteRef()
    {
        uint16_t exp = NOT_SUBMIT;
        uint16_t des = ALREADY_SUBMIT;
        return rc.compare_exchange_strong(exp, des, std::memory_order_relaxed);
    }
    void DecDeleteRef()
    {
        auto v = rc.fetch_sub(1);
        if (v == 1) {
            delete this;
        }
    }

    ffrt_function_header_t* functionHeader_ = nullptr;
    std::atomic_uint16_t rc = 1;
private:
    ~RefFunctionHeader()
    {
        ffrt::DestroyFunctionWrapper(functionHeader_, ffrt_function_kind_general);
    }
};
}
#endif