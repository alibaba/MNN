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

#ifndef FFRT_LOAD_PREDICTOR_H
#define FFRT_LOAD_PREDICTOR_H

#include <array>
#include <algorithm>

namespace ffrt {
template <typename T>
class LoadPredictor {
public:
    virtual ~LoadPredictor() = default;

    uint64_t GetPredictLoad() const
    {
        return static_cast<const T*>(this)->GetPredictLoadImpl();
    }

    void UpdateLoad(uint64_t load)
    {
        static_cast<T*>(this)->UpdateLoadImpl(load);
    }

    void Clear()
    {
        static_cast<T*>(this)->ClearImpl();
    }
};

class SimpleLoadPredictor : public LoadPredictor<SimpleLoadPredictor> {
    friend class LoadPredictor<SimpleLoadPredictor>;

public:
    SimpleLoadPredictor()
    {
        std::fill(loadHist.begin(), loadHist.end(), 0UL);
    }

private:
    uint64_t GetPredictLoadImpl() const
    {
        return maxLoad;
    }

    void UpdateLoadImpl(uint64_t load)
    {
        uint64_t sum = load;

        auto end = loadHist.rend() - 1;
        for (auto begin = loadHist.rbegin(); begin < end; ++begin) {
            *begin = *(begin + 1);
            sum += *begin;
        }
        *end = load;

        maxLoad = std::max({ sum / HIST_SIZE, loadHist[0], loadHist[1] });
    }

    void ClearImpl()
    {
        maxLoad = 0;
        std::fill(loadHist.begin(), loadHist.end(), 0UL);
    }

    static constexpr int HIST_SIZE = 5;
    std::array<uint64_t, HIST_SIZE> loadHist;

    uint64_t maxLoad = 0;
};
}; // namespace ffrt

#endif