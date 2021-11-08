//
//  Profiler.cpp
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include <algorithm>
#include <string>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif
#include "Profiler.hpp"
#include "core/Macro.h"

#define MFLOPS (1e6)

namespace MNN {
    
static inline int64_t getTime() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

static std::string toString(float value) {
    char typeString[100] = {};
    sprintf(typeString, "%f", value);
    return std::string(typeString);
}

static std::string toString(const std::vector<int>& shape) {
    char content[100] = {};
    auto current      = content;
    for (auto s : shape) {
        current = current + sprintf(current, "%d,", s);
    }
    return std::string(current);
}

Profiler* Profiler::gInstance = nullptr;
Profiler* Profiler::getInstance() {
    if (gInstance == nullptr) {
        gInstance = new Profiler;
    }
    return gInstance;
}

Profiler::Record& Profiler::getTypedRecord(const OperatorInfo* op) {
    auto typeStr = op->type();
    auto iter = mMapByType.find(typeStr);
    if (iter != mMapByType.end()) {
        return iter->second;
    }

    // create new
    mMapByType.insert(std::make_pair(typeStr, Record()));
    Record& record     = mMapByType.find(typeStr)->second;
    record.costTime    = 0.0f;
    record.calledTimes = 0;
    record.type        = op->type();
    record.flops       = 0.0f;

    return record;
}

Profiler::Record& Profiler::getNamedRecord(const OperatorInfo* op) {
    auto name = op->name();
    auto iter = mMapByName.find(name);
    if (iter != mMapByName.end()) {
        return iter->second;
    }

    // create new
    mMapByName.insert(std::make_pair(name, Record()));
    Record& record     = mMapByName.find(name)->second;
    record.costTime    = 0.0f;
    record.name        = op->name();
    record.type        = op->type();
    record.flops       = 0.0f;

    return record;
}

void Profiler::start(const OperatorInfo* info) {
    mStartTime = getTime();
    mTotalMFlops += info->flops();
    auto& typed = getTypedRecord(info);
    typed.calledTimes++;
    typed.flops += info->flops();
    auto& named = getNamedRecord(info);
    named.flops += info->flops();
}

void Profiler::end(const OperatorInfo* info) {
    mEndTime   = getTime();
    float cost = (float)(mEndTime - mStartTime) / 1000.0f;
    mMapByType[info->type()].costTime += cost;
    mMapByName[info->name()].costTime += cost;
    mTotalTime += cost;
}

static void printTable(const char* title, const std::vector<std::string>& header,
                       const std::vector<std::vector<std::string>>& data) {
    MNN_PRINT("%s\n", title);

    // calc column width
    std::vector<size_t> maxLength(header.size());
    for (int i = 0; i < header.size(); ++i) {
        size_t max = header[i].size();
        for (auto& row : data) {
            max = std::max(max, row[i].size());
        }
        maxLength[i] = max + 1;
    }

    // print header
    for (int i = 0; i < header.size(); ++i) {
        auto expand = header[i];
        expand.resize(maxLength[i], ' ');
        MNN_PRINT("%s\t", expand.c_str());
    }
    MNN_PRINT("\n");
    
    // print rows
    for (auto& row : data) {
        for (int i = 0; i < header.size(); ++i) {
            auto expand = row[i];
            expand.resize(maxLength[i], ' ');
            MNN_PRINT("%s\t", expand.c_str());
        }
        MNN_PRINT("\n");
    }
}
    
void Profiler::printTimeByType(int loops) {
    // sort by time cost
    std::vector<std::pair<float, std::string>> sorted;
    for (auto iter : mMapByType) {
        sorted.push_back(std::make_pair(iter.second.costTime, iter.first));
    }
    std::sort(sorted.begin(), sorted.end());
    
    // fill in columns
    const std::vector<std::string> header = {"Node Type", "Avg(ms)", "%", "Called times", "Flops Rate"};
    std::vector<std::vector<std::string>> rows;
    for (auto iter : sorted) {
        auto record = mMapByType.find(iter.second)->second;
        std::vector<std::string> columns;
        columns.push_back(iter.second);
        columns.push_back(toString(record.costTime / (float)loops));
        columns.push_back(toString((record.costTime / (float)mTotalTime) * 100));
        columns.push_back(toString(record.calledTimes / loops));
        columns.push_back(toString((record.flops / (float)mTotalMFlops) * 100));
        rows.emplace_back(columns);
    }
    printTable("Sort by time cost !", header, rows);
    float totalAvgTime = mTotalTime / (float)loops;
    MNN_PRINT("total time : %f ms, total mflops : %f \n", totalAvgTime, mTotalMFlops / loops);
}

void Profiler::printTimeByName(int loops) {
    const std::vector<std::string> header = {"Node Name", "Op Type", "Avg(ms)", "%", "Flops Rate"};
    std::vector<std::vector<std::string>> rows;
    // sort by name
    for (auto iter: mMapByName) {
        auto record = iter.second;
        std::vector<std::string> columns;
        columns.push_back(iter.first);
        columns.push_back(record.type);
        columns.push_back(toString(record.costTime / (float)loops));
        columns.push_back(toString((record.costTime / (float)mTotalTime) * 100));
        columns.push_back(toString((record.flops / (float)mTotalMFlops) * 100));
        rows.emplace_back(columns);
    }
    printTable("Sort by node name !", header, rows);
}
void Profiler::printSlowOp(const std::string& type, int topK, float rate) {
    MNN_PRINT("Print <=%d slowest Op for %s, larger than %.2f\n", topK, type.c_str(), rate * 100.0f);
    std::vector<std::pair<std::string, float>> result;
    for (auto& iter : mMapByName) {
        if (iter.second.type == type || type.empty()) {
            if (iter.second.flops > 0.0f && iter.second.costTime / mTotalTime >= rate) {
                result.emplace_back(std::make_pair(iter.second.name, iter.second.costTime / iter.second.flops));
            }
        }
    }
    if (result.size() < topK) {
        topK = result.size();
    }
    std::partial_sort(result.begin(), result.begin() + topK, result.end(), [&](const std::pair<std::string, float>& left, std::pair<std::string, float>& right) {
        return left.second > right.second;
    });
    for (int i=0; i<topK; ++i) {
        const auto& record = mMapByName[result[i].first];
        MNN_PRINT("%s -  %f GFlops, %.2f rate\n", record.name.c_str(), record.flops / record.costTime, record.costTime / mTotalTime * 100.0f);
    }
    MNN_PRINT("\n");
}


} // namespace MNN
