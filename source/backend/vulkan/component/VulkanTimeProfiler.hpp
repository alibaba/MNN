//
//  VulkanTimeProfiler.hpp
//  MNN
//
//  Created by MNN on 2026/01/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef ENABLE_VULKAN_TIME_PROFILE

#ifndef VulkanTimeProfiler_hpp
#define VulkanTimeProfiler_hpp

#include <stdint.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <MNN/MNNDefine.h>
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include "core/NonCopyable.hpp"
#include "VulkanDevice.hpp"

namespace MNN {

class VulkanTimeProfiler : public NonCopyable {
public:
    enum class Kind : uint8_t {
        Execution = 0,
        Shader    = 1
    };

    struct Token {
        uint32_t begin = 0;
        uint32_t end   = 0;
        uint32_t index = 0;
        bool valid     = false;
    };

    explicit VulkanTimeProfiler(const VulkanDevice& device);
    ~VulkanTimeProfiler();

    void reset();

    Token begin(VkCommandBuffer cmd, const char* name, Kind kind);
    void end(VkCommandBuffer cmd, const Token& token);

    void printTimeProfile() const;

private:
    struct Record {
        std::string name;
        Kind kind = Kind::Execution;
        uint32_t begin = 0;
        uint32_t end = 0;
    };

    void _printKind(Kind kind, const std::vector<uint64_t>& timestamps, double tickToMs) const;

private:
    const VulkanDevice& mDevice;
    VkQueryPool mQueryPool = VK_NULL_HANDLE;
    uint32_t mCapacity = 0;
    uint32_t mNext = 0;
    uint32_t mDroppedScopes = 0;

    std::vector<Record> mRecords;
};

class VulkanTimeProfileScope : public NonCopyable {
public:
    VulkanTimeProfileScope(VulkanTimeProfiler* profiler, VkCommandBuffer cmd, const char* name, VulkanTimeProfiler::Kind kind);
    ~VulkanTimeProfileScope();

private:
    VulkanTimeProfiler* mProfiler = nullptr;
    VkCommandBuffer mCmd = VK_NULL_HANDLE;
    VulkanTimeProfiler::Token mToken;
};

inline VulkanTimeProfiler::VulkanTimeProfiler(const VulkanDevice& device) : mDevice(device) {
    uint32_t tryCapacity = 4096;
    while (tryCapacity >= 2) {
        VkQueryPoolCreateInfo queryPoolCreateInfo = {};
        queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolCreateInfo.queryCount = tryCapacity;

        auto res = vkCreateQueryPool(mDevice.get(), &queryPoolCreateInfo, nullptr, &mQueryPool);
        if (VK_SUCCESS == res) {
            mCapacity = tryCapacity;
            return;
        }
        mQueryPool = VK_NULL_HANDLE;
        tryCapacity /= 2;
    }

    MNN_ERROR("Create VkQueryPool(VK_QUERY_TYPE_TIMESTAMP) failed, time profile disabled.\n");
}

inline VulkanTimeProfiler::~VulkanTimeProfiler() {
    if (mQueryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(mDevice.get(), mQueryPool, nullptr);
        mQueryPool = VK_NULL_HANDLE;
    }
}

inline void VulkanTimeProfiler::reset() {
    mNext = 0;
    mDroppedScopes = 0;
    mRecords.clear();
}

inline VulkanTimeProfiler::Token VulkanTimeProfiler::begin(VkCommandBuffer cmd, const char* name, Kind kind) {
    Token token;
    if (cmd == VK_NULL_HANDLE || mQueryPool == VK_NULL_HANDLE) {
        return token;
    }
    if (mNext + 2 > mCapacity) {
        mDroppedScopes++;
        return token;
    }

    token.begin = mNext;
    token.end = mNext + 1;
    token.index = (uint32_t)mRecords.size();
    token.valid = true;
    mNext += 2;

    Record record;
    if (nullptr != name) {
        record.name = name;
    }
    record.kind = kind;
    record.begin = token.begin;
    record.end = token.end;
    mRecords.emplace_back(std::move(record));

    vkCmdResetQueryPool(cmd, mQueryPool, token.begin, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, mQueryPool, token.begin);
    return token;
}

inline void VulkanTimeProfiler::end(VkCommandBuffer cmd, const Token& token) {
    if (!token.valid || cmd == VK_NULL_HANDLE || mQueryPool == VK_NULL_HANDLE) {
        return;
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, mQueryPool, token.end);
}

inline void VulkanTimeProfiler::_printKind(Kind kind, const std::vector<uint64_t>& timestamps, double tickToMs) const {
    float timeTotal = 0.0f;
    std::unordered_map<std::string, float> timeTable;
    for (const auto& record : mRecords) {
        if (record.kind != kind) {
            continue;
        }
        if (record.end >= timestamps.size() || record.begin >= timestamps.size()) {
            continue;
        }
        float timeCurr = (float)((timestamps[record.end] - timestamps[record.begin]) * tickToMs);
        timeTable[record.name] += timeCurr;
        timeTotal += timeCurr;
        MNN_PRINT("%-30s time is %4.2f ms.\n", record.name.c_str(), timeCurr);
    }

    std::vector<std::pair<std::string, float>> timeVectorForSort(timeTable.begin(), timeTable.end());
    std::sort(timeVectorForSort.begin(), timeVectorForSort.end(),
              [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) { return a.second > b.second; });

    MNN_PRINT("\nSummary:\n");
    for (const auto& it : timeVectorForSort) {
        MNN_PRINT("%-30s time is %4.2f ms.\n", it.first.c_str(), it.second);
    }
    MNN_PRINT("\nTotal time summed up is %6.2f ms\n", timeTotal);
}

inline void VulkanTimeProfiler::printTimeProfile() const {
    if (mNext == 0 || mQueryPool == VK_NULL_HANDLE) {
        return;
    }

    std::vector<uint64_t> timestamps(mNext);
    auto res = vkGetQueryPoolResults(mDevice.get(), mQueryPool, 0, mNext, sizeof(uint64_t) * timestamps.size(), timestamps.data(),
                                    sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (VK_SUCCESS != res) {
        MNN_ERROR("vkGetQueryPoolResults failed: %d\n", res);
        return;
    }

    double timestampPeriod = mDevice.getTimestampPeriod(); // ns per tick
    double tickToMs = timestampPeriod / double(1e6);

    if (mDroppedScopes > 0) {
        MNN_PRINT("Vulkan Time Profiling dropped %u scopes (capacity=%u)\n", mDroppedScopes, mCapacity);
    }

    MNN_PRINT("\n[Execution Profiling (start)]\n");
    _printKind(Kind::Execution, timestamps, tickToMs);
    MNN_PRINT("\n[Execution Profiling (end)]\n");

    bool hasShader = false;
    for (const auto& record : mRecords) {
        if (record.kind == Kind::Shader) {
            hasShader = true;
            break;
        }
    }
    if (hasShader) {
        MNN_PRINT("\n[Shader Profiling (start)]\n");
        _printKind(Kind::Shader, timestamps, tickToMs);
        MNN_PRINT("\n[Shader Profiling (end)]\n");
    }
}

inline VulkanTimeProfileScope::VulkanTimeProfileScope(VulkanTimeProfiler* profiler, VkCommandBuffer cmd, const char* name,
                                                      VulkanTimeProfiler::Kind kind) {
    mProfiler = profiler;
    mCmd = cmd;
    if (nullptr != mProfiler) {
        mToken = mProfiler->begin(cmd, name, kind);
    }
}

inline VulkanTimeProfileScope::~VulkanTimeProfileScope() {
    if (nullptr != mProfiler) {
        mProfiler->end(mCmd, mToken);
    }
}

} // namespace MNN

#endif /* VulkanTimeProfiler_hpp */

#endif
