/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SHARED_CUBIN_LOADER_H
#define SHARED_CUBIN_LOADER_H
#include "../../../common/cudaDriverWrapper.h"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "commonDatatype.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace MNN
{
namespace CUDA
{
template <typename TKernelMeta, typename TKernelParam>
class TSharedCubinKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;

    virtual uint64_t hashID(KernelMeta const& kernelMeta) const = 0;
    virtual uint64_t hashID(TKernelParam const& param) const = 0;

    TSharedCubinKernel(TKernelMeta const* pMetaStart, int32_t nMetaCount, MHFADataType type, int32_t sm)
        : mDataType(type)
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(sm)
    {
        MNN_ASSERT(mKernelMetaCount && "No kernels were loaded correctly.");
    }

    void loadCubinKernels(int32_t smVersion)
    {
        for (int32_t i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            auto const kernelKey = hashID(kernelMeta);
            if (kernelMeta.mSM == smVersion && kernelMeta.mDataType == mDataType
                && mFunctions.find(kernelKey) == mFunctions.end())
            {
                int32_t const DEFAULT_SMEM_SIZE{48 * 1024};
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE)
                {
                    int32_t deviceID{0};
                    cudaGetDevice(&deviceID);
                    int32_t sharedMemPerMultiprocessor{0};
                    if (cudaDeviceGetAttribute(
                            &sharedMemPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID)
                            != cudaSuccess
                        || sharedMemPerMultiprocessor < kernelMeta.mSharedMemBytes)
                    {
                        // skip load function because not enough shared memory to launch the kernel
                        continue;
                    }
                }

                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE)
                {
                    if (mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes)
                        != CUDA_SUCCESS)
                    {
                        // some chip may not have enough shared memory to launch the kernel
                        continue;
                    }
                }
                mFunctions.insert({kernelKey, funcInfo});
                //MNN_PRINT("func id:%d\n", kernelKey);
            }
        }
    }

    void loadCubinKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        loadCubinKernels(mSM);
    }

    bool isValid(int32_t s) const
    {
        return !mFunctions.empty();
    }

    virtual void run(TKernelParam& params, cudaStream_t ss) const
    {
        if (params.interleaved)
        {
            MNN_ASSERT(mDataType == DATA_TYPE_INT8);
        }

        auto const findIter = mFunctions.find(hashID(params));
        std::ostringstream errMsg;
        errMsg << "Could not find kernel for:\n"
               << "\t s: " << params.s << "\n"
               << "\t d: " << params.d << "\n"
               << "\t interleaved: " << params.interleaved << "\n"
               << "\t force_unroll: " << params.force_unroll << "\n"
               << "Was the plugin compiled on a compatible CUDA and SM version?\n"
               << "\t Compiled on CUDA " << CUDA_VERSION << "\n"
               << "\t Current SM version: " << mSM << "\n"
               << "\t SM versions enabled during compilation: "
#if defined(MNN_CUDA_ENABLE_SM75)
               << "75 "
#endif
#if defined(MNN_CUDA_ENABLE_SM80)
               << "80 "
#endif
#if defined(MNN_CUDA_ENABLE_SM86)
               << "86 "
#endif
#if defined(MNN_CUDA_ENABLE_SM89)
               << "89 "
#endif
               << "\n";
        if(findIter == mFunctions.end()) {
            MNN_ERROR("%s\n", errMsg.str().c_str());
        }

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        CUfunction const func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        if (!params.force_unroll)
        {
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
        else
        {
            checkKernelErrors;
            int32_t unroll = (params.s + kernelMeta.mUnrollStep - 1) / kernelMeta.mUnrollStep;
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
            checkKernelErrors;
        }
    }

    virtual ~TSharedCubinKernel() = default;

protected:
    MNN::CUDADriverWrapper mDriver;

    MHFADataType mDataType;
    TKernelMeta const* mKernelMeta;
    int32_t mKernelMetaCount;
    int32_t mSM;
    std::unordered_map<unsigned char const*, CUmodule> mModules;
    struct FusedMultiHeadAttentionKernelInfo
    {
        int32_t mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
};

template <typename TKernelList>
class TSharedCubinKernelFactory
{
public:
    TKernelList const* getCubinKernels(
        typename TKernelList::KernelMeta const* pKernelList, int32_t nbKernels, MHFADataType type, int32_t sm)
    {
        static std::mutex sMutex;
        std::lock_guard<std::mutex> lg(sMutex);

        auto const id = hashID(type, sm);
        //MNN_PRINT("kernel id: %d %d %d\n", type, sm, id);
        auto const findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            auto* newKernel = new TKernelList{pKernelList, nbKernels, type, sm};
            newKernel->loadCubinKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<TKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TSharedCubinKernelFactory<TKernelList>& Get()
    {
        static TSharedCubinKernelFactory<TKernelList> gFactory;
        return gFactory;
    }

private:
    TSharedCubinKernelFactory() = default;

    inline uint64_t hashID(MHFADataType type, int32_t sm) const
    {
        // use deviceID in hasID for multi GPU support before driver support context-less loading of cubin
        int32_t deviceID{0};
        cuda_check(cudaGetDevice(&deviceID));

        MNN_ASSERT((deviceID & 0xFFFF) == deviceID);
        MNN_ASSERT((type & 0xFFFF) == type);
        return (uint64_t) type << 48 | (uint64_t) deviceID << 32 | sm;
    }

    std::unordered_map<uint64_t, std::unique_ptr<TKernelList> const> mKernels;
};

} // namespace MNN
} // namespace CUDA
#endif // SHARED_CUBIN_LOADER_H
