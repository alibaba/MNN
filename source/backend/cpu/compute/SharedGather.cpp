#include "SharedGather.hpp"
#include "CommonOptFunction.h"
#include "../CPUBackend.hpp"
#include "core/BufferAllocator.hpp"
#include "core/Macro.h"

namespace MNN {

SharedGather::SharedGather(Backend* backend, std::shared_ptr<CPUConvolution::ResourceInt8> res) : Execution(backend) {
    mResource = res;
}

SharedGather::~SharedGather() {
    // Do nothing.
}

ErrorCode SharedGather::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto bytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
    auto output = outputs[0];
    int ic = output->length(output->dimensions() - 1);
    if (bytes != 4) {
        mCacheBuffer = static_cast<CPUBackend*>(backend())->getBufferAllocator()->alloc(ic * sizeof(float));
        static_cast<CPUBackend*>(backend())->getBufferAllocator()->free(mCacheBuffer);
    }
    return NO_ERROR;
}

ErrorCode SharedGather::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int outside = input->elementSize();
    int ic = output->length(output->dimensions() - 1);
    MNN_ASSERT(ic % mResource->mBlockNum == 0);
    int block = ic / mResource->mBlockNum;
    MNN_ASSERT(4 == mResource->mWeightBits || 8 == mResource->mWeightBits);
    auto outputPtr = output->host<int8_t>();
    auto indice = input->host<int>();
    auto perHpQuantSize = mResource->mBlockNum * 2 * sizeof(float) * mResource->mHp;
    auto perHpWeightSize = UP_DIV(ic, mResource->mLp) * mResource->mHp * mResource->mLp * mResource->mWeightBits / 8;
    auto perBlockWeightSize = block * mResource->mHp * mResource->mWeightBits / 8;
    auto perBlockQuantSize = 2 * mResource->mHp * sizeof(float);
    auto func = static_cast<CPUBackend*>(backend())->functions();
    auto bytes = func->bytes;

    MNN_ASSERT(mResource->mLp % 2 == 0);
    int lpStep = mResource->mWeightBits == 4 ? mResource->mLp / 2 : mResource->mLp;
    int blockUnit = block / mResource->mLp;
    int permuteUnit = mResource->mLp * mResource->mHp;
    int halfPermuteStride = static_cast<int32_t>(permuteUnit / 2);
    if (8 == mResource->mWeightBits) {
        for (int z = 0; z < outside; ++z) {
            auto index = indice[z];
            int zO = index / mResource->mHp;
            int zI = index % mResource->mHp;
            auto srcZ = mResource->mWeightInt8->host<int8_t>() + zO * (perHpQuantSize + perHpWeightSize);
            auto dstZInt8 = outputPtr + z * ic * bytes;
            float* dstZ = reinterpret_cast<float*>(dstZInt8);
            if (bytes == 2) {
                dstZ = reinterpret_cast<float*>(mCacheBuffer.ptr());
            }
            for (int i = 0; i < mResource->mBlockNum; ++i) {
                auto quantPtr = reinterpret_cast<const float*>(srcZ + i * (perBlockQuantSize + perBlockWeightSize) +
                                                               perBlockWeightSize);
                float scale = quantPtr[zI];
                float bias = quantPtr[zI + mResource->mHp];
                auto dstB = dstZ + i * block;
                auto srcB = srcZ + i * (perBlockQuantSize + perBlockWeightSize) + zI * lpStep;
                for (int j = 0; j < blockUnit; ++j) {
                    for (int k = 0; k < lpStep; ++k) {
                        dstB[j * lpStep + k] = srcB[j * lpStep * mResource->mHp + k] * scale + bias;
                    }
                }
            }
            if (bytes == 2) {
                func->MNNFp32ToLowp(dstZ, reinterpret_cast<int16_t*>(dstZInt8), ic);
            }
        }
        return NO_ERROR;
    }
    if (mResource->mPackMode == 0) {
        for (int z = 0; z < outside; ++z) {
            auto index = indice[z];
            int zO = index / mResource->mHp;
            int zI = index % mResource->mHp;
            int zI0 = zI / (mResource->mHp / 2);
            int zI1 = zI % (mResource->mHp / 2);
            int step = (1 - zI0) * 4;
            auto srcZ = mResource->mWeightInt8->host<int8_t>() + zO * (perHpQuantSize + perHpWeightSize);
            auto dstZInt8 = outputPtr + z * ic * bytes;
            float* dstZ = reinterpret_cast<float*>(dstZInt8);
            if (bytes == 2) {
                dstZ = reinterpret_cast<float*>(mCacheBuffer.ptr());
            }
            for (int i = 0; i < mResource->mBlockNum; ++i) {
                auto quantPtr = reinterpret_cast<const float*>(srcZ + i * (perBlockQuantSize + perBlockWeightSize) +
                                                               perBlockWeightSize);
                float scale = quantPtr[zI];
                float bias = quantPtr[zI + mResource->mHp];
                auto dstB = dstZ + i * block;
                auto srcB = srcZ + i * (perBlockQuantSize + perBlockWeightSize) + zI1 * mResource->mLp;
                for (int j = 0; j < blockUnit; ++j) {
                    for (int k = 0; k < mResource->mLp; ++k) {
                        uint8_t w = *reinterpret_cast<uint8_t*>(srcB + j * halfPermuteStride + k);
                        auto w1 = (w >> step) % 16;
                        dstB[j * mResource->mLp + k] = w1 * scale + bias;
                    }
                }
            }
            if (bytes == 2) {
                func->MNNFp32ToLowp(dstZ, reinterpret_cast<int16_t*>(dstZInt8), ic);
            }
        }
        return NO_ERROR;
    }
    for (int z = 0; z < outside; ++z) {
        auto index = indice[z];
        int zO = index / mResource->mHp;
        int zI = index % mResource->mHp;
        auto srcZ = mResource->mWeightInt8->host<int8_t>() + zO * (perHpQuantSize + perHpWeightSize);
        auto dstZInt8 = outputPtr + z * ic * bytes;
        float* dstZ = reinterpret_cast<float*>(dstZInt8);
        if (bytes == 2) {
            dstZ = reinterpret_cast<float*>(mCacheBuffer.ptr());
        }
        for (int i = 0; i < mResource->mBlockNum; ++i) {
            auto quantPtr = reinterpret_cast<const float*>(srcZ + i * (perBlockQuantSize + perBlockWeightSize) +
                                                           perBlockWeightSize);
            float scale = quantPtr[zI];
            float bias = quantPtr[zI + mResource->mHp];
            auto dstB = dstZ + i * block;
            auto srcB = srcZ + i * (perBlockQuantSize + perBlockWeightSize) + zI * lpStep;
            for (int j = 0; j < blockUnit; ++j) {
                for (int k = 0; k < lpStep; ++k) {
                    uint8_t w = *reinterpret_cast<uint8_t*>(srcB + j * lpStep * mResource->mHp + k);
                    auto w0 = w % 16;
                    auto w1 = w / 16;
                    dstB[2 * (j * lpStep + k) + 0] = w0 * scale + bias;
                    dstB[2 * (j * lpStep + k) + 1] = w1 * scale + bias;
                }
            }
        }
        if (bytes == 2) {
            func->MNNFp32ToLowp(dstZ, reinterpret_cast<int16_t*>(dstZInt8), ic);
        }
    }
    return NO_ERROR;
}

bool SharedGather::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new SharedGather(bn, mResource);
    return true;
}

} // namespace MNN
