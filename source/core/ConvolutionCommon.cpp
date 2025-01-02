//
//  ConvolutionCommon.cpp
//  MNN
//
//  Created by MNN on 2020/03/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionCommon.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/CPUBackend.hpp"
#include "half.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {

namespace IDSTDecoder {

static inline void *MNNMemoryAllocAlignZeroAlign(size_t size) {
    return MNNMemoryCallocAlign(size, MNN_MEMORY_ALIGN_DEFAULT);
}

static int ReadBlobDim(BaseLoader* myfile, unsigned int* shape, int shapeBufCnt, bool useInt32) {
    uint8_t uSize = 0;
    myfile->read((char*)&uSize, 1);
    if (uSize > 4) {
        printf("Read shape error!\n");
        return 0;
    }
    int copyLength = uSize;
    if (copyLength > shapeBufCnt) {
        copyLength = shapeBufCnt;
    }
    if (useInt32) {
        myfile->read((char*)shape, sizeof(unsigned int) * copyLength);
    } else {
        uint16_t shape_i16[32] = {0};
        myfile->read((char*)shape_i16, sizeof(uint16_t) * copyLength);
        for (int i = 0; i < copyLength; ++i) {
            shape[i] = shape_i16[i];
        }
    }
    return copyLength;
}

static double _log2(double x) {
    return log(x) / log(2);
}

static uint32_t atLestBitsCnt(uint32_t n) {
    for (uint32_t i = 0; i < 32; i++) {
        int32_t t = n << i;
        if (t < 0)
            return 32 - i - (((t << 1) == 0) ? 1 : 0);
    }
    return 0;
}

static void SplitBufToArray(uint8_t *buf, size_t bufLen, uint8_t *arr, size_t arrLen, size_t iNeedBits) {
    unsigned char cMask = (1 << (iNeedBits)) - 1;
    unsigned char *tmp  = (unsigned char *)buf;
    int iOffset         = 0;
    for (unsigned int i = 0; i < arrLen; i++) {
        unsigned char idx = 0;
        long uShift       = 8 - iNeedBits - iOffset % 8;
        if (uShift < 0) {
            idx = (tmp[iOffset / 8] << (0 - uShift)) & cMask;
            idx |= (tmp[(iOffset / 8) + 1] >> (8 + uShift)) & cMask;
        } else {
            idx = (tmp[iOffset / 8] >> uShift) & cMask;
        }
        iOffset += iNeedBits;
        if (iOffset % 8 == 0) {
            tmp += iOffset / 8;
            iOffset = 0;
        }
        arr[i] = idx;
    }
}

// fixme!!! not efficiency
typedef struct _SIMPLE_SET {
    int8_t *UniSet;
    uint32_t UniSetSize;
    uint32_t CurUniCnt;
} SIMPLE_SET, *PSIMPLE_SET;

static PSIMPLE_SET CreateSimpleSet(uint32_t maxSize) {
    PSIMPLE_SET set = (PSIMPLE_SET)calloc(1, sizeof(SIMPLE_SET));
    if (set == nullptr)
        return nullptr;
    set->UniSet     = (int8_t *)calloc(maxSize, sizeof(int8_t));
    set->UniSetSize = maxSize;
    set->CurUniCnt  = 0;
    return set;
}

static void SimpleRank(int8_t *data, uint32_t cnt, int up) {
    if (up) {
        for (uint32_t i = 0; i < cnt; i++) {
            for (uint32_t j = i + 1; j < cnt; j++) {
                if (data[i] > data[j]) {
                    int8_t tmp = data[i];
                    data[i]    = data[j];
                    data[j]    = tmp;
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < cnt; i++) {
            for (uint32_t j = i + 1; j < cnt; j++) {
                if (data[i] < data[j]) {
                    int8_t tmp = data[i];
                    data[i]    = data[j];
                    data[j]    = tmp;
                }
            }
        }
    }
}

static void InsertSimpleSet(PSIMPLE_SET set, int8_t value) {
    if (set->CurUniCnt >= set->UniSetSize)
        return;
    for (uint32_t i = 0; i < set->CurUniCnt; i++) {
        if (set->UniSet[i] == value)
            return;
    }
    set->UniSet[set->CurUniCnt++] = value;
    //    SimpleRank(set->UniSet, set->CurUniCnt, 1);
}

static void DestorySimpleSet(PSIMPLE_SET set) {
    if (set->UniSet != nullptr)
        free(set->UniSet);
    free(set);
}

typedef struct _SIMPLE_MAP {
    int8_t *CharCharMap;
    uint32_t CharMapSize;
    uint32_t CurMapCnt;
} SIMPLE_MAP, *PSIMPLE_MAP;

static PSIMPLE_MAP CreateSimpleMap(uint32_t MaxCnt) {
    PSIMPLE_MAP map = (PSIMPLE_MAP)calloc(1, sizeof(SIMPLE_MAP));
    if (map == nullptr)
        return nullptr;
    map->CharMapSize = MaxCnt * sizeof(int8_t);
    map->CurMapCnt   = 0;
    map->CharCharMap = (int8_t *)calloc(1, MaxCnt * 2);
    return map;
}

static void DestroySimpleMap(PSIMPLE_MAP map) {
    if (map->CharCharMap)
        free(map->CharCharMap);
    free(map);
}

static void InsertMap(PSIMPLE_MAP map, int8_t k, int8_t v) {
    for (uint32_t i = 0; i < map->CurMapCnt; i++) {
        if (map->CharCharMap[i * 2] == k) {
            map->CharCharMap[i * 2 + 1] = v;
            return;
        }
    }
    if (map->CurMapCnt >= map->CharMapSize)
        return;
    map->CharCharMap[map->CurMapCnt * 2]     = k;
    map->CharCharMap[map->CurMapCnt * 2 + 1] = v;
    map->CurMapCnt++;
}

static int8_t FindInMap(PSIMPLE_MAP map, int8_t k, int *found) {
    for (uint32_t i = 0; i < map->CurMapCnt; i++) {
        if (map->CharCharMap[i * 2] == k) {
            if (found != nullptr)
                *found = 1;
            return map->CharCharMap[i * 2 + 1];
        }
    }
    if (found != nullptr)
        *found = 0;
    return 0;
}

static bool isLinearSample(const std::vector<int8_t>& sample, int bit) {
    const int offset = 1 << (bit - 1);
    const int size = 1 << bit;
    if (sample.size() != size) {
        return false;
    }
    for (int i = 0; i < sample.size(); i++) {
        if (static_cast<int>(sample[i]) != i - offset) {
            return false;
        }
    }
    return true;
}

static void ReadQuanInfo(BaseLoader* s, size_t* len, ConvolutionCommon::Int8Common* result, bool shapeInt32) {
    *len  = 1;
    // blob shape
    unsigned int shape[32] = {0};
    uint32_t shapeDim = (uint32_t)ReadBlobDim(s, shape, 32, shapeInt32);
    if (shapeDim == 0 || shapeDim > 32)
        return;
    for (uint32_t i = 0; i < shapeDim; i++)
        *len *= shape[i];

    // sample
    uint32_t sampleCnt = 0;
    s->read((char*)&sampleCnt, 1);
    if (sampleCnt == 0) {
        sampleCnt = 256;
    }
    result->weightMap.resize(sampleCnt);
    auto samples = result->weightMap.data();
    if (samples == nullptr)
        return;
    s->read((char*)samples, sampleCnt);
    SimpleRank(samples, sampleCnt, 1);
    uint32_t idxBitsCnt = atLestBitsCnt(sampleCnt);
    result->canUseInt4 = idxBitsCnt == 4;
}

static int8_t *ReadQuanData_c(BaseLoader* s, size_t* len, ConvolutionCommon::Int8Common* result, bool shapeInt32, bool forceQuant) {
    int8_t *blob      = nullptr;
    uint8_t *idxBuf   = nullptr;
    size_t dataCnt  = 1;

    do {
        // blob shape
        unsigned int shape[32] = {0};
        uint32_t shapeDim = (uint32_t)ReadBlobDim(s, shape, 32, shapeInt32);
        if (shapeDim == 0 || shapeDim > 32)
            break;
        for (uint32_t i = 0; i < shapeDim; i++)
            dataCnt *= shape[i];

        // sample
        uint32_t sampleCnt = 0;
        s->read((char*)&sampleCnt, 1);
        if (sampleCnt == 0) {
            sampleCnt = 256;
        }
        result->weightMap.resize(sampleCnt);
        auto samples = result->weightMap.data();
        if (samples == nullptr)
            break;
        s->read((char*)samples, sampleCnt);
        SimpleRank(samples, sampleCnt, 1);
        uint32_t idxBitsCnt = atLestBitsCnt(sampleCnt);
        idxBitsCnt = idxBitsCnt < 1 ? 1 : idxBitsCnt;
        // index
        size_t idxBufSize   = ceil(idxBitsCnt * dataCnt * 0.125);
        idxBuf              = (uint8_t *)MNNMemoryAllocAlignZeroAlign(idxBufSize);
        if (nullptr == idxBuf) {
            MNN_ERROR("Not enought memory\n");
            break;
        }
        s->read((char*)idxBuf, idxBufSize);
        bool linear = isLinearSample(result->weightMap, idxBitsCnt);
        if (linear) {
            result->originBits = idxBitsCnt;
        }
        if (linear && (idxBitsCnt == 4 || idxBitsCnt == 8)) {
            if (!forceQuant && idxBitsCnt == 4) {
                // back to float, 4bit to 8bit
                *len = dataCnt;
                blob  = (int8_t *)MNNMemoryAllocAlignZeroAlign((size_t)UP_DIV(dataCnt, 2) * 2);
                for (int i = 0; i < idxBufSize; i++) {
                    int val = idxBuf[i];
                    int x1 = val / 16;
                    int x2 = val % 16;
                    blob[2 * i] = x1 - 8;
                    blob[2 * i + 1] = x2 - 8;
                }
            } else {
                // keep quant
                blob = (int8_t*)idxBuf;
                idxBuf = nullptr;
                if (idxBitsCnt == 4) {
                    result->canUseInt4 = true;
                } else {
                    for (int i = 0; i < idxBufSize; i++) {
                        blob[i] = (int)blob[i] - 128;
                    }
                }
                *len = idxBufSize;
            }
        } else {
            blob  = (int8_t *)MNNMemoryAllocAlignZeroAlign((size_t)UP_DIV(dataCnt, 2) * 2);
            if (nullptr == blob) {
                break;
            }
            bool success = true;
            int offset = (1 << (idxBitsCnt-1));
            do {
                if (linear) {
                    SplitBufToArray(idxBuf, (uint32_t)idxBufSize, (uint8_t*)blob, (uint32_t)dataCnt, (uint32_t)idxBitsCnt);
                    auto src = (uint8_t*)blob;
                    auto dst = blob;
                    for (int i=0; i<dataCnt; ++i) {
                        dst[i] = (int)src[i] - offset;
                    }
                    break;
                }
                // split index value into bytes
                uint8_t* idxBytes = (uint8_t *)MNNMemoryAllocAlignZeroAlign(dataCnt * sizeof(uint8_t));
                if (idxBitsCnt == 0 || nullptr == idxBytes) {
                    success = false;
                    break;
                }
                SplitBufToArray(idxBuf, (uint32_t)idxBufSize, idxBytes, (uint32_t)dataCnt, (uint32_t)idxBitsCnt);
                int i = 0;
                for (; i < dataCnt; i++) {
                    if (idxBytes[i] >= sampleCnt) {
                        MNN_PRINT("iNeedBits is %u\nRead quan weights error with idx:%d\n", idxBitsCnt, (int)idxBytes[i]);
                        success = false;
                        break;
                    }
                    blob[i] = samples[idxBytes[i]];
                }
                MNNMemoryFreeAlign(idxBytes);
            } while (false);

            if (!success) {
                MNNMemoryFreeAlign(blob);
                blob = nullptr;
                break;
            }
            if (len) {
                *len = blob ? dataCnt : 0;
            }
            if (result->originBits <= 4 && forceQuant) {
                // Reduce blob to 4 bit
                result->canUseInt4 = true;
                auto sizeDiv2 = UP_DIV(dataCnt, 2);
                auto newBlob  = (int8_t *)MNNMemoryAllocAlign((size_t)sizeDiv2, MNN_MEMORY_ALIGN_DEFAULT);
                for (int i=0; i<sizeDiv2; ++i) {
                    auto s0 = blob[2*i+0] + 8;
                    auto s1 = blob[2*i+1] + 8;
                    newBlob[i] = (s0 << 4) + s1;
                }
                MNNMemoryFreeAlign(blob);
                blob = newBlob;
            }
        }
    } while (0);

    if (idxBuf != nullptr)
        MNNMemoryFreeAlign(idxBuf);

    return blob;
}

static int8_t *ReadSparseQuanData_c(BaseLoader* myfile, size_t* len, const float* alpha_ptr, size_t alpha_size, ConvolutionCommon::Int8Common* result, bool useInt32) {    // MNN_ERROR("sparse:%d\n", 1);
    unsigned int shape[32];
    uint32_t ucMapSize = 0;
    PSIMPLE_SET setWeight = CreateSimpleSet(256);
    if (setWeight == nullptr) {
        return nullptr;
    }
    std::shared_ptr<unsigned int> __autoReleaseSetWeight(nullptr, [setWeight](void *) { DestorySimpleSet(setWeight); });
    unsigned int nnz;
    unsigned char iIdxNeedBits;
    int8_t *blob = nullptr;
    // 1. weights blob shape(unsigned int32)
    int ShapeDim = ReadBlobDim(myfile, shape, 32, useInt32);
    size_t Size     = sizeof(int8_t);
    for (int i = 0; i < ShapeDim; i++)
        Size *= shape[i];
    blob = (int8_t *)MNNMemoryAllocAlignZeroAlign((size_t)Size);
    if (blob == nullptr)
        return nullptr;
    // 2. nnz
    myfile->read((char *)&nnz, 4);
    // 3. max_step use # bits () (unsigned char)
    myfile->read((char *)&iIdxNeedBits, 1);
    // read idx array
    // 4. buf for steps ceil(nnz*step need bits/8)
    AutoStorage<unsigned char> arrIdxBuffer(nnz);
    unsigned char *arrIdx = arrIdxBuffer.get();
    if (nullptr == arrIdx) {
        return nullptr;
    }
    {
        size_t bufLen = (size_t)(ceil(0.125 * iIdxNeedBits * nnz));
        char *buf     = (char *)MNNMemoryAllocAlignZeroAlign(bufLen * sizeof(char));
        if (nullptr == buf) {
            return nullptr;
        }
        myfile->read((char *)buf, bufLen);
        SplitBufToArray((uint8_t *)buf, (uint32_t)bufLen, (uint8_t *)arrIdx, (uint32_t)nnz, (uint32_t)iIdxNeedBits);
        MNNMemoryFreeAlign(buf);
    }
    // 5. Avalable values Count(unsigned char)
    myfile->read((char *)&ucMapSize, 1);
    if (0 == ucMapSize) {
        ucMapSize = 256;
    }
    result->weightMap.resize(ucMapSize);
    // 6. valueset(signed char * valueset_size)
    for (int i = 0; i < ucMapSize; i++) {
        int8_t tmp;
        myfile->read((char *)&tmp, 1);
        InsertSimpleSet(setWeight, tmp);
        result->weightMap[i] = tmp;
    }
    SimpleRank(setWeight->UniSet, setWeight->CurUniCnt, 1);
    // map<unsigned char, signed char> mapWeight;
    PSIMPLE_MAP mapWeight = CreateSimpleMap(256);
    if (mapWeight == nullptr) {
        return nullptr;
    }
    std::shared_ptr<unsigned int> __autoReleaseMapWeight(nullptr, [mapWeight](void *) { DestroySimpleMap(mapWeight); });

    for (int i = 0; i < setWeight->CurUniCnt; i++) {
        InsertMap(mapWeight, i, setWeight->UniSet[i]);
    }
    //    unsigned char iIdx = 0;
    // 7. none zero weights indexes(nnz*ceil(log2(Avalable_values_Count))/8)
    AutoStorage<unsigned char> arrWeightIdxBuffer(nnz);
    unsigned char *arrWeightIdx = arrWeightIdxBuffer.get();
    if (nullptr == arrWeightIdx) {
        return nullptr;
    }
    int iDataNeedBits = (int)ceil(_log2(ucMapSize));
    iDataNeedBits = iDataNeedBits < 1 ? 1 : iDataNeedBits;
    {
        size_t bufLen     = (size_t)(ceil(0.125 * iDataNeedBits * nnz));
        char *buf         = (char *)MNNMemoryAllocAlignZeroAlign(bufLen * sizeof(char));
        if (nullptr == buf) {
            return nullptr;
        }
        myfile->read((char *)buf, bufLen);
        SplitBufToArray((uint8_t *)buf, (uint32_t)bufLen, (uint8_t *)arrWeightIdx, (uint32_t)nnz,
                        (uint32_t)iDataNeedBits);
        MNNMemoryFreeAlign(buf);
    }
    // set blob data with idx and weight idx
    {
        if (alpha_size == 2 * shape[0]) {
            const int min_value = -(1 << (iDataNeedBits - 1));
            auto alphaPtr = alpha_ptr;
            int area = Size / shape[0];
            for (int i = 0; i < shape[0]; i++) {
                float min = alphaPtr[2*i];
                float scale = alphaPtr[2*i+1];
                int zeroQuant = min_value;
                if (scale > 1e-6) {
                    zeroQuant = round((0.0f - min) / scale) + min_value;
                }
                memset(blob+area*i, zeroQuant, area * sizeof(signed char));
            }
        } else {
            memset(blob, 0, Size * sizeof(signed char)); //backward compability with previous symmetric weight quant
        }
        int iPreIdx = 0;
        for (int i = 0; i < nnz; i++) {
            iPreIdx += arrIdx[i];
            int found    = 0;
            int8_t value = FindInMap(mapWeight, arrWeightIdx[i], &found);
            if (!found) {
                MNN_ERROR("Read quan weights error with idx:%d\n", arrWeightIdx[i]);
                MNNMemoryFreeAlign(blob);
                return nullptr;
            }
            blob[iPreIdx] = value;
        }
    }
    *len = Size;
    return blob;
}


} // namespace IDSTDecoder

std::shared_ptr<ConvolutionCommon::Int8Common> ConvolutionCommon::load(const Op* op, Backend* backend, bool forceFloat, bool forceInt8) {
    auto conv = op->main_as_Convolution2D();
    auto quan = conv->quanParameter();
    std::shared_ptr<ConvolutionCommon::Int8Common> result(new Int8Common);
    result->quan = quan;
    size_t buffer_size = 0, alpha_size = 0;
    const int8_t* buffer_ptr = nullptr;
    const float* alpha_ptr = nullptr;
    std::unique_ptr<int8_t[]> external_buffer;
    size_t weightLength = 0;
    int8_t *buffer        = nullptr;
    bool useCachedMmap = false;
    if (backend && backend->getRuntime()) {
        useCachedMmap = backend->getRuntime()->hint().useCachedMmap > 1;
    }
    if (USE_EXTERNAL_DATA(conv) && op->externalPath() && quan->buffer() == nullptr) {
        auto external_info = conv->external()->data();
        buffer_size = external_info[1];
        alpha_size = external_info[2] / sizeof(float);
        std::unique_ptr<FileLoader> external_file(new FileLoader(op->externalPath()->c_str()));
        external_file->offset(external_info[0]);
        if (useCachedMmap) {
            if (alpha_size) {
                result->alpha.reset(alpha_size);
                IDSTDecoder::ReadQuanInfo(external_file.get(), &weightLength, result.get(), quan->shapeInt32());
            }
        } else {
            // external data
            if (0 != buffer_size) {
                if (1 == quan->type() && !forceFloat) {
                    buffer = IDSTDecoder::ReadQuanData_c(external_file.get(), &weightLength, result.get(), quan->shapeInt32(), forceInt8);
                } else {
                    external_buffer.reset(new int8_t[buffer_size]);
                    buffer_ptr = external_buffer.get();
                    external_file->read((char*)buffer_ptr, buffer_size);
                }
            }
            if (0 != alpha_size) {
                result->alpha.reset(alpha_size);
                if (nullptr == result->alpha.get()) {
                    MNN_PRINT("Alloc memory error for extract idst int8\n");
                    return nullptr;
                }
                alpha_ptr = result->alpha.get();
                external_file->read((char*)alpha_ptr, alpha_size * sizeof(float));
            }
        }
    } else {
        if (quan->buffer()) {
            buffer_size = quan->buffer()->size();
            buffer_ptr = quan->buffer()->data();
        }
        if (quan->alpha()) {
            alpha_size = quan->alpha()->size();
            alpha_ptr = quan->alpha()->data();
            result->alpha.reset(alpha_size);
            if (nullptr == result->alpha.get()) {
                MNN_PRINT("Alloc memory error for extract idst int8\n");
                return nullptr;
            }
            ::memcpy(result->alpha.get(), alpha_ptr, alpha_size * sizeof(float));
        }
    }
    if (quan->index() != nullptr) {
        if (forceFloat) {
            // Expand sparse to dense
            result->weightFloat.reset(quan->weightSize());
            if (nullptr == result->weightFloat.get()) {
                return nullptr;
            }
            ::memset(result->weightFloat.get(), 0, quan->weightSize() * sizeof(float));
            auto index = quan->index()->data();
            auto indexSize = quan->index()->size();
            if (nullptr == alpha_ptr || alpha_size != indexSize) {
                MNN_ERROR("The model is error, don't has alpha but has index\n");
                return nullptr;
            }
            for (uint32_t i=0; i<indexSize; ++i) {
                result->weightFloat.get()[index[i]] = alpha_ptr[i];
            }
        } // Otherwise needn't treat, just return result with quan info
        return result;
    }

    std::unique_ptr<MemoryLoader> originBuffer(new MemoryLoader((unsigned char*)buffer_ptr));
    if (1 == quan->type() && weightLength == 0) {
        buffer = IDSTDecoder::ReadQuanData_c(originBuffer.get(), &weightLength, result.get(), quan->shapeInt32(), forceInt8);
    }
    if (2 == quan->type()) {
        buffer = IDSTDecoder::ReadSparseQuanData_c(originBuffer.get(), &weightLength, alpha_ptr, alpha_size, result.get(), quan->shapeInt32());
    }
    // read fp16 data
    if (3 == quan->type()) {
        if (useCachedMmap) {
            weightLength = buffer_size / sizeof(half_float::half);
            result->weightFloat.reset(weightLength);
            return result;
        }
        weightLength = buffer_size / sizeof(half_float::half);
        std::vector<int8_t> tempHalfWeight(buffer_size);
        ::memcpy(tempHalfWeight.data(), buffer_ptr, buffer_size);
        auto halfWeight = reinterpret_cast<half_float::half *>(tempHalfWeight.data());
        result->weightFloat.reset(weightLength);
        if (nullptr == result->weightFloat.get()) {
            MNN_PRINT("Alloc memory error for extract fp16 back to float\n");
            return nullptr;
        }
        std::transform(halfWeight, halfWeight + weightLength, result->weightFloat.get(),
                       [](half_float::half h) { return float(h); });
        return result;
    }

    // weight int8 only
    if (4 == quan->type()) {
        weightLength = buffer_size;
        result->weight.reset(weightLength);
        ::memcpy(result->weight.get(), buffer_ptr, weightLength);
    }

    bool oldType4 = (quan->type() == 4 && quan->aMin() == 0 && std::abs(quan->quantScale()) < 1e-6);
    if (quan->readType() != 0 || oldType4) {
        result->asymmetric = true;
    } else {
        result->asymmetric = false;
    }
    if (!useCachedMmap) {
        if (result->weight.get() == nullptr) {
            if (nullptr == buffer) {
                MNN_PRINT("Alloc memory error for extract idst int8\n");
                return nullptr;
            }
            result->weight.set(buffer, weightLength);
        }
        int outputCount = 0;
        if (result->asymmetric) {
            outputCount   = result->alpha.size() / 2;
            // clampMin is minVal in asymmetric quant, clampMin = -(2^(bit))
            // and old version clampMin is -128
            float clampMin = quan->aMin() == 0 ? -128 : quan->aMin();
            if (clampMin < 0) {
                for (int o = 0; o < outputCount; ++o) {
                    result->alpha.get()[2 * o] = result->alpha.get()[2 * o] - clampMin * result->alpha.get()[2 * o + 1];
                }
            }
        } else {
            outputCount   = result->alpha.size(); // backward compability with previous symmetric quantization
        }
        if (!quan->has_scaleInt()) {
            float extraFactor = quan->quantScale();
            // for old type 4 models, their quan->quantScale is 0. which will introduce a bug here
            if (oldType4) {
                extraFactor = 1.0f;
            } else if (extraFactor != 1.0f) {
                for (int o=0; o<result->alpha.size(); ++o) {
                    result->alpha.get()[o] *= extraFactor;
                }
            }
        }
    }
    if (forceInt8) {
        return result;
    }
    if (!quan->has_scaleInt() || forceFloat) {
        // Back to float
        result->weightFloat.reset(weightLength);
        if (nullptr == result->weightFloat.get()) {
            MNN_PRINT("Alloc memory error for extract idst int8/ Back to float\n");
            return nullptr;
        }
        int outputCount = 0;
        if (result->asymmetric) {
            outputCount = result->alpha.size() / 2;
        } else {
            outputCount = result->alpha.size();
        }
        int partWeightSize = weightLength / outputCount;
        for (int o = 0; o < outputCount; ++o) {
            float min = 0.0f;
            float alpha = 0.0f;
            if (result->asymmetric) {
                min = result->alpha.get()[2*o];
                alpha = result->alpha.get()[2*o+1];
            } else {
                alpha = result->alpha.get()[o];
            }
            auto dstW   = result->weightFloat.get() + o * partWeightSize;
            auto srcW   = result->weight.get() + o * partWeightSize;
            for (int v=0; v < partWeightSize; ++v) {
                dstW[v] = (float)srcW[v] * alpha + min;
            }
        }
        result->weight.release();
        result->alpha.release();
    }
    return result;
}

void ConvolutionCommon::getConvParameters(std::shared_ptr<Int8Common> *quanCommon, Backend* backend, const MNN::Op *op, const float** originWeight, int* originWeightSize) {
    auto conv2d = op->main_as_Convolution2D();
    *originWeight = nullptr;
    *originWeightSize = 0;
    if (nullptr != conv2d->quanParameter()) {
        bool forceFloat = conv2d->quanParameter()->index() != nullptr;
        *quanCommon = load(op, backend, forceFloat);
        *originWeight     = (*quanCommon)->weightFloat.get();
        *originWeightSize = (*quanCommon)->weightFloat.size();
    }
    if (*originWeight == nullptr) {
        *originWeight = conv2d->weight()->data();
        *originWeightSize = conv2d->weight()->size();
    }
}

bool ConvolutionCommon::getConvInt8Parameters(const MNN::Op* op, std::shared_ptr<Int8Common>& quanCommon, Backend* backend,
                                              const int8_t*& weight, int& weightSize, float*& scale, int32_t*& bias, int32_t*& weightQuantZeroPoint) {
    // Compability for old quant model
    auto conv2d = op->main_as_Convolution2D();
    int outputCount = conv2d->common()->outputCount();
    weightSize = 0;
    // fix xcode UndefinedBehaviorSanitizer
    if (conv2d->symmetricQuan() && conv2d->symmetricQuan()->weight() != nullptr) {
        weight = conv2d->symmetricQuan()->weight()->data();
        weightSize = conv2d->symmetricQuan()->weight()->size();
    }
    if (conv2d->quanParameter() && (conv2d->quanParameter()->buffer() || conv2d->external())) { // int8 weight
        quanCommon = ConvolutionCommon::load(op, backend, false, true);
        MNN_ASSERT(quanCommon != nullptr);
        weight = quanCommon->weight.get();
        weightSize = quanCommon->weight.size();
    }
    if (weight == nullptr) {
        MNN_ERROR("ConvolutionCommon::getConvInt8Parameters: No weight data!");
        return false;
    }
    bool weightAsy = false;
    if (quanCommon && quanCommon->asymmetric) {
        weightAsy = true;
    }

    if (conv2d->symmetricQuan() && conv2d->symmetricQuan()->bias() && conv2d->symmetricQuan()->scale()) {
        // Compability for old model
        MNN_ASSERT(conv2d->symmetricQuan()->bias()->size() == outputCount && conv2d->symmetricQuan()->scale()->size() == outputCount);
        ::memcpy(bias, conv2d->symmetricQuan()->bias()->data(), outputCount * sizeof(int32_t));
        ::memcpy(scale, conv2d->symmetricQuan()->scale()->data(), outputCount * sizeof(float));
        return true;
    }
    if (conv2d->bias()) {
        ::memcpy(bias, conv2d->bias()->data(), outputCount * sizeof(float));
    }
    if (conv2d->quanParameter() && conv2d->quanParameter()->alpha()) {
        auto alphaAndBeta = conv2d->quanParameter()->alpha()->data();
        int quantCount    = conv2d->quanParameter()->alpha()->size();
        if (false == weightAsy) { // symmetric quant
            ::memcpy(scale, conv2d->quanParameter()->alpha()->data(), quantCount * sizeof(float));
        } else if (true == weightAsy) { // asymmetric
            // int ocx2 = 2 * outputCount;
            int scaleSize = quantCount / 2;
            float clampMin = conv2d->quanParameter()->aMin() == 0 ? -128 : conv2d->quanParameter()->aMin();
            for (int i = 0; i < scaleSize; ++i) {
                weightQuantZeroPoint[i] = static_cast<int32_t>(roundf((-1) * alphaAndBeta[2 * i] / alphaAndBeta[2 * i + 1])  + clampMin);
                scale[i] = alphaAndBeta[2 * i + 1];
            }
        }
        return true;
    }
    MNN_ERROR("ConvolutionCommon::getConvInt8Parameters: No bias & scale data!");
    return false;
}

std::pair<int, int> ConvolutionCommon::convolutionPad(const Tensor *input, const Tensor *output,
                                                      const Convolution2DCommon *mCommon) {
    if (mCommon->padMode() == PadMode_SAME) {
        int kernelWidthSize  = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

        int padNeededWidth  = (output->width() - 1) * mCommon->strideX() + kernelWidthSize - input->width();
        int padNeededHeight = (output->height() - 1) * mCommon->strideY() + kernelHeightSize - input->height();
        auto mPadX          = padNeededWidth / 2;
        auto mPadY          = padNeededHeight / 2;
        return std::make_pair(mPadX, mPadY);
    }
    auto mPadX = mCommon->padX();
    auto mPadY = mCommon->padY();
    if (nullptr != mCommon->pads() && mCommon->pads()->size() >= 2) {
        mPadX = mCommon->pads()->data()[1];
        mPadY = mCommon->pads()->data()[0];
    }
    return std::make_pair(mPadX, mPadY);
}

std::tuple<int, int, int, int> ConvolutionCommon::convolutionPadFull(const Tensor* input, const Tensor* output,
                                                         const Convolution2DCommon* common) {
    auto pad = convolutionPad(input, output, common);
    int iw = input->width();
    int ih = input->height();
    int ow = output->width();
    int oh = output->height();

    int right = (ow - 1) * common->strideX() + (common->kernelX() - 1) * common->dilateX() - pad.first;
    int padRight = 0;
    if (right >= iw) {
        padRight = right - iw + 1;
    }
    int bottom = (oh - 1) * common->strideY() + (common->kernelY() - 1) * common->dilateY() - pad.second;
    int padBottom = 0;
    if (bottom >= ih) {
        padBottom = bottom - ih + 1;
    }
    return std::make_tuple(pad.first, pad.second, padRight, padBottom);
}

std::pair<int, int> ConvolutionCommon::convolutionTransposePad(const Tensor *input, const Tensor *output,
                                                               const Convolution2DCommon *mCommon) {
    if (mCommon->padMode() == PadMode_SAME) {
        const int outputWidth  = output->width();
        const int outputHeight = output->height();

        const int outputWidthPadded  = (input->width() - 1) * mCommon->strideX() + mCommon->kernelX();
        const int outputHeightPadded = (input->height() - 1) * mCommon->strideY() + mCommon->kernelY();

        const int padNeededWidth  = outputWidthPadded - outputWidth;
        const int padNeededHeight = outputHeightPadded - outputHeight;

        auto mPadX = padNeededWidth / 2;
        auto mPadY = padNeededHeight / 2;
        return std::make_pair(mPadX, mPadY);
    }
    auto mPadX = mCommon->padX();
    auto mPadY = mCommon->padY();
    if (nullptr != mCommon->pads() && mCommon->pads()->size() >= 2) {
        mPadY = mCommon->pads()->data()[0];
        mPadX = mCommon->pads()->data()[1];
    }
    return std::make_pair(mPadX, mPadY);
}

} // namespace MNN
