//
//  IDSTDecoder.hpp
//  MNN
//
//  Created by MNN on 2024/03/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef IDSTDECODER_HPP
#define IDSTDECODER_HPP

#include <map>
#include <cmath>
#include "MNN_generated.h"
#include "core/ConvolutionCommon.hpp"

using namespace MNN;

namespace IDSTDecoder {

static inline void *MNNMemoryAllocAlignZeroAlign(size_t size) {
    return MNNMemoryCallocAlign(size, MNN_MEMORY_ALIGN_DEFAULT);
}

static int ReadBlobDim(unsigned char *&myfile, unsigned int* shape, int shapeBufCnt, bool useInt32) {
    int uSize = myfile[0];
    myfile++;
    if (uSize > 4) {
        printf("Read shape error!\n");
        return 0;
    }
    int copyLength = uSize;
    if (copyLength > shapeBufCnt) {
        copyLength = shapeBufCnt;
    }
    if (useInt32) {
        ::memcpy(shape, myfile, sizeof(unsigned int) * copyLength);
        myfile += copyLength * sizeof(unsigned int);
    } else {
        auto myfileint16 = (uint16_t*)myfile;
        for (int i=0; i<copyLength; ++i) {
            shape[i] = myfileint16[i];
        }
        myfile += copyLength * sizeof(unsigned short);
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

static void StreamSizeRead(void *dst, int unit, size_t count, unsigned char *&file) {
    ::memcpy(dst, file, unit * count);
    file += (unit * count);
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

static int8_t *ReadQuanData_c(unsigned char *&s, size_t* len, ConvolutionCommon::Int8Common* result, bool shapeInt32) {
    int8_t *blob      = nullptr;
    uint8_t *idxBuf   = nullptr;
    uint8_t *idxBytes = nullptr;
    uint32_t dataCnt  = 1;

    do {
        // blob shape
        unsigned int shape[32] = {0};
        uint32_t shapeDim        = (uint32_t)ReadBlobDim(s, shape, 32, shapeInt32);
        if (shapeDim == 0 || shapeDim > 32)
            break;
        for (uint32_t i = 0; i < shapeDim; i++)
            dataCnt *= shape[i];

        // sample
        uint32_t sampleCnt = 0;
        StreamSizeRead(&sampleCnt, 1, 1, s);
        if (sampleCnt == 0) {
            sampleCnt = 256;
        }
        result->weightMap.resize(sampleCnt);
        auto samples = result->weightMap.data();
        if (samples == nullptr)
            break;
        StreamSizeRead(samples, 1, sampleCnt, s);
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
        StreamSizeRead(idxBuf, 1, idxBufSize, s);
        if (idxBitsCnt == 4) {
            dataCnt = UP_DIV(dataCnt, 2) * 2;
        }
        blob  = (int8_t *)MNNMemoryAllocAlignZeroAlign((size_t)dataCnt);
        if (nullptr == blob) {
            break;
        }

        if (isLinearSample(result->weightMap, idxBitsCnt) && (idxBitsCnt == 4 || idxBitsCnt == 8)) {
            // fast sample for bit = 4 or 8
            if (idxBitsCnt == 4) {
                for (int i = 0; i < idxBufSize; i++) {
                    int val = idxBuf[i];
                    int x1 = val / 16;
                    int x2 = val % 16;
                    blob[2 * i] = x1 - 8;
                    blob[2 * i + 1] = x2 - 8;
                }
            }
            if (idxBitsCnt == 8) {
                for (int i = 0; i < idxBufSize; i++) {
                    int val = idxBuf[i];
                    blob[i] = val - 128;
                }
            }
        } else {
            // split index value into bytes
            idxBytes = (uint8_t *)MNNMemoryAllocAlignZeroAlign(dataCnt * sizeof(uint8_t));
            if (idxBitsCnt == 0 || nullptr == idxBytes) {
                break;
            }
            SplitBufToArray(idxBuf, (uint32_t)idxBufSize, idxBytes, (uint32_t)dataCnt, (uint32_t)idxBitsCnt);
            int i = 0;
            for (; i < dataCnt; i++) {
                if (idxBytes[i] >= sampleCnt) {
                    MNN_PRINT("iNeedBits is %u\nRead quan weights error with idx:%d\n", idxBitsCnt, (int)idxBytes[i]);
                    break;
                }
                blob[i] = samples[idxBytes[i]];
            }

            if (i < dataCnt) {
                MNNMemoryFreeAlign(blob);
                blob = nullptr;
                break;
            }
            MNNMemoryFreeAlign(idxBytes);
            idxBytes = nullptr;
        }
    } while (0);

    if (idxBuf != nullptr)
        MNNMemoryFreeAlign(idxBuf);
    if (idxBytes != nullptr)
        MNNMemoryFreeAlign(idxBytes);
    if (len)
        *len = blob ? dataCnt : 0;
    return blob;
}

static int8_t *ReadSparseQuanData_c(unsigned char *&myfile, size_t* len, const float* alpha_ptr, size_t alpha_size, ConvolutionCommon::Int8Common* result, bool useInt32) {    // MNN_ERROR("sparse:%d\n", 1);
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
    StreamSizeRead(&nnz, 4, 1, myfile);
    // 3. max_step use # bits () (unsigned char)
    StreamSizeRead(&iIdxNeedBits, 1, 1, myfile);
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
        StreamSizeRead(buf, 1, bufLen, myfile);
        SplitBufToArray((uint8_t *)buf, (uint32_t)bufLen, (uint8_t *)arrIdx, (uint32_t)nnz, (uint32_t)iIdxNeedBits);
        MNNMemoryFreeAlign(buf);
    }
    // 5. Avalable values Count(unsigned char)
    StreamSizeRead(&ucMapSize, 1, 1, myfile);
    if (0 == ucMapSize) {
        ucMapSize = 256;
    }
    result->weightMap.resize(ucMapSize);
    // 6. valueset(signed char * valueset_size)
    for (int i = 0; i < ucMapSize; i++) {
        int8_t tmp;
        StreamSizeRead(&tmp, 1, 1, myfile);
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
        StreamSizeRead(buf, 1, bufLen, myfile);
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

#endif // IDSTDECODER_HPP
