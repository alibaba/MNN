//
//  ConvolutionIntFactory.cpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionIntFactory.hpp"
#include <math.h>
#include "Convolution3x3Int8.hpp"
#include "ConvolutionGroup.hpp"
#include "ConvolutionInt8Executor.hpp"

namespace MNN {
static inline void *MNNMemoryAllocAlignZeroAlign(size_t size) {
    return MNNMemoryCallocAlign(size, MNN_MEMORY_ALIGN_DEFAULT);
}
static int ReadBlobDim(unsigned char *&myfile, int *shape, int shapeBufCnt) {
    int uSize = myfile[0];
    myfile++;
    if (uSize > 4) {
        printf("Read shape error!\n");
        return 0;
    }
    int dimCnt = 0;
    for (unsigned char i = 0; i < uSize && dimCnt < shapeBufCnt; i++) {
        auto shortData  = (unsigned short *)myfile;
        shape[dimCnt++] = *shortData;

        myfile += 2;
    }
    return dimCnt;
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

void DestorySimpleSet(PSIMPLE_SET set) {
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

static int8_t *ReadQuanData_c(unsigned char *&s, uint32_t *len) {
    int8_t *blob      = nullptr;
    int8_t *samples   = nullptr;
    uint8_t *idxBuf   = nullptr;
    uint8_t *idxBytes = nullptr;
    uint32_t dataCnt  = 1;

    do {
        // blob shape
        int32_t shape[64] = {0};
        uint32_t shapeDim = (uint32_t)ReadBlobDim(s, shape, 64);
        if (shapeDim == 0 || shapeDim > 64)
            break;
        for (uint32_t i = 0; i < shapeDim; i++)
            dataCnt *= shape[i];

        // sample
        uint32_t sampleCnt = 0;
        StreamSizeRead(&sampleCnt, 1, 1, s);
        if (0 == sampleCnt) {
            sampleCnt = 256;
        }
        samples = (int8_t *)MNNMemoryAllocAlignZeroAlign(sampleCnt);
        if (samples == nullptr)
            break;
        StreamSizeRead(samples, 1, sampleCnt, s);
        SimpleRank(samples, sampleCnt, 1);
        // index
        uint32_t idxBitsCnt = atLestBitsCnt(sampleCnt);
        size_t idxBufSize   = ceil(idxBitsCnt * dataCnt * 0.125);
        idxBuf              = (uint8_t *)MNNMemoryAllocAlignZeroAlign(idxBufSize);
        if (nullptr == idxBuf) {
            MNN_ERROR("Not enought memory\n");
            break;
        }
        StreamSizeRead(idxBuf, 1, idxBufSize, s);
        // split index value into bytes
        idxBytes = (uint8_t *)MNNMemoryAllocAlignZeroAlign(dataCnt * sizeof(uint8_t));
        if (idxBitsCnt == 0 || nullptr == idxBytes) {
            break;
        }
        SplitBufToArray(idxBuf, (uint32_t)idxBufSize, idxBytes, (uint32_t)dataCnt, (uint32_t)idxBitsCnt);
        int i = 0;
        blob  = (int8_t *)MNNMemoryAllocAlignZeroAlign((size_t)dataCnt);
        if (nullptr == blob) {
            break;
        }
        for (i = 0; i < dataCnt; i++) {
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
    } while (0);

    if (samples != nullptr)
        MNNMemoryFreeAlign(samples);
    if (idxBuf != nullptr)
        MNNMemoryFreeAlign(idxBuf);
    if (idxBytes != nullptr)
        MNNMemoryFreeAlign(idxBytes);
    if (len)
        *len = blob ? dataCnt : 0;
    return blob;
}

static int8_t *ReadSparseQuanData_c(unsigned char *&myfile, uint32_t *len) {
    // MNN_ERROR("sparse:%d\n", 1);
    int shape[64] = {0};
    unsigned char ucMapSize;
    PSIMPLE_SET setWeight = CreateSimpleSet(256);
    if (setWeight == nullptr) {
        return nullptr;
    }
    std::shared_ptr<unsigned int> __autoReleaseSetWeight(nullptr, [setWeight](void *) { DestorySimpleSet(setWeight); });
    unsigned int nnz;
    unsigned char iIdxNeedBits;
    int8_t *blob = nullptr;
    // 1. weights blob shape(unsigned int32)
    int ShapeDim = ReadBlobDim(myfile, shape, 64);
    int Size     = sizeof(int8_t);
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
    // 6. valueset(signed char * valueset_size)
    for (unsigned char i = 0; i < ucMapSize; i++) {
        int8_t tmp;
        StreamSizeRead(&tmp, 1, 1, myfile);
        InsertSimpleSet(setWeight, tmp);
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
    {
        int iDataNeedBits = (int)ceil(_log2(ucMapSize));
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
        memset(blob, 0, Size * sizeof(signed char));
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

std::shared_ptr<ConvolutionIntFactory::Int8Common> ConvolutionIntFactory::load(const IDSTQuan *quan, bool forceFloat) {
    auto result           = std::make_shared<Int8Common>();
    uint32_t weightLength = 0;
    int8_t *buffer        = nullptr;
    auto originBuffer     = (unsigned char *)quan->buffer()->data();
    if (1 == quan->type()) {
        buffer = ReadQuanData_c(originBuffer, &weightLength);
    }
    if (2 == quan->type()) {
        buffer = ReadSparseQuanData_c(originBuffer, &weightLength);
    }
    if (nullptr == buffer) {
        MNN_PRINT("Alloc memory error for extract idst int8\n");
        return nullptr;
    }
    result->weight.set(buffer, weightLength);
    result->quan = quan;
    result->alpha.reset(quan->alpha()->size());
    if (nullptr == result->alpha.get()) {
        MNN_PRINT("Alloc memory error for extract idst int8\n");
        return nullptr;
    }
    ::memcpy(result->alpha.get(), quan->alpha()->data(), quan->alpha()->size() * sizeof(float));

    if (!quan->has_scaleInt() || forceFloat) {
        // Back to float
        result->weightFloat.reset(weightLength);
        if (nullptr == result->weightFloat.get()) {
            MNN_PRINT("Alloc memory error for extract idst int8/ Back to float\n");
            return nullptr;
        }
        auto outputCount   = result->alpha.size();
        int partWeightSize = weightLength / outputCount;
        for (int o = 0; o < outputCount; ++o) {
            auto dstW   = result->weightFloat.get() + o * partWeightSize;
            auto srcW   = result->weight.get() + o * partWeightSize;
            float alpha = result->alpha.get()[o];
            for (int j = 0; j < partWeightSize; ++j) {
                dstW[j] = ((float)srcW[j]) * alpha * quan->quantScale();
            }
        }

        result->weight.release();
        result->alpha.release();
    }

    return result;
}

Execution *ConvolutionIntFactory::createUnit(const Tensor *input, const Tensor *output, const MNN::Op *op,
                                             Backend *backend, const Int8Common *common, const float *bias,
                                             size_t biasSize) {
    auto conv2d     = op->main_as_Convolution2D();
    auto convCommon = conv2d->common();
    if (convCommon->kernelX() == 3 && convCommon->kernelY() == 3 && convCommon->strideX() == 1 &&
        convCommon->strideY() == 1 && convCommon->dilateX() == 1 && convCommon->dilateY() == 1 &&
        output->width() >= 8 && output->height() >= 8) {
        return new Convolution3x3Int8(conv2d->common(), backend, common, bias, biasSize);
    }
    return new ConvolutionInt8Executor(conv2d->common(), backend, common, bias, biasSize);
}

Execution *ConvolutionIntFactory::create(const Tensor *input, const Tensor *output, const MNN::Op *op, Backend *backend,
                                         const Int8Common *common) {
    auto conv2d = op->main_as_Convolution2D();
    auto group  = conv2d->common()->group();
    if (1 == group) {
        return createUnit(input, output, op, backend, common, conv2d->bias()->data(), conv2d->bias()->size());
    }
    MNN_ASSERT(common->weight.get() != nullptr);

    // Split
    std::vector<std::shared_ptr<Execution>> subConvolution;
    auto groupOutputCount = conv2d->common()->outputCount() / group;
    auto groupWeightSize  = common->weight.size() / group;
    for (int i = 0; i < group; ++i) {
        auto subCommon = std::make_shared<Int8Common>();
        subCommon->alpha.reset(groupOutputCount);
        ::memcpy(subCommon->alpha.get(), common->alpha.get() + groupOutputCount * i, groupOutputCount * sizeof(float));
        subCommon->quan = common->quan;
        subCommon->weight.reset(groupWeightSize);
        ::memcpy(subCommon->weight.get(), common->weight.get() + groupWeightSize * i, groupWeightSize * sizeof(int8_t));
        subConvolution.push_back(
            std::shared_ptr<Execution>(createUnit(input, output, op, backend, subCommon.get(),
                                                  conv2d->bias()->data() + groupOutputCount * i, groupOutputCount)));
    }
    return new ConvolutionGroup(backend, subConvolution);
}

} // namespace MNN
