//
//  IDSTEncoder.hpp
//  MNN
//
//  Created by MNN on 2021/02/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef IDSTENCODER_HPP
#define IDSTENCODER_HPP

#include <map>
#include <sstream>
#include "MNN_generated.h"
#include <cmath>

using namespace MNN;

namespace IDSTEncoder {

static bool WriteBlobDim(std::ostream &out, std::vector<int> dims)
{
    char tmp[4];
    bool useInt32 = false;
    ((unsigned char *)tmp)[0] = (unsigned char)dims.size();
    out.write(tmp, 1);
    for (int i = 0; i < dims.size(); i++) {
        if (dims[i] > ((1<<16)-1)) {
            useInt32 = true;
            break;
        }
    }
    if (useInt32) {
        for (int i = 0; i < dims.size(); i++) {
            unsigned int tmpShort = (unsigned int)dims[i];
            out.write((const char*)(&tmpShort), 4);
        }
    } else {
        for (int i = 0; i < dims.size(); i++) {
            unsigned short tmpShort = (unsigned short)dims[i];
            out.write((const char*)(&tmpShort), 2);
        }
    }
    return useInt32;
}

static void FillBuffer(char *buf, unsigned int buf_len, const char *arr, unsigned int arr_len, unsigned char iNeedBits)
{
    memset(buf, 0, buf_len);
    char *tmp = buf;
    int iOffset = 0;
    unsigned char cMask = (1 << iNeedBits) - 1;
    for (int i = 0; i < arr_len; i++)
    {
        char value = arr[i];
        int uShift = 8 - iNeedBits - iOffset % 8;
        if (uShift < 0)
        {
            tmp[iOffset / 8] |= ((value & cMask) >> (0 - uShift));
            tmp[(iOffset / 8) + 1] |= ((value & cMask) << (8 + uShift));
        }
        else
        {
            tmp[iOffset / 8] |= ((value & cMask) << uShift);
        }
        iOffset += iNeedBits;
        if (iOffset % 8 == 0)
        {
            tmp += iOffset / 8;
            iOffset = 0;
        }
    }
}

static void GetWeightSet(std::set<int> &setWeight, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, const int bits)
{
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    setWeight.clear();
#define LINEAR_WEIGHT_SET
#ifdef LINEAR_WEIGHT_SET
    // using linear weight map
    for (int i = min_value; i <= max_value; i++) {
        setWeight.insert(i);
    }
    return;
#endif
    if (asymmetricQuantFlag) {
        for (int i = 0; i < channel; i++)
        {
            float min = alphaData[2*i];
            float alpha = alphaData[2*i+1];
            if (alpha <= 1e-6f)
            {
                setWeight.insert(min_value);
                continue;
            }
            for (int j = 0; j < area; j++)
            {
                float weight = weightData[i * area + j];
                setWeight.insert(fmax(fmin(round((weight - min) / alpha) + min_value, max_value), min_value));
            }
        }
    } else {
        for (int i = 0; i < channel; i++)
        {
            float alpha = alphaData[i];
            if (alpha <= 1e-6f)
            {
                setWeight.insert(0);
                continue;
            }
            for (int j = 0; j < area; j++)
            {
                float weight = weightData[i * area + j];
                setWeight.insert(fmax(fmin(round(weight / alpha), max_value), min_value));
            }
        }
    }
}

static float GetSparsity(const float* weightData, int weightSize, unsigned int& nnz, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, const int bits, int iMaxStep = -1)
{
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    nnz = 0;
    int iPreIdx = 0;
    float sparsity;
    if (asymmetricQuantFlag) {
        for (int i = 0; i < weightSize; i++)
        {
            float min = alphaData[2*(i/area)];
            float alpha = alphaData[2*(i/area)+1];
            int zeroQuant = min_value;
            if (alpha > 1e-6) {
                zeroQuant = round((0.0f - min) / alpha) + min_value;
            }

            float weight = weightData[i];
            int value = min_value;
            if (alpha > 1e-6)
            {
                value = round((weight - min) / alpha) + min_value;
            }

            if (value != zeroQuant)
            {
                nnz++;
                iPreIdx = i;
            }
            if ((i - iPreIdx >= iMaxStep) && (iMaxStep != -1))
            {
                nnz++;
                iPreIdx = i;
            }
        }
    } else {
        for (int i = 0; i < weightSize; i++)
        {
            float alpha = alphaData[i / area];
            float weight = weightData[i];
            int value = 0;
            if (alpha > 1e-6f)
            {
                value = round(weight / alpha);
            }

            if (value != 0)
            {
                nnz++;
                iPreIdx = i;
            }
            if ((i - iPreIdx >= iMaxStep) && (iMaxStep != -1))
            {
                nnz++;
                iPreIdx = i;
            }
        }
    }
    sparsity = 1 - 1.0f * nnz / weightSize;
    return sparsity;
}

static unsigned int GetBestMaxStep(const float* weightData, int weightSize, unsigned char& iMaxStepBits, int BlobDataSize, const float* alphaData, int area, int channel, bool asymmetricQuantFlag)
{
    size_t szBestSize = 1000000000;
    unsigned int best_nnz = 0;
    for (int i = 2; i < 9; i++)
    {
        unsigned int nnz = 0;
        GetSparsity(weightData, weightSize, nnz, alphaData, area, channel, asymmetricQuantFlag, BlobDataSize, pow(2, i) - 1);
        size_t tmp = ceil(0.125 * nnz * i) + ceil(0.125 * nnz * BlobDataSize);
        if (tmp < szBestSize)
        {
            iMaxStepBits = (unsigned char) i;
            szBestSize = tmp;
            best_nnz = nnz;
        }
    }
    return best_nnz;
}

static void WriteCQBlobs(std::ostream &out, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, bool& shapeUseInt32, const int bits)
{
    //push values into buffer
    //Find int values in all blobs and check;
    std::set<int> setWeight;
    GetWeightSet(setWeight, weightData, alphaData, area, channel, asymmetricQuantFlag, bits);
    int iCount = setWeight.size();
    int iNeedBits = ceil(log2(iCount));
    iNeedBits = iNeedBits < 1 ? 1 : iNeedBits;
    if (iNeedBits > 8) {
        MNN_ERROR("The Bits need large than 8, the model may be error for user\n");
        return;
    }
    std::map<int, unsigned char> mapWeight;
    int iIdx = 0;
    for (std::set<int>::iterator it = setWeight.begin(); it != setWeight.end(); it++)
    {
        mapWeight[*it] = iIdx++;
    }
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    size_t buf_len = size_t(ceil(0.125 * iNeedBits * area * channel));
    char *buf = new char[buf_len];
    {
        char *arr = new char[area * channel];
        unsigned char *tmp = (unsigned char*)arr;
        if (asymmetricQuantFlag) {
            for (int i = 0; i < channel; i++)
            {
                float min = alphaData[2*i];
                float alpha = alphaData[2*i+1];
                for (int j = 0; j < area; j++)
                {
                    float weight = weightData[i * area + j];
                    int value = min_value;
                    if (alpha > 1e-6f)
                    {
                        value = fmax(fmin(round((weight - min) / alpha) + min_value, max_value), min_value);
                    }
                    *tmp = mapWeight[value];
                    tmp++;
                }
            }
        } else {
            for (int i = 0; i < channel; i++)
            {
                float alpha = alphaData[i];
                for (int j = 0; j < area; j++)
                {
                    float weight = weightData[i * area + j];
                    int value = 0;
                    if (alpha > 1e-6f)
                    {
                        value = fmax(fmin(round(weight / alpha), max_value), min_value);
                    }
                    *tmp = mapWeight[value];
                    tmp++;
                }
            }
        }
        FillBuffer(buf, buf_len, arr, area * channel, iNeedBits);
        delete[] arr;
    }
    //begin write to file
    {
        char tmp[100];
        //1. weights blob shape(unsigned int32)
        shapeUseInt32 = WriteBlobDim(out, {channel, area});
        // 2. Avalable values Count(unsigned char)
        tmp[0] = (unsigned char)iCount;
        out.write(tmp, 1);
        // 3. valueset(signed char * valueset_size)
        for (auto it = setWeight.begin(); it != setWeight.end(); it++)
        {
            tmp[0] = (unsigned char)*it;
            out.write(tmp, 1);
        }
        // 4. weights indexes(size = ceil(0.125*weights_count*ceil(log2(Avalable_values_Count))))
        out.write(buf, buf_len);
        //g_totalSize += 1 + setWeight.size() + buf_len;
    }
    delete[] buf;
}

static bool WriteSparseQuanBlobs(std::ostream &out, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, bool& shapeUseInt32, const int bits)
{
    std::set<int> setWeight;
    GetWeightSet(setWeight, weightData, alphaData, area, channel, asymmetricQuantFlag, bits);
    int iDataNeedBits = ceil(log2(setWeight.size()));
    iDataNeedBits = iDataNeedBits < 1 ? 1 : iDataNeedBits;
    std::map<int, unsigned char> mapWeight;
    {
        int iIdx = 0;
        for (auto it = setWeight.begin(); it != setWeight.end(); it++)
        {
            mapWeight[*it] = iIdx++;
        }
    }
    unsigned int nnz = 0;
    int weightSize = area * channel;
    unsigned char iNeedBits;
    nnz = GetBestMaxStep(weightData, weightSize, iNeedBits, iDataNeedBits, alphaData, area, channel, asymmetricQuantFlag);
    if (nnz <= 0) {
        return false;
    }
    //weight buf
    size_t data_buf_len = size_t(ceil(0.125 * iDataNeedBits * nnz));
    char* data_buf = new char[data_buf_len];
    //sparse COO buf
    const int offset = 1 << (bits - 1);
    int min_value = -offset;
    int max_value = offset - 1;
    size_t buf_len = size_t(ceil(0.125 * iNeedBits * nnz));
    char* buf = new char[buf_len];
    { //fill buf with step values;
        unsigned char* arr_idx = new unsigned char[nnz];
        unsigned char* data_arr = new unsigned char[nnz];
        unsigned char* tmp = arr_idx;
        int iMaxStep = pow(2, iNeedBits) - 1;
        int iPreIdx = 0;
        unsigned char* dTmp = data_arr;
        if (asymmetricQuantFlag) {
            for (int i = 0; i < weightSize; i++)
            {
                float min = alphaData[2*(i/area)];
                float alpha = alphaData[2*(i/area)+1];
                int zeroQuant = min_value;
                if (alpha > 1e-6) {
                    zeroQuant = round((0.0f - min) / alpha) + min_value;
                }

                float weight = weightData[i];
                int value = min_value;
                if (alpha > 1e-6)
                {
                    value = round((weight - min) / alpha) + min_value;
                }

                if (value != zeroQuant)
                {
                    *dTmp = mapWeight[value];
                    *tmp = i - iPreIdx;
                    iPreIdx = i;
                    tmp++;
                    dTmp++;
                }
                if (i - iPreIdx >= iMaxStep)
                {
                    *dTmp = mapWeight[zeroQuant];
                    *tmp = i - iPreIdx;
                    iPreIdx = i;
                    tmp++;
                    dTmp++;
                }
            }
        } else {
            for (int i = 0; i < weightSize; i++)
            {
                float alpha = alphaData[i / area];
                float weight = weightData[i];
                int value = 0;
                if (alpha > 1e-6f)
                {
                    value = round(weight / alpha);
                }

                if (value != 0)
                {
                    *dTmp = mapWeight[value];
                    *tmp = i - iPreIdx;
                    iPreIdx = i;
                    tmp++;
                    dTmp++;
                }
                if (i - iPreIdx >= iMaxStep)
                {
                    *dTmp = mapWeight[0];
                    *tmp = i - iPreIdx;
                    iPreIdx = i;
                    tmp++;
                    dTmp++;
                }
            }
        }
        FillBuffer(buf, buf_len, (char*) arr_idx, nnz, iNeedBits);
        FillBuffer(data_buf, data_buf_len, (char*) data_arr, nnz, iDataNeedBits);
        delete[] arr_idx;
        delete[] data_arr;
    }
    { //write
        char tmp[100];
        // 1.weights blob shape(unsigned int32)
        shapeUseInt32 = WriteBlobDim(out, {channel, area});
        // 2. nnz
        out.write((const char*) &nnz, 4);
        // 3. max_step use # bits () (unsigned char)
        out.write((const char*) &iNeedBits, 1);
        // 4. buf for steps ceil(nnz*step need bits/8)
        out.write(buf, buf_len);
        // 5. Avalable values Count(unsigned char)
        tmp[0] = (unsigned char) setWeight.size();
        out.write(tmp, 1);
        // 6. valueset(signed char * valueset_size)
        for (auto it = setWeight.begin(); it != setWeight.end(); it++)
        {
            tmp[0] = (unsigned char) *it;
            out.write(tmp, 1);
        }
        // 7. none zero weights indexes(nnz*ceil(log2(Avalable_values_Count))/8)
        out.write((const char*) data_buf, data_buf_len);
    }
    delete[] buf;
    delete[] data_buf;
    return true;
}

static std::unique_ptr<IDSTQuanT> encode(const float* weight, const std::vector<float>& scale, int kernelSize, int kernelNum,
                                         bool asymmetricQuantFlag, const int8_t* quantWeightPtr, const int clampMin, const int bits = 8, bool detectSparse = true) {
        // compute block_size

    int alpha_size = scale.size(), block_size = kernelSize, block_num = 1;
    if (asymmetricQuantFlag) alpha_size /= 2;
    if (alpha_size > kernelNum) {
        block_num = alpha_size / kernelNum;
        block_size = kernelSize / block_num;
    }
    bool shapeUseInt32 = false;
    std::unique_ptr<IDSTQuanT> idst(new IDSTQuanT);
    std::ostringstream outputStringStreamCQ;
    WriteCQBlobs(outputStringStreamCQ, weight, scale.data(), kernelSize, kernelNum, asymmetricQuantFlag, shapeUseInt32, bits);
    auto cqStr = outputStringStreamCQ.str();
    if (detectSparse) {
        std::ostringstream outputStringStreamSQ;
        bool sparseValid = WriteSparseQuanBlobs(outputStringStreamSQ, weight, scale.data(), kernelSize, kernelNum, asymmetricQuantFlag, shapeUseInt32, bits);
        auto sqStr = outputStringStreamSQ.str();
        int int8Size = kernelNum * kernelSize;
        if (quantWeightPtr && (int8Size <= cqStr.size() && int8Size <= sqStr.size())) {
            idst->type = 4;
            idst->aMax = kernelNum;
            idst->buffer.resize(int8Size);
            ::memcpy(idst->buffer.data(), quantWeightPtr, int8Size);
        } else if (cqStr.size() <= sqStr.size() || (!sparseValid)) {
            idst->type = 1;
            idst->buffer.resize(cqStr.size());
            ::memcpy(idst->buffer.data(), cqStr.data(), cqStr.size());
        } else {
            idst->type = 2;
            idst->buffer.resize(sqStr.size());
            ::memcpy(idst->buffer.data(), sqStr.data(), sqStr.size());
        }
    } else {
        idst->type = 1;
        idst->buffer.resize(cqStr.size());
        ::memcpy(idst->buffer.data(), cqStr.data(), cqStr.size());
    }
    idst->shapeInt32 = shapeUseInt32;
    idst->alpha.resize(scale.size());
    ::memcpy(idst->alpha.data(), scale.data(), scale.size() * sizeof(float));
    idst->quantScale = 1.f;
    if (asymmetricQuantFlag) {
        idst->readType = kernelNum;
        idst->aMin = clampMin;
    }
    return idst;
}

} // namespace IDSTEncoder

#endif // IDSTENCODER_HPP
