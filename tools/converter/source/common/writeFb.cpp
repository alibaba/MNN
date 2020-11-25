//
//  writeFb.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <iostream>
#include <algorithm>
#include <set>

#include "MNN_generated.h"
#include "half.hpp"
#include "logkit.h"
#include "writeFb.hpp"
#include "cpp/ConfigFile.hpp"
#include <MNN/MNNDefine.h>
#include "cli.hpp"
#include "../../common/Global.hpp"

using namespace MNN;
using namespace std;

static float findAbsMax(const float *weights, const int count) {
    float absMax = abs(weights[0]);
    for (int i = 1; i < count; i++) {
        float value = abs(weights[i]);
        if (value > absMax) {
            absMax = value;
        }
    }

    return absMax;
}

static std::vector<float> findMinMax(const float *weights, const int count) {
    float min = weights[0];
    float max = weights[0];

    for (int i = 1; i < count; i++) {
        float value = weights[i];
        if (value > max) {
            max = value;
        }
        if (value < min) {
            min = value;
        }
    }

    return {min, max};
}

static void WriteBlobDim(ostream &out, std::vector<int> dims)
{
    char tmp[4];
    ((unsigned char *)tmp)[0] = (unsigned char)dims.size();
    out.write(tmp, 1);
    for (int i = 0; i < dims.size(); i++)
    {
        unsigned short tmpShort = (unsigned short)dims[i];
        out.write((const char*)(&tmpShort), 2);
    }
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

static void GetWeightSet(set<int> &setWeight, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag)
{
    setWeight.clear();
    if (asymmetricQuantFlag) {
        for (int i = 0; i < channel; i++)
        {
            float min = alphaData[2*i];
            float alpha = alphaData[2*i+1];
            if (alpha <= 1e-6f)
            {
                setWeight.insert(-128);
                continue;
            }
            for (int j = 0; j < area; j++)
            {
                float weight = weightData[i * area + j];
                setWeight.insert(round((weight - min) / alpha) + (-128));
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
                setWeight.insert(round(weight / alpha));
            }
        }
    }
}

static float GetSparsity(const float* weightData, int weightSize, unsigned int& nnz, const float* alphaData, int area, int channel, bool asymmetricQuantFlag, int iMaxStep = -1)
{
	nnz = 0;
	int iPreIdx = 0;
	float sparsity;
    if (asymmetricQuantFlag) {
        for (int i = 0; i < weightSize; i++)
        {
            float min = alphaData[2*(i/area)];
            float alpha = alphaData[2*(i/area)+1];
            int zeroQuant = -128;
            if (alpha > 1e-6) {
                zeroQuant = round((0.0f - min) / alpha) + (-128);
            }

            float weight = weightData[i];
            int value = -128;
            if (alpha > 1e-6)
            {
                value = round((weight - min) / alpha) + (-128);
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

unsigned int GetBestMaxStep(const float* weightData, int weightSize, unsigned char& iMaxStepBits, int BlobDataSize, const float* alphaData, int area, int channel, bool asymmetricQuantFlag)
{
	size_t szBestSize = 1000000000;
	unsigned int best_nnz = 0;
	for (int i = 2; i < 9; i++)
	{
		unsigned int nnz = 0;
		GetSparsity(weightData, weightSize, nnz, alphaData, area, channel, asymmetricQuantFlag, pow(2, i) - 1);
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

static void WriteCQBlobs(ostream &out, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag)
{
    //push values into buffer
    //Find int values in all blobs and check;
    set<int> setWeight;
    GetWeightSet(setWeight, weightData, alphaData, area, channel, asymmetricQuantFlag);
    int iCount = setWeight.size();
    int iNeedBits = ceil(log2(iCount));
    if (iNeedBits > 8) {
        MNN_ERROR("The Bits need large than 8, the model may be error for user\n");
        return;
    }
    map<int, unsigned char> mapWeight;
    int iIdx = 0;
    for (set<int>::iterator it = setWeight.begin(); it != setWeight.end(); it++)
    {
        mapWeight[*it] = iIdx++;
    }
    size_t buf_len = size_t(ceil(0.125 * iNeedBits * area * channel));
    char *buf = new char[buf_len];
    {
        char *arr = new char[area * channel];
        char *tmp = arr;
        if (asymmetricQuantFlag) {
            for (int i = 0; i < channel; i++)
            {
                float min = alphaData[2*i];
                float alpha = alphaData[2*i+1];
                for (int j = 0; j < area; j++)
                {
                    float weight = weightData[i * area + j];
                    int value = -128;
                    if (alpha > 1e-6f)
                    {
                        value = round((weight - min) / alpha) + (-128);
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
                        value = round(weight / alpha);
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
        WriteBlobDim(out, {channel, area});
        // 2. Avalable values Count(unsigned char)
        tmp[0] = (unsigned char)iCount;
        out.write(tmp, 1);
        // 3. valueset(signed char * valueset_size)
        for (set<int>::iterator it = setWeight.begin(); it != setWeight.end(); it++)
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

static void WriteSparseQuanBlobs(ostream &out, const float* weightData, const float* alphaData, int area, int channel, bool asymmetricQuantFlag)
{
	set<int> setWeight;
	GetWeightSet(setWeight, weightData, alphaData, area, channel, asymmetricQuantFlag);
	int iDataNeedBits = ceil(log2(setWeight.size()));
	unsigned int nnz = 0;
    int weightSize = area * channel;
	map<int, unsigned char> mapWeight;
	{
		int iIdx = 0;
		for (set<int>::iterator it = setWeight.begin(); it != setWeight.end(); it++)
		{
			mapWeight[*it] = iIdx++;
		}
	}
	unsigned char iNeedBits;
	nnz = GetBestMaxStep(weightData, weightSize, iNeedBits, iDataNeedBits, alphaData, area, channel, asymmetricQuantFlag);
    //weight buf
	size_t data_buf_len = size_t(ceil(0.125 * iDataNeedBits * nnz));
	char* data_buf = new char[data_buf_len];
    //sparse COO buf
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
                int zeroQuant = -128;
                if (alpha > 1e-6) {
                    zeroQuant = round((0.0f - min) / alpha) + (-128);
                }

                float weight = weightData[i];
                int value = -128;
                if (alpha > 1e-6)
                {
                    value = round((weight - min) / alpha) + (-128);
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
		WriteBlobDim(out, {channel, area});
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
		for (set<int>::iterator it = setWeight.begin(); it != setWeight.end(); it++)
		{
			tmp[0] = (unsigned char) *it;
			out.write(tmp, 1);
		}
		// 7. none zero weights indexes(nnz*ceil(log2(Avalable_values_Count))/8)
		out.write((const char*) data_buf, data_buf_len);
	}
	delete[] buf;
	delete[] data_buf;
}

int writeFb(std::unique_ptr<MNN::NetT>& netT, const std::string& MNNModelFile, modelConfig config) {
    auto RemoveParams = [](std::unique_ptr<MNN::OpT>& op) {
        const auto opType = op->type;
        switch (opType) {
            case MNN::OpType_Convolution:
            case MNN::OpType_Deconvolution:
            case MNN::OpType_ConvolutionDepthwise: {
                auto param = op->main.AsConvolution2D();
                param->weight.clear();
                param->bias.clear();
                break;
            }
            case MNN::OpType_TfQuantizedConv2D: {
                auto param = op->main.AsTfQuantizedConv2D();
                param->weight.clear();
                param->bias.clear();
                break;
            }
            case MNN::OpType_MatMul: {
                auto param = op->main.AsMatMul();
                param->weight.clear();
                param->bias.clear();
                break;
            }
            case MNN::OpType_BatchNorm: {
                auto param = op->main.AsBatchNorm();
                param->slopeData.clear();
                param->meanData.clear();
                param->varData.clear();
                param->biasData.clear();
                param->Adata.clear();
                param->Bdata.clear();
                break;
            }
            case MNN::OpType_Scale: {
                auto param = op->main.AsScale();
                param->scaleData.clear();
                param->biasData.clear();
                break;
            }
            default:
                break;
        }
    };
    if (config.benchmarkModel) {
        for (auto& op : netT->oplists) {
            RemoveParams(op);
        }
        for (auto& subgraph : netT->subgraphs) {
            for (auto& op : subgraph->nodes) {
                RemoveParams(op);
            }
        }
    }

    auto CastParamsToHalf = [](std::unique_ptr<MNN::OpT>& op) {
        const auto opType = op->type;
        switch (opType) {
            case MNN::OpType_Convolution:
            case MNN::OpType_ConvolutionDepthwise: {
                auto param           = op->main.AsConvolution2D();
                const int weightSize = param->weight.size();
                // const int biasSize = param->bias.size();
                std::vector<half_float::half> quantizedFp16Weight;
                quantizedFp16Weight.resize(weightSize);
                std::transform(param->weight.begin(), param->weight.end(), quantizedFp16Weight.begin(),
                               [](float w) { return half_float::half(w); });
                // std::vector<half_float::half> quantizedFp16Bias;
                // quantizedFp16Bias.resize(biasSize);
                // std::transform(param->bias.begin(), param->bias.end(), quantizedFp16Bias.begin(), [](float
                // b){return half_float::half(b); });
                param->weight.clear();
                // param->bias.clear();

                param->quanParameter.reset(new MNN::IDSTQuanT);
                param->quanParameter->type = 3;
                int8_t* halfWeight         = reinterpret_cast<int8_t*>(quantizedFp16Weight.data());
                param->quanParameter->buffer.assign(halfWeight, halfWeight + sizeof(half_float::half) * weightSize);
                break;
            }
            case MNN::OpType_Const: {
                auto blob = op->main.AsBlob();
                if (blob->dataType == MNN::DataType_DT_FLOAT) {
                    blob->dataType = MNN::DataType_DT_HALF;
                    blob->uint8s.resize(sizeof(half_float::half) * blob->float32s.size());
                    auto size = blob->float32s.size();
                    auto dst = (half_float::half*)blob->uint8s.data();
                    for (int i=0; i<size; ++i) {
                        dst[i] = blob->float32s[i];
                    }
                    blob->float32s.clear();
                }
                break;
            }
            default:
                break;
        }
    };
    if (config.saveHalfFloat) {
        for (auto& op : netT->oplists) {
            CastParamsToHalf(op);
        }
        for (auto& subgraph : netT->subgraphs) {
            for (auto& op : subgraph->nodes) {
                CastParamsToHalf(op);
            }
        }
    }

    auto CastParamsToInt8 = [](std::unique_ptr<MNN::OpT>& op, int bits) {
        auto gConverterConfig = Global<modelConfig>::Get();
        bool asymmetricQuantFlag = gConverterConfig->weightQuantAsymmetric;
        const auto opType = op->type;
        // Bits must from 2-8
        bits = std::max(bits, 2);
        bits = std::min(bits, 8);
        switch (opType) {
            case MNN::OpType_Convolution:
            case MNN::OpType_ConvolutionDepthwise:
            case MNN::OpType_Deconvolution:
            case MNN::OpType_DeconvolutionDepthwise: {
                auto param           = op->main.AsConvolution2D();
                auto& common = param->common;
                float thredhold = (float)(1 << (bits - 1)) - 1.0f;
                if (param->quanParameter.get() == nullptr) {
                    const int weightSize = param->weight.size();
                    int kernelNum = common->outputCount;
                    int kernelSize = weightSize / kernelNum;
                    auto weightData = param->weight.data();

                    std::vector<float> scales;
                    if (asymmetricQuantFlag) {
                        scales.resize(kernelNum*2);
                        for (int k = 0; k < kernelNum; k++) {
                            int beginIndex = k * kernelSize;
                            auto minAndMax = findMinMax(weightData + beginIndex, kernelSize);
                            float min = minAndMax[0];
                            float max = minAndMax[1];
                            float scale = (max - min) / (127 + 128);

                            scales[2*k] = min;
                            scales[2*k+1] = scale;
                        }
                    } else {
                        scales.resize(kernelNum);
                        for (int k = 0; k < kernelNum; k++) {
                            int beginIndex = k * kernelSize;
                            auto absMax = findAbsMax(weightData + beginIndex, kernelSize);

                            scales[k] = absMax / thredhold;
                        }
                    }
                    
                    std::ostringstream outputStringStreamCQ;
                    WriteCQBlobs(outputStringStreamCQ, weightData, scales.data(), kernelSize, kernelNum, asymmetricQuantFlag);
                    std::ostringstream outputStringStreamSQ;
                    WriteSparseQuanBlobs(outputStringStreamSQ, weightData, scales.data(), kernelSize, kernelNum, asymmetricQuantFlag);

                    param->quanParameter.reset(new MNN::IDSTQuanT);
                    auto tempString = outputStringStreamCQ.str();
                    param->quanParameter->type = 1;
                    if (outputStringStreamSQ.str().size() < tempString.size()) {
                        tempString = outputStringStreamSQ.str();
                        param->quanParameter->type = 2;
                    }
                    param->weight.clear();
                    param->quanParameter->buffer.resize(tempString.size());
                    ::memcpy(param->quanParameter->buffer.data(), tempString.data(), tempString.size());
                    param->quanParameter->alpha = std::move(scales);
                    param->quanParameter->quantScale = 1.0f;
                    if (asymmetricQuantFlag) {
                        param->quanParameter->readType = kernelNum;
                    }
                }
                break;
            }
            default:
                break;
        }
    };
    if (config.weightQuantBits > 0) {
        for (auto& op : netT->oplists) {
            CastParamsToInt8(op, config.weightQuantBits);
        }
        for (auto& subgraph : netT->subgraphs) {
            for (auto& op : subgraph->nodes) {
                CastParamsToInt8(op, config.weightQuantBits);
            }
        }
    }

    std::set<std::string> notSupportOps;
    auto CheckIfNotSupported = [&] (const std::unique_ptr<MNN::OpT>& op) {
        if (op->type == MNN::OpType_Extra) {
            if (op->main.AsExtra()->engine != "MNN") {
                notSupportOps.insert(op->main.AsExtra()->engine + "::" + op->main.AsExtra()->type);
            }
        }
    };
    for (auto& op : netT->oplists) {
        CheckIfNotSupported(op);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            CheckIfNotSupported(op);
        }
    }

    std::ostringstream notSupportInfo;
    if (!notSupportOps.empty()) {
        for (auto name : notSupportOps) {
            notSupportInfo << name << " | ";
        }
        auto opNames = notSupportInfo.str();
        LOG(FATAL) << "These Op Not Support: " << opNames.substr(0, opNames.size() - 2);
    }

    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);
    int sizeOutput    = builderOutput.GetSize();
    auto bufferOutput = builderOutput.GetBufferPointer();

    if (config.saveStaticModel && netT->usage != MNN::Usage_INFERENCE_STATIC) {
        std::map<std::string, std::vector<int>> inputConfig;
        // get config to set input size
        if (config.inputConfigFile.size() > 0) {
            ConfigFile conf(config.inputConfigFile);
            auto numOfInputs = conf.Read<int>("input_size");
            auto inputNames  = splitNames(numOfInputs, conf.Read<std::string>("input_names"));
            auto inputDims   = splitDims(numOfInputs, conf.Read<std::string>("input_dims"));
            for (int i = 0; i < numOfInputs; i++) {
                inputConfig.insert(std::make_pair(inputNames[i], inputDims[i]));
            }
        }
        const Net* net = flatbuffers::GetRoot<MNN::Net>(bufferOutput);
        converToStaticModel(net, inputConfig, MNNModelFile);
    } else {
        std::ofstream output(MNNModelFile, std::ofstream::binary);
        output.write((const char*)bufferOutput, sizeOutput);
    }

#ifdef MNN_DUMP_SUBGRAPH
    for (int i = 0; i < netT->subgraphs.size(); ++i) {
        std::unique_ptr<MNN::NetT> subnet(new MNN::NetT);
        auto& subgraph = netT->subgraphs[i];
        subnet->oplists = std::move(subgraph->nodes);
        subnet->tensorName = subgraph->tensors;
        subnet->sourceType = netT->sourceType;
        subnet->bizCode = netT->bizCode;

        flatbuffers::FlatBufferBuilder builder(1024);
        builder.ForceDefaults(true);
        auto len = MNN::Net::Pack(builder, subnet.get());
        builder.Finish(len);
        int output_size = builder.GetSize();
        auto* output_ptr = builder.GetBufferPointer();

        std::string filename =
            MNNModelFile + "_subgraph_" + std::to_string(i) + ".mnn";
        std::ofstream output(filename.c_str(), std::ofstream::binary);
        output.write((const char*)output_ptr, output_size);
    }
#endif
    return 0;
}
