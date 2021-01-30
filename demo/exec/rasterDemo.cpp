//
//  rasterDemo.cpp
//  MNN
//
//  Created by MNN on 2020/10/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/Execution.hpp"
#include "core/Backend.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
using namespace MNN;
/*
 1.Raster will do the index mapping like below:

    for (region : regions)
        src = region.src, dst = region.dst;
    for (i = 0 -> size[0])
    for (j = 0 -> size[1])
    for (k = 0 -> size[2])
        output[dst.offset + i * dst.stride[0] + j * dst.stride[1] + k * dst.stride[2]] =
        region.origion[src.offset + i * src.stride[0] + j * src.stride[1] + k * src.stride[2]];

 2. Raster Op has a input and a output, but the input is not the real input tensor, it's a
    middle tensor whith VIRTUAL type that has many regions point to inputs tensors, like below.

                input_0 --> region_0 --\
                                        \
                input_1 --> region_1 ---- middle ----> output
                                        /
                input_2 --> region_2 --/

 3. This example read a json file and construct some Rasters and compute.
    Example input file at $<MNN-ROOT>/resource/exec/rasterDemo_transpose.json
    The input json file format is as below:
    {
       "inputs" : [
           {
               "id" : int,
               "type" : "type_name", // float or int
               "dims" : [int],
               "data" : [int/float] // if null, fill with random number
           }
       ],
       "outputs" : [
           // same with inputs
       ],
       "regions" : [
           {
               "id" : int, // points to outputs
               "size" : [int],
               "src" : {
                   "offset" : int,
                   "stride" : [int]
               },
               "dst" : { // same with src },
               "origin" : int // point to inputs
           }
       ]
    }
 */

static std::string runRaster(std::string jsonString, int runNum) {
    srand(0);
    rapidjson::Document document;
    document.Parse(jsonString.c_str());
    if (document.HasParseError()) {
        MNN_ERROR("Invalid Json Format!\n");
        return 0;
    }

    // prepare CPU backend
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_High;
    config.backendConfig = &backendConfig;
    Backend::Info compute;
    compute.type = config.type;
    compute.numThread = config.numThread;
    compute.user = config.backendConfig;
    const RuntimeCreator* runtimeCreator(MNNGetExtraRuntimeCreator(compute.type));
    std::unique_ptr<Runtime> runtime(runtimeCreator->onCreate(compute));
    std::unique_ptr<Backend> backend(runtime->onCreate());

    // build Op
    std::unique_ptr<OpT> opt(new OpT);
    opt->type = OpType_Raster;
    flatbuffers::FlatBufferBuilder builder(1024);
    builder.ForceDefaults(true);
    auto len = Op::Pack(builder, opt.get());
    builder.Finish(len);
    auto buffer = builder.GetBufferPointer();
    const Op* op = flatbuffers::GetMutableRoot<Op>(buffer);
    // build tensors (NCHW) from json
    std::vector<std::unique_ptr<Tensor>> inputs;
    std::vector<std::unique_ptr<Tensor>> outputs;
    auto readTensors = [&document, &backend](std::vector<std::unique_ptr<Tensor>>& tensors, const char* type) {
        if (document.HasMember(type)) {
            auto info = document[type].GetArray();
            tensors.resize(info.Size());
            for (auto iter = info.begin(); iter != info.end(); iter++) {
                auto obj = iter->GetObject();
                int id = obj["id"].GetInt();
                tensors[id].reset(new Tensor(4));
                auto tensor = tensors[id].get();
                auto dataType = obj["type"].GetString();
                bool isFloat = !strcmp(dataType, "float");
                tensor->setType(isFloat ? DataType_DT_FLOAT : DataType_DT_INT32);
                auto dims = obj["dims"].GetArray();
                for (auto d = dims.begin(); d != dims.end(); d++) {
                    tensor->setLength(d - dims.begin(), d->GetInt());
                }
                TensorUtils::setLinearLayout(tensor);
                backend->onAcquireBuffer(tensor, Backend::STATIC);
                TensorUtils::getDescribe(tensor)->backend = backend.get();
                auto data = obj["data"].GetArray();
                if (!strcmp(type, "inputs")) {
                    bool hasData = data.Size() == tensor->elementSize();
                    auto dataIter = data.begin();
                    for (int i = 0; i < tensor->elementSize(); i++, dataIter++) {
                        if (isFloat) {
                            tensor->host<float>()[i] = hasData ? dataIter->GetFloat() : rand() % 10 / 10.0;
                        } else {
                            tensor->host<int>()[i] = hasData ? dataIter->GetInt() : rand() % 10;
                        }
                    }
                }
            }
        }
    };
    readTensors(inputs, "inputs");
    readTensors(outputs, "outputs");

    // build middle tensors' region info from json
    std::vector<std::unique_ptr<Tensor>> middles;
    middles.resize(outputs.size());
    if (document.HasMember("regions")) {
        auto info = document["regions"].GetArray();
        for (auto iter = info.begin(); iter != info.end(); iter++) {
            auto obj = iter->GetObject();
            int id = obj["id"].GetInt();
            if (middles[id] == nullptr) {
                middles[id].reset(new Tensor(4));
            }
            auto des = TensorUtils::getDescribe(middles[id].get());
            des->memoryType = MNN::Tensor::InsideDescribe::MEMORY_VIRTUAL;
            Tensor::InsideDescribe::Region region;
            int origin = obj["origin"].GetInt();
            region.origin = inputs[origin].get();
            auto size = obj["size"].GetArray();
            auto src = obj["src"].GetObject();
            auto dst = obj["dst"].GetObject();
            auto srcStride = src["stride"].GetArray();
            auto dstStride = dst["stride"].GetArray();
            for (int i = 0; i < 3; i++) {
                region.size[i] = size[i].GetInt();
                region.src.stride[i] = srcStride[i].GetInt();
                region.dst.stride[i] = dstStride[i].GetInt();
            }
            region.src.offset = src["offset"].GetInt();
            region.dst.offset = dst["offset"].GetInt();
            des->regions.push_back(region);
        }
    }

    // build execution of Raster and run them
    for (int i = 0; i < outputs.size(); i++) {
        std::vector<Tensor*> ins = {middles[i].get()}, outs = {outputs[i].get()};
        std::unique_ptr<Execution> exe(backend->onCreate(ins, outs, op));
        exe->onResize(ins, outs);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < runNum; j++) {
            exe->onExecute(ins, outs);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        double time = time_span.count() * 1000.0 / runNum;
        printf("For output_id = %d, run %d times, the average time is %f ms.\n", i, runNum, time);
    }

    auto writeTensors = [&document](std::vector<std::unique_ptr<Tensor>>& tensors, const char* type) {
        auto info = document[type].GetArray();
        for (auto iter = info.begin(); iter != info.end(); iter++) {
            auto obj = iter->GetObject();
            int id = obj["id"].GetInt();
            auto data = obj["data"].GetArray();
            if (data.Size() == tensors[id]->elementSize()) {
                // has data, dont write
                return;
            }
            bool isFloat = !strcmp(obj["type"].GetString(), "float");
            data.Reserve(tensors[id]->elementSize(), document.GetAllocator());
            for (int i = 0; i < tensors[id]->elementSize(); i++) {
                if (isFloat) {
                    data.PushBack(tensors[id]->host<float>()[i], document.GetAllocator());
                } else {
                    data.PushBack(tensors[id]->host<int>()[i], document.GetAllocator());
                }
            }
        }
    };
    writeTensors(inputs, "inputs");
    writeTensors(outputs, "outputs");
    rapidjson::StringBuffer stringBuffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(stringBuffer);
    document.Accept(writer);
    return stringBuffer.GetString();
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printf("Usage: ./rasterDemo.out input.json [output.json] [runNum]\ndefault output is input, and default runNum is 100.\n");
        return 0;
    }
    const char* inputFile = argv[1];
    const char* outputFile = argv[1];
    int runNum = 100;
    if (argc >= 3) {
        outputFile = argv[2];
    }
    if (argc >= 4) {
        runNum = ::atoi(argv[3]);
    }
    std::ifstream in(inputFile);
    if (in.fail()) {
        printf("Invalid input Json File!\n");
        return 0;
    }
    std::ofstream out(outputFile);
    if (out.fail()) {
        printf("Invalid output Json File!\n");
        return 0;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    out << runRaster(ss.str(), runNum);
    out.close();
    printf("Run Raster Done!\n");
    return 0;
}
