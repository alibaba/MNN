//
//  rawDataTransform.cpp
//  MNN
//
//  Created by MNN on 2019/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <sstream>
#include <map>
#include "MNNDefine.h"
#include "converter/source/IR/MNN_generated.h"
#include "rapidjson/document.h"
using namespace MNN;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./rawDataTransform.out dataConfig.json data.bin\n");
        return 0;
    }
    FUNC_PRINT_ALL(argv[1], s);
    rapidjson::Document document;
    {
        std::ifstream fileNames(argv[1]);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        FUNC_PRINT(document.IsArray());
        FUNC_PRINT(document.IsObject());
    }
    auto dataConfig = document.GetObject();

    auto dataArray = dataConfig["data"].GetArray();
    std::unique_ptr<NetT> data(new NetT);
    std::vector<std::unique_ptr<OpT>> ops;
    for (auto iter = dataArray.begin(); iter != dataArray.end(); iter++) {
        auto dataObj = iter->GetObject();
        auto path    = dataObj["path"].GetString();
        std::ifstream read(path);
        if (read.fail()) {
            MNN_PRINT("Error to open %s\n", path);
            continue;
        }
        auto name = dataObj["name"].GetString();
        data->tensorName.emplace_back(name);
        std::unique_ptr<OpT> newOp(new OpT);
        newOp->type          = OpType_Const;
        newOp->name          = name;
        newOp->outputIndexes = {(int)data->tensorName.size()};
        newOp->main.type     = OpParameter_Blob;
        auto blobT           = new BlobT;
        newOp->main.value    = blobT;
        int size             = 1;
        auto dimArray        = dataObj["dim"].GetArray();
        for (auto dimIter = dimArray.begin(); dimIter != dimArray.end(); dimIter++) {
            auto dim = dimIter->GetInt();
            blobT->dims.emplace_back(dim);
            size *= dim;
        }
        blobT->dataFormat = MNN_DATA_FORMAT_NHWC;
        blobT->dataType   = DataType_DT_FLOAT;
        if (dataObj.HasMember("type")) {
            std::string dataType = dataObj["type"].GetString();
            static std::map<std::string, DataType> typeMaps {
                {"double", DataType_DT_FLOAT},//Use float instead of double
                {"float64", DataType_DT_FLOAT},//Use float instead of float64
                {"float", DataType_DT_FLOAT},
                {"float32", DataType_DT_FLOAT},
                {"int32", DataType_DT_INT32},
                {"int", DataType_DT_INT32},
                {"int64", DataType_DT_INT32},//Use int32 instead of int64
                {"int8", DataType_DT_INT8},
                {"char", DataType_DT_INT8},
                {"uint8", DataType_DT_UINT8},
                {"unsigned char", DataType_DT_UINT8},
            };
            auto iter = typeMaps.find(dataType);
            if (iter == typeMaps.end()) {
                MNN_ERROR("Error for name=%s, type=%s\n", name, dataType.c_str());
                continue;
            }
            blobT->dataType = iter->second;
        }
        switch (blobT->dataType) {
            case MNN::DataType_DT_FLOAT:
                blobT->float32s.resize(size);
                for (int i = 0; i < size; ++i) {
                    read >> blobT->float32s[i];
                }
                break;
            case MNN::DataType_DT_INT32:
                blobT->int32s.resize(size);
                for (int i = 0; i < size; ++i) {
                    read >> blobT->int32s[i];
                }
                break;
            case MNN::DataType_DT_INT8:
                blobT->int8s.resize(size);
                for (int i = 0; i < size; ++i) {
                    read >> blobT->int8s[i];
                }
                break;
            case MNN::DataType_DT_UINT8:
                blobT->uint8s.resize(size);
                for (int i = 0; i < size; ++i) {
                    read >> blobT->uint8s[i];
                }
                break;
            default:
                MNN_ERROR("Error for load name=%s\n", name);
                break;
        }
        data->oplists.emplace_back(std::move(newOp));
    }
    {
        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = Net::Pack(builder, data.get());
        builder.Finish(offset);
        std::ofstream os(argv[2]);
        os.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    }

    return 0;
}
