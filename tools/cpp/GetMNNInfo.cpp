//
//  ModuleBasic.cpp
//  MNN
//
//  Created by MNN on 2021/10/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include "rapidjson/document.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <set>
using namespace MNN::Express;
using namespace MNN;
static std::string _getDataType(const halide_type_t& type) {
    switch (type.code) {
        case halide_type_float:
            if (type.bits == 32) {
                return "float";
            }
            if (type.bits == 16) {
                return "half";
            }
            break;
        case halide_type_uint:
            if (type.bits == 32) {
                return "uint32";
            }
            if (type.bits == 16) {
                return "uint16";
            }
            if (type.bits == 8) {
                return "uint8";
            }
            break;
        case halide_type_int:
            if (type.bits == 32) {
                return "int32";
            }
            if (type.bits == 16) {
                return "int16";
            }
            if (type.bits == 8) {
                return "int8";
            }
            break;
        default:
            break;
    }
    return "Unknown";
}
static std::string _getFormatString(Dimensionformat format) {
    switch (format) {
        case MNN::Express::NCHW:
            return "NCHW";
        case MNN::Express::NHWC:
            return "NHWC";
        case MNN::Express::NC4HW4:
            return "NC4HW4";
        default:
            break;
    }
    return "Unknown";
}
int main(int argc, char *argv[]) {
    if (argc < 2) {
        MNN_ERROR("Usage: ./GetMNNInfo ${test.mnn}\n");
        return 0;
    }
    std::string modelName = argv[1];
    std::vector<std::string> empty;
    std::shared_ptr<Module> net(Module::load(empty, empty, argv[1]));
    if (nullptr == net.get()) {
        MNN_ERROR("Load MNN from %s Failed\n", argv[1]);
        return 1;
    }
    auto info = net->getInfo();
    MNN_ASSERT(info->inputNames.size() == info->inputs.size());
    MNN_PRINT("Model default dimensionFormat is %s\n", _getFormatString(info->defaultFormat).c_str());
    MNN_PRINT("Model Inputs:\n");
    for (int i=0; i<info->inputNames.size(); ++i) {
        auto& varInfo = info->inputs[i];
        MNN_PRINT("[ %s ]: dimensionFormat: %s, ", info->inputNames[i].c_str(), _getFormatString(varInfo.order).c_str());
        MNN_PRINT("size: [ ");
        if (varInfo.dim.size() > 0) {
            for (int j=0; j<(int)varInfo.dim.size() - 1; ++j) {
                MNN_PRINT("%d,", varInfo.dim[j]);
            }
            MNN_PRINT("%d ", varInfo.dim[(int)varInfo.dim.size() - 1]);
        }
        MNN_PRINT("], ");
        MNN_PRINT("type is %s\n", _getDataType(varInfo.type).c_str());
    }
    MNN_PRINT("Model Outputs:\n");
    for (int i=0; i<info->outputNames.size(); ++i) {
        MNN_PRINT("[ %s ]\n", info->outputNames[i].c_str());
    }
    if (info->version.empty()) {
        MNN_PRINT("Model Version: < 2.0.0\n");
    } else {
        MNN_PRINT("Model Version: %s \n", info->version.c_str());
    }
    if (!info->bizCode.empty()) {
        MNN_PRINT("Model bizCode: %s\n", info->bizCode.c_str());
    }
    if (!info->metaData.empty()) {
        MNN_PRINT("MetaData: Begin \n");
        for (auto& iter : info->metaData) {
            MNN_PRINT("[Meta] %s : %s\n", iter.first.c_str(), iter.second.c_str());
        }
        MNN_PRINT("MetaData: End \n");
    }
    MNN_PRINT("Get Op info to op.txt\n");
    std::set<std::string> originTypes;
    {
        MNN_PRINT("Appen op lists to op.txt\n");
        std::ifstream is("op.txt");
        if (!is.fail()) {
            std::string tmp;
            while (std::getline(is, tmp, '\n')) {
                originTypes.insert(tmp);
            }
        }
    }
    MNN_PRINT("Origin Op: %d\n", originTypes.size());

    // Load origin tyes
    {
        std::shared_ptr<MNN::Interpreter> bufferTmp(MNN::Interpreter::createFromFile(modelName.c_str()));
        auto buffer = bufferTmp->getModelBuffer();
        auto collect = [&](const flatbuffers::Vector<flatbuffers::Offset<Op>>* oplists) {
            for (int i=0; i<oplists->size(); ++i) {
                auto op = oplists->GetAs<Op>(i);
                originTypes.insert(EnumNameOpType(op->type()));
            }
        };
        auto net = GetNet(buffer.first);
        collect(net->oplists());
        if (nullptr != net->subgraphs()) {
            for (int i=0; i<net->subgraphs()->size(); ++i) {
                auto graph = net->subgraphs()->GetAs<SubGraphProto>(i);
                collect(graph->nodes());
            }
        }
    }
    MNN_PRINT("Current Op: %d\n", originTypes.size());

    {
        MNN_PRINT("Appen op lists to op.txt\n");
        std::ofstream os("op.txt");
        for (auto& s : originTypes) {
            os << s << std::endl;
        }
    }
    return 0;
}

