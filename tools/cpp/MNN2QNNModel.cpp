#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/OpCommonUtils.hpp"
#include "MNN_generated.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <rapidjson/document.h>
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "core/MNNFileUtils.h"

using namespace rapidjson;

using namespace MNN::Express;
using namespace MNN;

static std::string modifyQnnConfigFile(const std::string qnnContextConfig, const std::string qnnSdkPath, std::vector<std::string> newNamesVec, const std::string dstModelPath) {
    // load context_config.json file and modify contents
    std::string newContextConfigPath = dstModelPath + std::string("/new_context_config.json");
    {
        std::string oriHtpConfigPath;
        std::ifstream config_file(qnnContextConfig);
        if (config_file.is_open()) {
            std::ostringstream ostr;
            ostr << config_file.rdbuf();
            rapidjson::Document doc;
            doc.Parse(ostr.str().c_str());
            if (doc.HasParseError()) {
                MNN_ERROR("Parse qnn context_config.json error!\n");
                return "";
            }

            // 3) 找到 backend_extensions 对象
            if (!doc.HasMember("backend_extensions") || !doc["backend_extensions"].IsObject()) {
                std::cerr << "backend_extensions missing or not an object\n";
                return "";
            }
            Value &be = doc["backend_extensions"];

            // 4) 修改 shared_library_path 的内容
            std::string shared_lib_path = qnnSdkPath + std::string("/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so");
            if (be.HasMember("shared_library_path") && be["shared_library_path"].IsString()) {
                be["shared_library_path"].SetString(shared_lib_path.c_str(), doc.GetAllocator());
            } else {
                be.AddMember("shared_library_path",
                            Value().SetString(shared_lib_path.c_str(), doc.GetAllocator()),
                            doc.GetAllocator());
            }

            // 4) 修改 config_file_path 的内容
            std::string config_file_path = dstModelPath + std::string("/new_htp_backend_extensions.json");
            if (be.HasMember("config_file_path") && be["config_file_path"].IsString()) {
                oriHtpConfigPath = be["config_file_path"].GetString();
                be["config_file_path"].SetString(config_file_path.c_str(), doc.GetAllocator());
            } else {
                be.AddMember("config_file_path",
                            Value().SetString(config_file_path.c_str(), doc.GetAllocator()),
                            doc.GetAllocator());
            }


            // 5) 将修改后的 JSON 写回文件（美化输出）
            StringBuffer buffer;
            PrettyWriter<StringBuffer> writer(buffer);
            doc.Accept(writer);

            std::ofstream ofs(newContextConfigPath);
            if (!ofs) {
                std::cerr << "Failed to open " << newContextConfigPath << " for writing\n";
                return "";
            }
            ofs << buffer.GetString();
            ofs.close();
        } else {
            MNN_ERROR("Open qnn context_config.json error!\n");
            return "";
        }


        // 修改 new_htp_backend_extensions.json内容
        if(!oriHtpConfigPath.empty()) {
            std::ifstream config_file(oriHtpConfigPath);
            if (config_file.is_open()) {
                std::ostringstream ostr;
                ostr << config_file.rdbuf();
                rapidjson::Document doc;
                doc.Parse(ostr.str().c_str());
                if (doc.HasParseError()) {
                    MNN_ERROR("Parse qnn context_config.json error!\n");
                    return "";
                }

                // 3) 找到 graphs[0].graph_names
                if (!doc.HasMember("graphs") || !doc["graphs"].IsArray() || doc["graphs"].Size() == 0) {
                    MNN_ERROR("graphs missing or empty\n");
                    return "";
                }
                Value &graphs = doc["graphs"];
                Value &graph0 = graphs[0];
                if (!graph0.IsObject()) {
                    MNN_ERROR("graphs[0] is not an object\n");
                    return "";
                }

                // 如果 graph_names 存在且是数组，则清空并重新填充
                if (graph0.HasMember("graph_names") && graph0["graph_names"].IsArray()) {
                    Value &names = graph0["graph_names"];
                    names.Clear();
                    for (const auto &s : newNamesVec) {
                        names.PushBack(Value().SetString(s.c_str(), (SizeType)s.length(), doc.GetAllocator()),
                                    doc.GetAllocator());
                    }
                } else {
                    // 如果不存在或不是数组，直接创建一个新的数组并赋值
                    Value newNames(kArrayType);
                    for (const auto &s : newNamesVec) {
                        newNames.PushBack(Value().SetString(s.c_str(), (SizeType)s.length(), doc.GetAllocator()),
                                        doc.GetAllocator());
                    }
                    graph0.AddMember("graph_names", newNames, doc.GetAllocator());
                }

                // 4) 将修改后的 JSON 写回文件（美化输出）
                StringBuffer buffer;
                PrettyWriter<StringBuffer> writer(buffer);
                doc.Accept(writer);

                std::string newHtpConfigPath = dstModelPath + std::string("/new_htp_backend_extensions.json");
                std::ofstream ofs(newHtpConfigPath);
                if (!ofs) {
                    MNN_ERROR("Failed to open %s for writing\n", newHtpConfigPath.c_str());
                    return "";
                }
                ofs << buffer.GetString();
                ofs.close();
            }
        } else {
            MNN_ERROR("no oriHtpConfigPath, please fill in new_htp_backend_extensions.json\n");
        }
    }
    return newContextConfigPath;
}

static bool parseDims(const std::string& s, std::vector<std::vector<int>>& out) {
    auto isLegal = [](char c) { 
        return c == 'x' || c == '_' || (c >= '0' && c <= '9'); 
    };
    bool allLegal = std::all_of(s.begin(), s.end(), isLegal);
    if(!allLegal) {
        return false;
    }

    out.clear();
    std::stringstream ss(s);
    std::string segment;
    MNN_PRINT("param dims: %s\n", s.c_str());
    while (std::getline(ss, segment, '_')) {
        if (segment.empty()) {
            MNN_ERROR("Param argv[7]: %s parse error, format should be like 1x3x512x512;1x3x256x256;1x3x512x256\n", s.c_str());
            return false;
        }
        std::vector<int> dims;
        std::stringstream inner(segment);
        std::string token;

        while (std::getline(inner, token, 'x')) {
            if (token.empty()) {
                MNN_ERROR("Param argv[7]: %s parse error, format should be like 1x3x512x512;1x3x256x256;1x3x512x256\n", s.c_str());
                return false;
            }
            int val = std::stoi(token);
            dims.push_back(val);
        }
        if (dims.empty()) {
            MNN_ERROR("Param argv[7]: %s parse error, format should be like 1x3x512x512;1x3x256x256;1x3x512x256\n", s.c_str());
            return false;
        }
        out.push_back(dims);
    }
    return true;
}

int main(int argc, const char* argv[]) {
    if (argc < 6) {
        MNN_PRINT("Usage: ./MNN2QNNModel src.mnn dst.mnn qnn_sdk_path qnn_model_name qnn_context_config.json\n");
        return 0;
    }
    const char* srcMNN = argv[1];
    const char* dstMNN = argv[2];
    std::string qnnSdkPath = argv[3];
    std::string qnnModelName = argv[4];
    std::string qnnContextConfig = argv[5];

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<MNN::Express::VARP> inputs;
    std::vector<MNN::Express::VARP> outputs;
    // Suggestion: using argv[6] to assign input shape for single input model.
    // Suggestion: using argv[7]/argv[8]/... to assign input shapes for multi input model.
    std::vector<std::vector<std::vector<int>>> inputShapeLists;
    bool hasInputsVarp = false;
    std::vector<std::vector<MNN::Express::VARP>> inputsVarpList;

    int totalShapeType = 1;
    if(argc > 6) {
        totalShapeType = std::stoi(argv[6]);
        std::vector<std::vector<int>> temp;
        if(parseDims(argv[7], temp)) {
            inputShapeLists.resize(totalShapeType);
            for(int i = 0; i < totalShapeType; i++) {
                // Each inputs shape in model: 128x1x896_1x1x128x128_1x128
                if(!parseDims(argv[7+i], inputShapeLists[i])) {
                    return -1;
                }
            }
        } else {
            inputsVarpList.resize(totalShapeType);
            for(int i = 0; i < totalShapeType; i++) {
                inputsVarpList[i] = MNN::Express::Variable::load(argv[7+i]);
            }
            inputs = MNN::Express::Variable::load(argv[7]);
            for (int i=0; i<inputs.size(); ++i) {
                inputNames.emplace_back(inputs[i]->name());
            }
            if(argc > 7+totalShapeType) {
                outputs = MNN::Express::Variable::load(argv[7+totalShapeType]);
                for (int i=0; i<outputs.size(); ++i) {
                    outputNames.emplace_back(outputs[i]->name());
                }
            }
            hasInputsVarp = true;
        }
    }

    /**
    generate qnn .cpp and .bin
    */
    std::string dstModelName = dstMNN;
    size_t pos = dstModelName.find_last_of("/\\");
    std::string dstModelPath;
    if (pos == std::string::npos) {
        // current path
        dstModelPath = "./";
    } else {
        dstModelPath = dstModelName.substr(0, pos);
    }
    std::string qnnModelPath = dstModelPath + "/" + qnnModelName;
    std::string totalQnnSo;
    std::vector<std::string> qnnGraphNames;
    std::vector<std::string> inputShapesStr;
    std::vector<std::vector<MNN::Express::Variable::Info>> outputInfos;

    MNN_PRINT("Total input shape type size:%d\n", totalShapeType);
    for(int index = 0; index < totalShapeType; index++)
    {
        std::string curQnnModelName = qnnModelName + std::string("_") + std::to_string(index);
        qnnGraphNames.emplace_back(curQnnModelName);
        std::string curQnnModelPath = dstModelPath + "/" + curQnnModelName;
        MNN_PRINT("[Temp Product]: Qnn temp product generate at %s\n", curQnnModelPath.c_str());
        MNNCreateDir(curQnnModelPath.c_str());
        if(index < totalShapeType-1) {
            totalQnnSo += (curQnnModelPath + std::string("/lib/x86_64-linux-clang/lib") + \
                curQnnModelName + std::string(".so,"));
        } else {
            totalQnnSo += (curQnnModelPath + std::string("/lib/x86_64-linux-clang/lib") + \
                curQnnModelName + std::string(".so "));
        }

        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_NN;
        std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
        rtmgr->setCache(curQnnModelPath.c_str());
        MNN::Express::Module::Config mConfig;
        mConfig.shapeMutable = false;
        std::shared_ptr<MNN::Express::Module> m(MNN::Express::Module::load(inputNames, outputNames, srcMNN, rtmgr, &mConfig), MNN::Express::Module::destroy);
        auto minfo = m->getInfo();
        if(outputNames.empty()) {
            outputNames = minfo->outputNames;
        }
        if(inputNames.empty()) {
            inputNames = minfo->inputNames;
        }

        if(!hasInputsVarp) {
            inputs.resize(minfo->inputs.size());
            for (int i=0; i<minfo->inputs.size(); ++i) {
                auto info = minfo->inputs[i];
                std::vector<int> inputDims = info.dim;
                if(!inputShapeLists.empty()) {
                    inputDims = inputShapeLists[index][i];
                }
                MNN_PRINT("input %d shape:", i);
                for(int d = 0; d < inputDims.size(); d++) {
                    MNN_PRINT("%d ", inputDims[d]);
                }
                MNN_PRINT("\n");
                auto varp = MNN::Express::_Input(inputDims, info.order, info.type);
                varp->writeMap<void>();
                inputs[i] = varp;
                inputs[i]->setName(inputNames[i]);
            }
        } else {
            inputs = inputsVarpList[index];
        }
        outputs = m->onForward(inputs);
        // sync
        for(int i = 0; i < outputs.size(); i++) {
            outputs[i]->readMap<void>();
        }

        int ret = 0;
        std::string tarBinCmd = "cd " + curQnnModelPath + \
            " && " + \
            "tar -cf " + curQnnModelName + ".bin *.raw";
        ret = system(tarBinCmd.c_str());
        if(ret) {
            MNN_ERROR("taf qnn raw file error!\n");
            return -1;
        }

        std::string modelLibCmd = qnnSdkPath + "/bin/x86_64-linux-clang/qnn-model-lib-generator " + \
            "-c " + curQnnModelPath + "/" + curQnnModelName + ".cpp " + \
            "-b " + curQnnModelPath + "/" + curQnnModelName + ".bin " + \
            "-t x86_64-linux-clang " + \
            "-o " + curQnnModelPath + "/lib ";
        ret = system(modelLibCmd.c_str());
        if(ret) {
            MNN_ERROR("[Error]: qnn-model-lib-generator error!\n");
            return -1;
        } else {
            MNN_PRINT("[Pass]: qnn-model-lib-generator success!\n");
        }


        std::vector<MNN::Express::Variable::Info> inputInfos(inputs.size());
        for (int i=0; i<inputInfos.size(); ++i) {
            inputInfos[i] = *inputs[i]->getInfo();
        }
        std::string inputsShapeStr = "";
        for (int i = 0; i < inputInfos.size(); i++) {
            if (i > 0) {
                inputsShapeStr += "_";
            }
            for (int j = 0; j < inputInfos[i].dim.size(); j++) {
                if (j > 0) {
                    inputsShapeStr += "x";
                }
                inputsShapeStr += std::to_string(inputInfos[i].dim[j]);
            }
        }
        inputShapesStr.emplace_back(inputsShapeStr);

        std::vector<MNN::Express::Variable::Info> outputInfo(outputs.size());
        for (int i=0; i<outputInfo.size(); ++i) {
            outputInfo[i] = *outputs[i]->getInfo();
        }
        outputInfos.emplace_back(outputInfo);
        
    }

    std::vector<MNN::Express::Variable::Info> inputInfos(inputs.size());
    for (int i=0; i<inputInfos.size(); ++i) {
        inputInfos[i] = *inputs[i]->getInfo();
    }


    // Get inputs/outputs index in mnn model
    std::vector<int> inputIndexes(inputNames.size());
    std::vector<int> outputIndexes(outputNames.size());
    {
        std::shared_ptr<MNN::Interpreter> netC(MNN::Interpreter::createFromFile(srcMNN), MNN::Interpreter::destroy);
        auto bufferPair = netC->getModelBuffer();
        auto buffer = bufferPair.first;
        auto length = bufferPair.second;
        auto net = GetNet(buffer);

        for (int i=0; i<net->tensorName()->size(); ++i) {
            auto tname = net->tensorName()->GetAsString(i)->str();
            for (int j=0; j<inputNames.size(); ++j) {
                if (tname == inputNames[j]) {
                    inputIndexes[j] = i;
                    break;
                }
            }
            for (int j=0; j<outputNames.size(); ++j) {
                if (tname == outputNames[j]) {
                    outputIndexes[j] = i;
                    break;                
                }
            }
        }
    }

    std::shared_ptr<MNN::NetT> dstNet(new NetT);

    for (int i=0; i<inputInfos.size(); ++i) {
        std::unique_ptr<OpT> input(new OpT);
        input->type = OpType_Input;
        auto param(new InputT);
        param->dims = inputInfos[i].dim;

        input->main.type = OpParameter_Input;
        input->main.value = param;
        input->name = inputNames[i];
        input->outputIndexes.push_back(i);
        dstNet->oplists.emplace_back(std::move(input));
    }

    std::string npuFile = std::string("/") + qnnModelName + std::string("_combined.bin");

    MNN_PRINT("npu model file relative path:%s\n", npuFile.c_str());
    /** Fuse to Op*/
    std::unique_ptr<MNN::OpT> op(new OpT);
    for(int i = 0; i < inputs.size(); i++) {
        op->inputIndexes.push_back(i);
    }
    for(int i = 0; i < outputs.size(); i++) {
        op->outputIndexes.push_back(inputs.size() + i);
    }
    op->name = "qnn/plugin/op";
    op->main.Reset();
    op->type = MNN::OpType_Plugin;
    op->main.type = MNN::OpParameter_Plugin;
    op->main.value = new MNN::PluginT;
    auto extra = op->main.AsPlugin();
    extra->type = "QNN";
    std::unique_ptr<MNN::AttributeT> attr(new MNN::AttributeT);

    
    dstNet->tensorName = inputNames;
    dstNet->tensorName.insert(dstNet->tensorName.end(), outputNames.begin(), outputNames.end());
    dstNet->tensorName.push_back(op->name);
    dstNet->outputName = outputNames;


    attr->key = "allInputShape";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(inputShapesStr.size());
    for(int i = 0; i < inputShapesStr.size(); i++) {
        attr->list->s[i] = inputShapesStr[i];
    }
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);

    attr->key = "allGraphName";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(qnnGraphNames.size());
    for(int i = 0; i < qnnGraphNames.size(); i++) {
        attr->list->s[i] = qnnGraphNames[i];
    }
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);

    attr->key = "path";
    attr->s = npuFile;
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);
    attr->key = "inputs";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(inputNames.size());
    for (int i=0; i<inputNames.size(); ++i) {
        // ::TODO
        attr->list->s[i] = std::string("t") + std::to_string(inputIndexes[i]);
    }
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);
    attr->key = "outputs";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(outputNames.size());
    for (int i=0; i<outputNames.size(); ++i) {
        // ::TODO
        attr->list->s[i] = std::string("t") + std::to_string(outputIndexes[i]);
    }
    extra->attr.emplace_back(std::move(attr));


    for (int i=0; i<outputInfos.size(); ++i) {
        attr.reset(new MNN::AttributeT);
        for(int j = 0; j < outputInfos[i].size(); j++) {
            attr->key = "o_" + std::to_string(i) + std::string("_") +  std::to_string(j);
            attr->tensor.reset(new BlobT);
            attr->tensor->dataType = OpCommonUtils::convertDataType(outputInfos[i][j].type);
            attr->tensor->dims = outputInfos[i][j].dim;
            switch(outputInfos[i][j].order) {
                case MNN::Express::NHWC:
                    attr->tensor->dataFormat = MNN_DATA_FORMAT_NHWC;
                    break;
                case MNN::Express::NCHW:
                    attr->tensor->dataFormat = MNN_DATA_FORMAT_NCHW;
                    break;
                case MNN::Express::NC4HW4:
                    attr->tensor->dataFormat = MNN_DATA_FORMAT_NC4HW4;
                    break;
                default:
                    attr->tensor->dataFormat = MNN_DATA_FORMAT_NCHW;
                    break;
            }
        }
        extra->attr.emplace_back(std::move(attr));
    }

    // Compile NPU Module
    std::unique_ptr<OpT> npuOp;
    npuOp = std::move(op);

    // Merge to dst
    dstNet->oplists.emplace_back(std::move(npuOp));

    // Store
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Net::Pack(builder, dstNet.get()));
    std::ofstream outputOs(dstMNN, std::ios::binary);
    outputOs.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    outputOs.close();


    auto newContextConfigPath = modifyQnnConfigFile(qnnContextConfig, qnnSdkPath, qnnGraphNames, dstModelPath);

    std::string binaryGenCmd = qnnSdkPath + "/bin/x86_64-linux-clang/qnn-context-binary-generator " + \
        "--model " + totalQnnSo + \
        "--backend " + qnnSdkPath + "/lib/x86_64-linux-clang/libQnnHtp.so " + \
        "--binary_file " + qnnModelName + "_combined " + \
        "--config_file " + newContextConfigPath + " " + \
        "--output_dir " + dstModelPath;
    auto res = system(binaryGenCmd.c_str());
    if(res) {
        MNN_ERROR("[Error]: qnn-context-binary-generator error!\n");
        return -1;
    } else {
        MNN_PRINT("[Pass]: qnn-context-binary-generator success!\n");
    }

    std::string qnnBin = dstModelPath + npuFile;

    MNN_PRINT("[All Pass]: npu model generator success!\n");
    MNN_PRINT("[Output Product]:\nNew mnn model path: %s\nNpu model path: %s\n", dstMNN, qnnBin.c_str());
    return 0;
}
