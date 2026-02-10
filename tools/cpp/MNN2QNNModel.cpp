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
#include <sys/utsname.h>

using namespace rapidjson;

using namespace MNN::Express;
using namespace MNN;

static bool generateConfigFile(const std::string & qnnSDKPath, int socID, int dspArch, const std::vector<std::string> & graphNameVec, const std::string & outputDir, std::string & configPath, std::string & subConfigPath) {
    configPath = MNNFilePathConcat(outputDir, "context_config.json");
    subConfigPath = MNNFilePathConcat(outputDir, "htp_backend_extensions.json");

    // Write context_config.json
    rapidjson::Document contextConfigDoc;
    contextConfigDoc.SetObject();
    rapidjson::Document::AllocatorType& contextAllocator = contextConfigDoc.GetAllocator();
    rapidjson::Value backendExtensions(rapidjson::kObjectType);
    std::string htpBackendExtPath = MNNFilePathConcat(qnnSDKPath, "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so");
    backendExtensions.AddMember("shared_library_path", rapidjson::Value(htpBackendExtPath.c_str(), contextAllocator).Move(), contextAllocator);
    backendExtensions.AddMember("config_file_path", rapidjson::Value(subConfigPath.c_str(), contextAllocator).Move(), contextAllocator);
    contextConfigDoc.AddMember("backend_extensions", backendExtensions, contextAllocator);
    rapidjson::StringBuffer contextBuffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> contextWriter(contextBuffer);
    contextConfigDoc.Accept(contextWriter);
    std::ofstream contextConfigOut(configPath);
    contextConfigOut << contextBuffer.GetString();
    contextConfigOut.close();

    // Write htp_backend_extensions.json
    rapidjson::Document htpConfigDoc;
    htpConfigDoc.SetObject();
    rapidjson::Document::AllocatorType& htpConfigAllocator = htpConfigDoc.GetAllocator();

    // "graphs" section
    rapidjson::Value graphs(rapidjson::kArrayType);
    rapidjson::Value graphObj(rapidjson::kObjectType);
    graphObj.AddMember("vtcm_mb", 8, htpConfigAllocator);
    rapidjson::Value names(rapidjson::kArrayType);
    for (const auto& name : graphNameVec) {
        names.PushBack(rapidjson::Value(name.c_str(), contextAllocator).Move(), htpConfigAllocator);
    }
    graphObj.AddMember("graph_names", names, htpConfigAllocator);
    graphObj.AddMember("O", 3.0, htpConfigAllocator);
    graphObj.AddMember("fp16_relaxed_precision", 1, htpConfigAllocator);
    graphObj.AddMember("weights_packing", true, htpConfigAllocator);
    graphObj.AddMember("hvx_threads", 4, htpConfigAllocator);
    graphs.PushBack(graphObj, htpConfigAllocator);
    htpConfigDoc.AddMember("graphs", graphs, htpConfigAllocator);

    // "devices" section
    rapidjson::Value devices(rapidjson::kArrayType);
    rapidjson::Value deviceObj(rapidjson::kObjectType);
    deviceObj.AddMember("soc_id", socID, htpConfigAllocator);
    std::string hexagonArchStr = "v" + std::to_string(dspArch);
    deviceObj.AddMember("dsp_arch", rapidjson::Value(hexagonArchStr.c_str(), contextAllocator).Move(), htpConfigAllocator);
    rapidjson::Value cores(rapidjson::kArrayType);
    rapidjson::Value coreObj(rapidjson::kObjectType);
    coreObj.AddMember("core_id", 0, htpConfigAllocator);
    coreObj.AddMember("perf_profile", "burst", htpConfigAllocator);
    coreObj.AddMember("rpc_control_latency", 100, htpConfigAllocator);
    cores.PushBack(coreObj, htpConfigAllocator);
    deviceObj.AddMember("cores", cores, htpConfigAllocator);
    devices.PushBack(deviceObj, htpConfigAllocator);
    htpConfigDoc.AddMember("devices", devices, htpConfigAllocator);

    // "context" section
    rapidjson::Value contextObj(rapidjson::kObjectType);
    contextObj.AddMember("weight_sharing_enabled", true, htpConfigAllocator);
    htpConfigDoc.AddMember("context", contextObj, htpConfigAllocator);

    rapidjson::StringBuffer htpConfigBuffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> htpConfigWriter(htpConfigBuffer);
    htpConfigDoc.Accept(htpConfigWriter);
    std::ofstream htpConfigOut(subConfigPath);
    htpConfigOut << htpConfigBuffer.GetString();
    htpConfigOut.close();
    return true;
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
            MNN_ERROR("%s parse error, format should be like 1x3x512x512_1x256\n", s.c_str());
            return false;
        }
        std::vector<int> dims;
        std::stringstream inner(segment);
        std::string token;

        while (std::getline(inner, token, 'x')) {
            if (token.empty()) {
                MNN_ERROR("%s parse error, format should be like 1x3x512x512_1x256\n", s.c_str());
                return false;
            }
            int val = std::stoi(token);
            dims.push_back(val);
        }
        if (dims.empty()) {
            MNN_ERROR("%s parse error, format should be like 1x3x512x512_1x256\n", s.c_str());
            return false;
        }
        out.push_back(dims);
    }
    return true;
}

static bool checkSystem() {
    struct utsname buf;
    if (uname(&buf) != 0) {
        MNN_ERROR("uname error\n");
        return false;
    }
    if (std::string(buf.sysname) == "Linux" && std::string(buf.machine) == "x86_64") {
        return true;
    }
    MNN_ERROR("This program must be run on a x86_64 Linux system. Current system: %s %s\n", buf.sysname, buf.machine);
    return false;
}

int main(int argc, const char* argv[]) {
    if (argc < 6) {
        MNN_PRINT("This tool generates offline caches for the QNN backend.");
        MNN_PRINT("Usage: %s <qnnSDKPath> <socId> <hexagonArch> <srcMNNPath> <outputDir> [totalShapeNum] [inputShape1] [inputShape2] ...\n", argv[0]);
        MNN_PRINT("    <qnnSDKPath>      : Path to the QNN SDK directory.\n");
        MNN_PRINT("    <socId>           : Target SoC ID.\n");
        MNN_PRINT("                        Common SoCs: 8Gen2 -> 43, 8Gen3 -> 57, 8 Elite -> 69. For others, please refer to Qualcomm's documentation.\n");
        MNN_PRINT("    <hexagonArch>     : Hexagon architecture version. This tool requires v73 or higher for weight sharing.\n");
        MNN_PRINT("                        Common Archs: 8Gen2 -> 73, 8Gen3 -> 75, 8 Elite -> 79. For others, please refer to Qualcomm's documentation.\n");
        MNN_PRINT("    <srcMNNPath>      : Path to the source MNN model file.\n");
        MNN_PRINT("    <outputDir>       : Directory to save the generated files, including a MNN model file with the suffix '.mnn' and a QNN serialized artifact with the suffix '.bin'.\n");
        MNN_PRINT("    [<totalShapeNum>] : Optional. Number of dynamic input shape configurations.\n");
        MNN_PRINT("    [<inputShapeN>]   : Optional. Input shape configuration. Can be a shape string or a path to a .mnn file.\n");
        MNN_PRINT("                     Shape string format for multiple inputs: dim1xdim2_dim3xdim4. Example: 1x3x512x512_1x256\n");
        MNN_PRINT("Examples:\n");
        MNN_PRINT("  1. Use default shape from the MNN model:\n");
        MNN_PRINT("     %s /path/to/qnn/sdk 57 75 /path/to/model.mnn /path/to/output\n", argv[0]);
        MNN_PRINT("  2. Specify two dynamic input shapes:\n");
        MNN_PRINT("     %s /path/to/qnn/sdk 57 75 /path/to/model.mnn /path/to/output 2 1x3x512x512_1x256 1x3x256x256_1x128\n", argv[0]);
        MNN_PRINT("     %s /path/to/qnn/sdk 57 75 /path/to/model.mnn /path/to/output 2 input_0.mnn input_1.mnn\n", argv[0]);

        return 1;
    }

    if (!checkSystem()) {
        return -1;
    }

    std::string qnnSdkPath = argv[1];
    int socId = std::stoi(std::string(argv[2]));
    int hexagonArch = std::stoi(std::string(argv[3]));
    const char* srcMNNPath = argv[4];
    std::string modelBaseName = [](const std::string& path) {
        std::string filename = path;
        auto pos = path.find_last_of("/\\");
        if (pos != std::string::npos) {
            filename = path.substr(pos + 1);
        }
        pos = filename.find_last_of('.');
        if (pos != std::string::npos) {
            return filename.substr(0, pos);
        }
        return filename;
    }(srcMNNPath);
    std::string modelSignature = "_" + std::to_string(socId) + "_" + std::to_string(hexagonArch);
    std::string outputDir = argv[5];
    std::string dstMNNPath = MNNFilePathConcat(outputDir, modelBaseName + modelSignature + ".mnn");

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<MNN::Express::VARP> inputs;
    std::vector<MNN::Express::VARP> outputs;
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
                // Each inputs shape in model: 128x1x897_1x1x128x128_1x128
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
    std::string totalQnnSo;
    std::vector<std::string> qnnGraphNames;
    std::vector<std::vector<MNN::Express::Variable::Info>> outputInfos;
    std::vector<std::string> qnnModelDirs;
    std::vector<int> allInputShape;

    MNN_PRINT("Total input shape type size:%d\n", totalShapeType);
    for(int index = 0; index < totalShapeType; index++)
    {
        std::string curQnnModelName = modelBaseName + std::string("_") + std::to_string(index);
        qnnGraphNames.emplace_back(curQnnModelName);
        std::string curQnnModelDir = MNNFilePathConcat(outputDir, curQnnModelName);
        MNN_PRINT("[Temp Product]: Qnn temp product generate at %s\n", curQnnModelDir.c_str());
        MNNCreateDir(curQnnModelDir.c_str());
        qnnModelDirs.push_back(curQnnModelDir);
        if(index < totalShapeType-1) {
            totalQnnSo += (curQnnModelDir + std::string("/lib/x86_64-linux-clang/lib") + \
                curQnnModelName + std::string(".so,"));
        } else {
            totalQnnSo += (curQnnModelDir + std::string("/lib/x86_64-linux-clang/lib") + \
                curQnnModelName + std::string(".so "));
        }

        MNN::ScheduleConfig config;
        config.type = MNN_CONVERT_QNN;
        std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
        rtmgr->setCache(curQnnModelDir.c_str());
        MNN::Express::Module::Config mConfig;
        mConfig.shapeMutable = false;
        std::shared_ptr<MNN::Express::Module> m(MNN::Express::Module::load(inputNames, outputNames, srcMNNPath, rtmgr, &mConfig), MNN::Express::Module::destroy);
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

        // tar weight
        std::string binPath = MNNFilePathConcat(curQnnModelDir, curQnnModelName + ".bin");
        std::string command = "tar -cf " + binPath + " -C " + curQnnModelDir + " $(find " + curQnnModelDir + " -maxdepth 1 -name '*.raw' -printf '%f ') && rm " + curQnnModelDir + "/*.raw";
        int ret = std::system(command.c_str());
        if (ret != 0) {
            MNN_ERROR("Failed to execute command: %s\n", command.c_str());
        }

        std::string modelLibCmd = qnnSdkPath + "/bin/x86_64-linux-clang/qnn-model-lib-generator" + \
            " -c " + MNNFilePathConcat(curQnnModelDir, curQnnModelName + ".cpp") + \
            " -b " + binPath + \
            " -t x86_64-linux-clang " + \
            " -o " + curQnnModelDir + "/lib";
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
        std::vector<int> currInputShape;
        for (int i = 0; i < inputInfos.size(); i++) {
            for (int j = 0; j < inputInfos[i].dim.size(); j++) {
                currInputShape.emplace_back(inputInfos[i].dim[j]);
            }
        }
        allInputShape.insert(allInputShape.end(), currInputShape.begin(), currInputShape.end());

        std::vector<MNN::Express::Variable::Info> outputInfo(outputs.size());
        for (int i=0; i<outputInfo.size(); ++i) {
            outputInfo[i] = *outputs[i]->getInfo();
        }
        outputInfos.emplace_back(outputInfo);
        
    }

    std::string npuArtifactName = modelBaseName + modelSignature + ".bin";
    std::string npuArtifactPath = MNNFilePathConcat(outputDir, npuArtifactName);
    {
        std::string configPath, subConfigPath;
        if (!generateConfigFile(qnnSdkPath, socId, hexagonArch, qnnGraphNames, outputDir, configPath, subConfigPath)) {
            MNN_ERROR("[Error]: Failed to generate the config file!\n");
            return -1;
        }

        std::string binaryGenCmd = qnnSdkPath + "/bin/x86_64-linux-clang/qnn-context-binary-generator" + \
            " --model " + totalQnnSo + \
            " --backend " + qnnSdkPath + "/lib/x86_64-linux-clang/libQnnHtp.so" + \
            " --binary_file " + modelBaseName + modelSignature + \
            " --config_file " + configPath + " " + \
            " --output_dir " + outputDir;
        auto res = system(binaryGenCmd.c_str());
        if(res) {
            MNN_ERROR("[Error]: qnn-context-binary-generator error!\n");
            return -1;
        } else {
            MNN_PRINT("[Pass]: qnn-context-binary-generator success!\n");
        }

        // Remove intermediate files
        MNNRemoveFile(configPath.c_str());
        MNNRemoveFile(subConfigPath.c_str());
        for (const auto& dir : qnnModelDirs) {
            std::string cmd = "rm -rf " + dir;
            int ret = system(cmd.c_str());
            if (ret != 0) {
                MNN_PRINT("[Warning]: failed to remove temp dir: %s\n", dir.c_str());
            }
        }
    }

    std::vector<MNN::Express::Variable::Info> inputInfos(inputs.size());
    for (int i=0; i<inputInfos.size(); ++i) {
        inputInfos[i] = *inputs[i]->getInfo();
    }


    // Get inputs/outputs index in mnn model
    std::vector<int> inputIndexes(inputNames.size());
    std::vector<int> outputIndexes(outputNames.size());
    {
        std::shared_ptr<MNN::Interpreter> netC(MNN::Interpreter::createFromFile(srcMNNPath), MNN::Interpreter::destroy);
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
    attr->list->i.insert(attr->list->i.end(), allInputShape.begin(), allInputShape.end());
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
    attr->s = npuArtifactName;
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);

    attr->key = "offset";
    attr->list.reset(new MNN::ListValueT);
    attr->list->i.push_back(0);
    attr->list->i.push_back(0);
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);

    file_t binaryFile = MNNOpenFile(npuArtifactPath.c_str(), MNN_FILE_READ);
    size_t binarySize = MNNGetFileSize(binaryFile);
    MNNCloseFile(binaryFile);
    attr->key = "size";
    attr->list.reset(new MNN::ListValueT);
    uint32_t lowSrc = binarySize & 0xFFFFFFFF;
    uint32_t highSrc = binarySize >> 32;
    int32_t lowDst, highDst;
    ::memcpy(&lowDst, &lowSrc, sizeof(int32_t));
    ::memcpy(&highDst, &highSrc, sizeof(int32_t));
    attr->list->i.push_back(lowDst);
    attr->list->i.push_back(highDst);
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
    std::ofstream outputOs(dstMNNPath.c_str(), std::ios::binary);
    outputOs.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    outputOs.close();

    MNN_PRINT("[All passed]\n");
    return 0;
}
