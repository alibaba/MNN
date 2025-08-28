#include <fstream>
#include <sstream>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/OpCommonUtils.hpp"
#include "MNN_generated.h"
#include <cstdlib>

using namespace MNN::Express;
using namespace MNN;
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
    if (argc > 6) {
        inputs = MNN::Express::Variable::load(argv[6]);
        for (int i=0; i<inputs.size(); ++i) {
            inputNames.emplace_back(inputs[i]->name());
        }
        auto outputs = MNN::Express::Variable::load(argv[7]);
        for (int i=0; i<outputs.size(); ++i) {
            outputNames.emplace_back(outputs[i]->name());
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
    MNN_PRINT("[Temp Product]: Qnn temp product generate at %s\n", qnnModelPath.c_str());
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_NN;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setCache(qnnModelPath.c_str());
    MNN::Express::Module::Config mConfig;
    mConfig.shapeMutable = false;
    std::shared_ptr<MNN::Express::Module> m(MNN::Express::Module::load(inputNames, outputNames, srcMNN, rtmgr, &mConfig), MNN::Express::Module::destroy);
    auto minfo = m->getInfo();
    outputNames = minfo->outputNames;
    inputNames = minfo->inputNames;
    inputs.resize(minfo->inputs.size());
    for (int i=0; i<minfo->inputs.size(); ++i) {
        auto& info = minfo->inputs[i];
        auto varp = MNN::Express::_Input(info.dim, info.order, info.type);
        varp->writeMap<void>();
        inputs[i] = varp;
        inputs[i]->setName(inputNames[i]);
    }

    auto outputs = m->onForward(inputs);
    // sync
    for(int i = 0; i < outputs.size(); i++) {
        outputs[i]->readMap<void>();
    }

    int ret = 0;
    std::string tarBinCmd = "cd " + qnnModelPath + \
        " && " + \
        "tar -cf " + qnnModelName + ".bin *.raw";
    ret = system(tarBinCmd.c_str());
    if(ret) {
        MNN_ERROR("taf qnn raw file error!\n");
        return -1;
    }

    std::string modelLibCmd = qnnSdkPath + "/bin/x86_64-linux-clang/qnn-model-lib-generator " + \
        "-c " + qnnModelPath + "/" + qnnModelName + ".cpp " + \
        "-b " + qnnModelPath + "/" + qnnModelName + ".bin " + \
        "-t x86_64-linux-clang " + \
        "-o " + qnnModelPath + "/lib ";
    ret = system(modelLibCmd.c_str());
    if(ret) {
        MNN_ERROR("[Error]: qnn-model-lib-generator error!\n");
        return -1;
    } else {
        MNN_PRINT("[Pass]: qnn-model-lib-generator success!\n");
    }

    std::string qnnBin = dstModelPath + "/" + qnnModelName + ".bin";
    std::string binaryGenCmd = qnnSdkPath + "/bin/x86_64-linux-clang/qnn-context-binary-generator " + \
        "--model " + qnnModelPath + "/lib/x86_64-linux-clang/lib" + qnnModelName + ".so " + \
        "--backend " + qnnSdkPath + "/lib/x86_64-linux-clang/libQnnHtp.so " + \
        "--binary_file " + qnnModelName + " " + \
        "--config_file " + qnnContextConfig + " " + \
        "--output_dir " + dstModelPath;
    ret = system(binaryGenCmd.c_str());
    if(ret) {
        MNN_ERROR("[Error]: qnn-context-binary-generator error!\n");
        return -1;
    } else {
        MNN_PRINT("[Pass]: qnn-context-binary-generator success!\n");
    }


    std::vector<MNN::Express::Variable::Info> inputInfos(inputs.size());
    for (int i=0; i<inputInfos.size(); ++i) {
        inputInfos[i] = *inputs[i]->getInfo();
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

    std::string npuPath = std::string("/") + qnnModelName + std::string(".bin");
 
    MNN_PRINT("npu model path:%s\n", npuPath.c_str());
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

    std::string graphName = qnnModelName;

    attr->key = "allInputShape";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(1);
    attr->list->s[0] = inputsShapeStr;
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);

    attr->key = "allGraphName";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(1);
    attr->list->s[0] = graphName;
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);

    attr->key = "path";
    attr->s = npuPath;
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);
    attr->key = "inputs";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(inputNames.size());
    for (int i=0; i<inputNames.size(); ++i) {
        // ::TODO
        attr->list->s[i] = std::string("t") + std::to_string(i);
    }
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);
    attr->key = "outputs";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(outputNames.size());
    for (int i=0; i<outputNames.size(); ++i) {
        // ::TODO
        attr->list->s[i] = std::string("t") + std::to_string(TensorUtils::getDescribe(outputs[i]->getTensor())->index);
    }
    extra->attr.emplace_back(std::move(attr));

    std::vector<MNN::Express::Variable::Info> outputInfos(outputs.size());
    for (int i=0; i<outputInfos.size(); ++i) {
        outputInfos[i] = *outputs[i]->getInfo();
    }
    for (int i=0; i<outputInfos.size(); ++i) {
        attr.reset(new MNN::AttributeT);
        attr->key = "o_0_" + std::to_string(i);
        attr->tensor.reset(new BlobT);
        attr->tensor->dataType = OpCommonUtils::convertDataType(outputInfos[i].type);
        attr->tensor->dims = outputInfos[i].dim;
        switch(outputInfos[i].order) {
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

    MNN_PRINT("[All Pass]: npu model generator success!\n");
    MNN_PRINT("[Output Product]:\nNew mnn model path: %s\nNpu model path: %s\n", dstMNN, qnnBin.c_str());
    return 0;
}
