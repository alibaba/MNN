//
//  cli.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"
#include "commonKit.hpp"
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif
#include "OpCount.hpp"
#include "cxxopts.hpp"
#include "config.hpp"
#include "logkit.h"
#include <MNN/MNNDefine.h>
#include "MNN_generated.h"
#include "PostConverter.hpp"
#include "addBizCode.hpp"
#include "caffeConverter.hpp"
#include "liteConverter.hpp"
#include "onnxConverter.hpp"
#include "tensorflowConverter.hpp"
#include "torchConverter.hpp"
#include "writeFb.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "PostConverter.hpp"
#include "rapidjson/document.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include "core/MemoryFormater.h"

namespace MNN {
using namespace MNN::Express;
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
static std::string _getFormatString(MNN::Express::Dimensionformat format) {
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
static int dumpModelInfo(const char* modelName) {
    std::vector<std::string> empty;
    std::shared_ptr<MNN::Express::Module> module(MNN::Express::Module::load(empty, empty, modelName));
    if (nullptr == module.get()) {
        MNN_ERROR("Load MNN from %s Failed\n", modelName);
        return 1;
    }
    auto info = module->getInfo();
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
    return 0;
}

bool Cli::initializeMNNConvertArgs(modelConfig &modelPath, int argc, char **argv) {
    cxxopts::Options options("MNNConvert");

    options.positional_help("[optional args]").show_positional_help();

    options.allow_unrecognised_options().add_options()(std::make_pair("h", "help"), "Convert Other Model Format To MNN Model\n")(
                                                                                                                                 std::make_pair("v", "version"), "show current version")
    (std::make_pair("f", "framework"),
#ifdef MNN_BUILD_TORCH
     "model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN,TORCH,JSON]",
#else
     "model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN,JSON]",
#endif
     cxxopts::value<std::string>())
    (
     "modelFile",
     "tensorflow Pb or caffeModel, ex: *.pb,*caffemodel",
     cxxopts::value<std::string>()
     )
    (
     "batch",
     "if model input's batch is not set, set as the batch size you set",
     cxxopts::value<int>()
     )
    (
     "keepInputFormat",
     "keep input dimension format or not, default: true",
     cxxopts::value<bool>()
     )
    (
     "optimizeLevel",
     "graph optimize option, 0: don't run optimize(only support for MNN source), 1: use graph optimize only for every input case is right, 2: normally right but some case may be wrong, default 1",
     cxxopts::value<int>()
     )
    (
     "optimizePrefer",
     "graph optimize option, 0 for normal, 1 for smalleset, 2 for fastest",
     cxxopts::value<int>()
     )
    (
     "prototxt",
     "only used for caffe, ex: *.prototxt",
     cxxopts::value<std::string>())
    (
     "MNNModel",
     "MNN model, ex: *.mnn",
     cxxopts::value<std::string>())
    (
     "fp16",
     "save Conv's weight/bias in half_float data type")
    (
     "benchmarkModel",
     "Do NOT save big size data, such as Conv's weight,BN's gamma,beta,mean and variance etc. Only used to test the cost of the model")
    (
     "bizCode",
     "MNN Model Flag, ex: MNN",
     cxxopts::value<std::string>())
    (
     "debug",
     "Enable debugging mode."
     )
    (
     "forTraining",
     "whether or not to save training ops BN and Dropout, default: false",
     cxxopts::value<bool>()
     )
    (
     "weightQuantBits",
     "save conv/matmul/LSTM float weights to int8 type, only optimize for model size, 2-8 bits, default: 0, which means no weight quant",
     cxxopts::value<int>()
     )
    (
     "weightQuantAsymmetric",
     "the default weight-quant uses SYMMETRIC quant method, which is compatible with old MNN versions. "
     "you can try set --weightQuantAsymmetric to use asymmetric quant method to improve accuracy of the weight-quant model in some cases, "
     "but asymmetric quant model cannot run on old MNN versions. You will need to upgrade MNN to new version to solve this problem. default: false",
     cxxopts::value<bool>()
     )
    (
     "weightQuantBlock",
     "using block-wise weight quant, set block size, defaut: -1, which means channel-wise weight quant",
     cxxopts::value<int>()
     )
    (
     "compressionParamsFile",
     "The path of the compression parameters that stores activation, "
     "weight scales and zero points for quantization or information "
     "for sparsity.",
     cxxopts::value<std::string>()
     )
    (
     "OP",
     "print framework supported op",
     cxxopts::value<bool>()
     )
    (
     "saveStaticModel",
     "save static model with fix shape, default: false",
     cxxopts::value<bool>()
     )
    (
     "targetVersion",
     "compability for old mnn engine, default the same as converter",
     cxxopts::value<float>()
     )
    (
     "customOpLibs",
     "custom op libs ex: libmy_add.so;libmy_sub.so",
     cxxopts::value<std::string>()
     )
    (
     "info",
     "dump MNN's model info"
     )
    (
     "authCode",
     "code for model authentication.",
     cxxopts::value<std::string>()
     )
    (
     "inputConfigFile",
     "set input config file for static model, ex: ~/config.txt",
     cxxopts::value<std::string>()
     )
    (
     "testdir",
     "set test dir, mnn will convert model and then check the result",
     cxxopts::value<std::string>()
     )
    (
     "testconfig",
     "set test config json, example: tools/converter/forward.json",
     cxxopts::value<std::string>()
     )
    (
     "thredhold",
     "if set test dir, thredhold mean the max rate permit for run MNN model and origin error",
     cxxopts::value<float>()
     )
    (
     "JsonFile",
     "if input model is MNN and give jsonfile, while Dump MNN model to the JsonFile.",
     cxxopts::value<std::string>()
     )
    (
     "alignDenormalizedValue",
     "if 1, converter would align denormalized float(|x| < 1.18e-38) as zero, because of in ubuntu/protobuf or android/flatbuf, system behaviors are different. default: 1, range: {0, 1}",
     cxxopts::value<int>()
     )
    (
     "detectSparseSpeedUp",
     "if 1 converter would detect weights sparsity and check sparse speedup. default: 1, range : {0, 1}",
     cxxopts::value<int>()
     )
    (
     "saveExternalData",
     "save weight to extenal bin file.",
     cxxopts::value<bool>()
     )
    (
     "convertMatmulToConv",
     "if 1, converter matmul with constant input to convolution. default: 1, range: {0, 1}",
     cxxopts::value<int>()
     )
    (
     "transformerFuse",
     "fuse attention op, like fmhaV2/fmhca/splitGelu/groupNorm. default: false",
     cxxopts::value<bool>()
     );

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help({""}) << std::endl;
        return false;
    }

    if (result.count("version")) {
        std::cout << MNN_VERSION << std::endl;
        return false;
    }

    modelPath.model = modelPath.MAX_SOURCE;
    // model source
    std::string frameWork;
    if (result.count("framework")) {
        frameWork = result["framework"].as<std::string>();
        if ("TF" == frameWork) {
            modelPath.model = modelConfig::TENSORFLOW;
        } else if ("CAFFE" == frameWork) {
            modelPath.model = modelConfig::CAFFE;
        } else if ("ONNX" == frameWork) {
            modelPath.model = modelConfig::ONNX;
        } else if ("MNN" == frameWork) {
            modelPath.model = modelConfig::MNN;
        } else if ("TFLITE" == frameWork) {
            modelPath.model = modelConfig::TFLITE;
#ifdef MNN_BUILD_TORCH
        } else if ("TORCH" == frameWork) {
            modelPath.model = modelConfig::TORCH;
#endif
        } else if ("JSON" == frameWork) {
            modelPath.model = modelConfig::JSON;
        } else {
            std::cout << "Framework Input ERROR or Not Support This Model Type Now!" << std::endl;
            return false;
        }
    } else {
        std::cout << options.help({""}) << std::endl;
        DLOG(INFO) << "framework Invalid, use -f CAFFE/MNN/ONNX/TFLITE/TORCH/JSON !";
        return false;
    }
    if (result.count("OP")) {
        MNN_PRINT("Dump %s support Ops\n", frameWork.c_str());
        const auto& res = OpCount::get()->getMap().find(frameWork);
        if (res == OpCount::get()->getMap().end()) {
            return false;
        }
        for (const auto& iter : res->second) {
            MNN_PRINT("%s\n", iter.c_str());
        }
        MNN_PRINT("Total: %d\n", (int)res->second.size());
        return false;
    }

    // model file path
    if (result.count("modelFile")) {
        const std::string modelFile = result["modelFile"].as<std::string>();
        if (CommonKit::FileIsExist(modelFile)) {
            modelPath.modelFile = modelFile;
        } else {
            DLOG(INFO) << "Model File Does Not Exist! ==> " << modelFile;
            return false;
        }
    } else {
        DLOG(INFO) << "modelFile Not set Invalid, use --modelFile to set!";
        return false;
    }
    // Optimize Level
    if (result.count("optimizeLevel")) {
        modelPath.optimizeLevel = result["optimizeLevel"].as<int>();
        if (modelPath.optimizeLevel > 1) {
            DLOG(INFO) << "\n optimizeLevel > 1, some case may be wrong";
        }
    }

    // prototxt file path
    if (result.count("prototxt")) {
        const std::string prototxt = result["prototxt"].as<std::string>();
        if (CommonKit::FileIsExist(prototxt)) {
            modelPath.prototxtFile = prototxt;
        } else {
            DLOG(INFO) << "Proto File Does Not Exist!";
            return false;
        }
    } else {
        // caffe model must have this option
        if (modelPath.model == modelPath.CAFFE) {
            DLOG(INFO) << "Proto File Not Set, use --prototxt XXX.prototxt to set it!";
            return false;
        }
    }

    // MNN model output path
    if (result.count("MNNModel")) {
        const std::string MNNModelPath = result["MNNModel"].as<std::string>();
        modelPath.MNNModel             = MNNModelPath;
    } else if (result.count("JsonFile")) {
        const std::string JsonFilePath = result["JsonFile"].as<std::string>();
        modelPath.mnn2json             = true;
        modelPath.MNNModel             = JsonFilePath;
    } else if (result.count("info") && modelPath.model == modelConfig::MNN) {
        modelPath.dumpInfo = true;
        return true;
    } else {
        DLOG(INFO) << "MNNModel File Not Set, use --MNNModel XXX.prototxt to set it!";
        return false;
    }
    if (result.count("targetVersion")) {
        auto version = result["targetVersion"].as<float>();
        std::cout << "TargetVersion is " << version << std::endl;
        modelPath.targetVersion = version;
    }
    // add MNN bizCode
    if (result.count("bizCode")) {
        const std::string bizCode = result["bizCode"].as<std::string>();
        modelPath.bizCode         = bizCode;
    } else {
        MNN_ERROR("Don't has bizCode, use MNNTest for default\n");
        modelPath.bizCode = "MNNTest";
    }

    // input config file path
    if (result.count("inputConfigFile")) {
        const std::string inputConfigFile = result["inputConfigFile"].as<std::string>();
        modelPath.inputConfigFile         = inputConfigFile;
    }

    // half float
    if (result.count("fp16")) {
        modelPath.saveHalfFloat = true;
    }
    if (result.count("forTraining")) {
        modelPath.forTraining = true;
    }
    if (result.count("batch")) {
        modelPath.defaultBatchSize = result["batch"].as<int>();
    }
    if (result.count("keepInputFormat")) {
        modelPath.keepInputFormat = result["keepInputFormat"].as<bool>();
    }
    if (result.count("weightQuantBits")) {
        modelPath.weightQuantBits = result["weightQuantBits"].as<int>();
    }
    if (result.count("weightQuantAsymmetric")) {
        modelPath.weightQuantAsymmetric = result["weightQuantAsymmetric"].as<bool>();
    }
    if (result.count("weightQuantBlock")) {
        modelPath.weightQuantBlock = result["weightQuantBlock"].as<int>();
    }
    if (result.count("saveStaticModel")) {
        modelPath.saveStaticModel = true;
    }
    if (result.count("optimizePrefer")) {
        modelPath.optimizePrefer = result["optimizePrefer"].as<int>();
    }
    // Int8 calibration table path.
    if (result.count("compressionParamsFile")) {
        modelPath.compressionParamsFile =
        result["compressionParamsFile"].as<std::string>();
    }
    if (result.count("customOpLibs")) {
        modelPath.customOpLibs = result["customOpLibs"].as<std::string>();
    }
    if (result.count("authCode")) {
        modelPath.authCode = result["authCode"].as<std::string>();
    }
    if (result.count("alignDenormalizedValue")) {
        modelPath.alignDenormalizedValue = result["alignDenormalizedValue"].as<int>();
    }
    if (result.count("detectSparseSpeedUp")) {
        modelPath.detectSparseSpeedUp = result["detectSparseSpeedUp"].as<int>();
    }
    if (result.count("convertMatmulToConv")) {
        modelPath.convertMatmulToConv = result["convertMatmulToConv"].as<int>();
    }

    if (result.count("testdir")) {
        modelPath.testDir = result["testdir"].as<std::string>();
    }
    if (result.count("testconfig")) {
        modelPath.testConfig = result["testconfig"].as<std::string>();
    }
    if (result.count("thredhold")) {
        modelPath.testThredhold = result["thredhold"].as<float>();
    }
    if (result.count("saveExternalData")) {
        modelPath.saveExternalData = true;
    }
    if (result.count("transformerFuse")) {
        modelPath.transformerFuse = true;
    }
    return true;
}

typedef VARP (*unaryProc)(VARP input);
static unaryProc selectUnaryProc(int type) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return MNN::Express::_Abs;
        case UnaryOpOperation_SQUARE:
            return MNN::Express::_Square;
        case UnaryOpOperation_NEG:
            return MNN::Express::_Negative;
        case UnaryOpOperation_RSQRT:
            return MNN::Express::_Rsqrt;
        case UnaryOpOperation_EXP:
            return MNN::Express::_Exp;
        case UnaryOpOperation_COS:
            return MNN::Express::_Cos;
        case UnaryOpOperation_SIN:
            return MNN::Express::_Sin;
        case UnaryOpOperation_SIGMOID:
            return MNN::Express::_Sigmoid;
        case UnaryOpOperation_TANH:
            return MNN::Express::_Tanh;
        case UnaryOpOperation_TAN:
            return MNN::Express::_Tan;
        case UnaryOpOperation_ATAN:
            return MNN::Express::_Atan;
        case UnaryOpOperation_SQRT:
            return MNN::Express::_Sqrt;
        case UnaryOpOperation_RECIPROCAL:
            return MNN::Express::_Reciprocal;
        case UnaryOpOperation_LOG1P:
            return MNN::Express::_Log1p;
        case UnaryOpOperation_LOG:
            return MNN::Express::_Log;
        case UnaryOpOperation_ACOSH:
            return MNN::Express::_Acosh;
        case UnaryOpOperation_SINH:
            return MNN::Express::_Sinh;
        case UnaryOpOperation_ASINH:
            return MNN::Express::_Asinh;
        case UnaryOpOperation_ATANH:
            return MNN::Express::_Atanh;
        case UnaryOpOperation_SIGN:
            return MNN::Express::_Sign;
        case UnaryOpOperation_COSH:
            return MNN::Express::_Cosh;
        case UnaryOpOperation_ERF:
            return MNN::Express::_Erf;
        case UnaryOpOperation_ERFC:
            return MNN::Express::_Erfc;
        case UnaryOpOperation_ERFINV:
            return MNN::Express::_Erfinv;
        case UnaryOpOperation_EXPM1:
            return MNN::Express::_Expm1;
        case UnaryOpOperation_ASIN:
            return MNN::Express::_Asin;
        case UnaryOpOperation_ACOS:
            return MNN::Express::_Acos;
        case UnaryOpOperation_HARDSWISH:
            return MNN::Express::_Hardswish;
        case UnaryOpOperation_GELU:
            return MNN::Express::_Gelu;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}
static void computeUnaryBuffer(MNN::NetT* net) {
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); ++iter) {
        auto op = iter->get();
        auto opType = op->type;
        std::map<int, TensorDescribeT*> describes;
        for (auto& des : net->extraTensorDescribe) {
            describes.insert(std::make_pair(des->index, des.get()));
        }
        if (opType == MNN::OpType_Sigmoid || opType == MNN::OpType_TanH) {
            op->type = OpType_UnaryOp;
            op->main.value = new UnaryOpT;
            op->main.type = OpParameter_UnaryOp;
            op->main.AsUnaryOp()->opType = UnaryOpOperation_SIGMOID;
            if (opType == MNN::OpType_TanH) {
                op->main.AsUnaryOp()->opType = UnaryOpOperation_TANH;
            }
            opType = op->type;
        }
        if (opType == MNN::OpType_UnaryOp) {
            auto type = op->main.AsUnaryOp()->opType;
            if (type == UnaryOpOperation_ABS || type == UnaryOpOperation_NEG || type == UnaryOpOperation_SIGN) {
                continue;
            }
            op->main.AsUnaryOp()->tableInt8.resize(255);
            auto unaryParam = op->main.AsUnaryOp()->tableInt8.data();

            auto outputId = op->outputIndexes[0];
            if (describes.find(outputId) == describes.end()) {
                continue;
            }
            auto unaryDes = describes.find(outputId)->second;
            float outScale = unaryDes->quantInfo->scale;
            float outZero  = unaryDes->quantInfo->zero;
            auto inputId = op->inputIndexes[0];
            if (describes.find(inputId) == describes.end()) {
                auto iter = describes.find(outputId);
                
            }
            unaryDes = describes.find(inputId)->second;
            float inpScale = unaryDes->quantInfo->scale;
            float inpZero  = unaryDes->quantInfo->zero;

            // Read input data.
            std::vector<float> dataInput;
            float fx = 0.f;
            auto input = _Input({255}, NCHW, halide_type_of<float>());
            input->setName("input_tensor");
            auto ptr_in = input->template writeMap<float>();
            for (int i = -127; i <= 127; ++i) {
                fx = (i - inpZero) * inpScale;
                dataInput.push_back(fx);
                ptr_in[i + 127] = fx;
            }
            input->unMap();
            // Compute output data.
            VARP output;
            auto func = selectUnaryProc(type);
            if (nullptr == func) {
                MNN_ERROR("Don't support quantizing UnaryOP: %s to Int8\n", op->name.c_str());
            }
            output = func(input);
            auto gotOutput = output->template readMap<float>();
            // Write output data.
            int val;
            for (int i = 0; i < 255; ++i) {
                val = (int)roundf(gotOutput[i] / outScale) + outZero;
                if (val > 127) {
                    val = 127;
                }
                if (val < -127) {
                    val = -127;
                }
                unaryParam[i] = val;
                            }
        }
    }
}

bool Cli::convertModel(modelConfig& modelPath) {
    if (modelPath.dumpInfo) {
        dumpModelInfo(modelPath.modelFile.c_str());
        return true;
    }
    std::cout << "Start to Convert Other Model Format To MNN Model..., target version: " << modelPath.targetVersion << std::endl;
    std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
    int parseRes = 1;
    if (modelPath.model == modelConfig::CAFFE) {
        parseRes = caffe2MNNNet(modelPath.prototxtFile, modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::TENSORFLOW) {
        parseRes = tensorflow2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::MNN) {
        if (modelPath.mnn2json) {
            if (mnn2json(modelPath.modelFile.c_str(), modelPath.MNNModel.c_str())) {
                MNN_PRINT("MNNModel %s has convert to JsonFile %s.\n", modelPath.modelFile.c_str(), modelPath.MNNModel.c_str());
                return true;
            } else {
                MNN_ERROR("[ERROR] MNN to Json failed.\n");
                return false;
            }
        } else {
            parseRes = addBizCode(modelPath.modelFile, modelPath.bizCode, netT);
        }
    } else if (modelPath.model == modelConfig::ONNX) {
        parseRes = onnx2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::TFLITE) {
        parseRes = tflite2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
#ifdef MNN_BUILD_TORCH
    } else if (modelPath.model == modelConfig::TORCH) {
        parseRes = torch2MNNNet(modelPath.modelFile, modelPath.bizCode, netT, modelPath.customOpLibs);
#endif
    } else if (modelPath.model == modelConfig::JSON) {
        if (json2mnn(modelPath.modelFile.c_str(), modelPath.MNNModel.c_str())) {
            MNN_PRINT("JsonFile %s has convert to MNNModel %s.\n", modelPath.modelFile.c_str(), modelPath.MNNModel.c_str());
            return true;
        } else {
            MNN_ERROR("[ERROR] Json to MNN failed.\n");
            return false;
        }
    } else {
        MNN_ERROR("[ERROR] Not Support Model Type.\n");
    }
    if (netT.get() == nullptr || parseRes) {
        MNN_ERROR("[ERROR] Convert error, please check your file format.\n");
        return false;
    }
    int error = 0;
    if (modelPath.defaultBatchSize > 0) {
        for (const auto& op : netT->oplists) {
            if (op->type != OpType_Input || nullptr == op->main.AsInput()) {
                continue;
            }
            auto inputP = op->main.AsInput();
            if (inputP->dims.size() >= 1 && inputP->dims[0] <= 0) {
                std::cout << "Set " << op->name << " batch = " << modelPath.defaultBatchSize << std::endl;
                inputP->dims[0] = modelPath.defaultBatchSize;
            }
        }
    }
    if (modelPath.model != modelConfig::MNN || modelPath.optimizeLevel >= 2) {
        std::cout << "Start to Optimize the MNN Net..." << std::endl;
        std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining, modelPath);
        if (newNet->extraTensorDescribe.size()>0) {
            MNN_PRINT("MNN net has tensor quant info\n");
            computeUnaryBuffer(newNet.get());
        }
        
        error = writeFb(newNet, modelPath.MNNModel, modelPath);
    } else {
        error = writeFb(netT, modelPath.MNNModel, modelPath);
    }
    if (0 == error) {
        std::cout << "Converted Success!" << std::endl;
    } else {
        std::cout << "Converted Failed!" << std::endl;
    }
    if (modelPath.testDir.size() > 0) {
        std::cout << "Check convert result by " << modelPath.testDir << ", thredhold is " << modelPath.testThredhold << std::endl;
        Cli::testconvert(modelPath.MNNModel, modelPath.testDir, modelPath.testThredhold, modelPath.testConfig);
    }
    return true;
}

static bool compareOutput(MNN::Express::VARP output, const std::string& directName, const std::string& name, MNN::Express::Dimensionformat dataFormat, int order, float maxError) {
    auto info = output->getInfo();
    auto ptr = output->readMap<float>();
    if (info && info->size <= 0) {
        MNN_PRINT("skip checking value for zero content tensor %s\n", name.c_str());
        return true;
    }

    if (nullptr == info || nullptr == ptr) {
        MNN_ERROR("TESTERROR name:%s, info:%p, ptr:%p.\n", name.c_str(), info, ptr);
        return false;
    }
    std::ifstream outputOrigin;
    // First find key
    {
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << name <<".txt";
        outputOrigin.open(outputFileOs.str().c_str());
    }
    // Second find order
    if (outputOrigin.fail()) {
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << order <<".txt";
        outputOrigin.open(outputFileOs.str().c_str());
    }
    if (outputOrigin.fail()) {
        MNN_PRINT("Skip check %s\n", name.c_str());
        return true;
    }
    if (info->order == MNN::Express::NC4HW4 && info->dim.size() > 1) {
        output = _Convert(output, dataFormat);
        info = output->getInfo();
    }
    if (info->type.code != halide_type_float) {
        output = MNN::Express::_Cast<float>(output);
        info = output->getInfo();
    }
    MNN_PRINT("%s: (", name.c_str());
    for (int i=0; i<info->dim.size(); ++i) {
        MNN_PRINT("%d, ", info->dim[i]);
    }
    MNN_PRINT(")\n");
    auto targetValue = _Input({info->dim}, info->order, info->type);
    auto targetPtr = targetValue->writeMap<float>();
    for (int i=0; i<info->size; ++i) {
        double tempValue;
        outputOrigin >> tempValue;
        targetPtr[i] = tempValue;
    }

    auto absMax = MNN::Express::_ReduceMax(MNN::Express::_Abs(targetValue), {});
    absMax = MNN::Express::_Maximum(absMax, MNN::Express::_Scalar<float>(0.0001f));
    auto diff = MNN::Express::_Abs(targetValue - output);
    auto outputPtr = output->readMap<float>();
    auto diffAbsMax = MNN::Express::_ReduceMax(diff);
    auto absMaxV = absMax->readMap<float>()[0];
    auto diffAbsMaxV = diffAbsMax->readMap<float>()[0];
    if (absMaxV * maxError < diffAbsMaxV || std::isnan(absMaxV)) {
        MNN_ERROR("TESTERROR %s value error : absMaxV:%f - DiffMax %f\n", name.c_str(), absMaxV, diffAbsMaxV);
        return false;
    }
    return true;
}

int Cli::testconvert(const std::string& defaultCacheFile, const std::string& directName, float maxErrorRate, const std::string& backendConfigJson) {
    std::map<std::string, float> inputInfo;
    std::map<std::string, std::vector<int>> inputShape;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    {
        rapidjson::Document document;
        std::ostringstream jsonNameOs;
        jsonNameOs << directName << "/input.json";
        std::ifstream fileNames(jsonNameOs.str().c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        if (document.HasMember("inputs")) {
            auto inputsInfo = document["inputs"].GetArray();
            for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
                auto obj = iter->GetObject();
                std::string name = obj["name"].GetString();
                inputNames.emplace_back(name);
                MNN_PRINT("%s\n", name.c_str());
                if (obj.HasMember("value")) {
                    float value = obj["value"].GetFloat();
                    inputInfo.insert(std::make_pair(name, value));
                }
                if (obj.HasMember("shape")) {
                    auto dims = obj["shape"].GetArray();
                    std::vector<int> shapes;
                    for (auto iter = dims.begin(); iter != dims.end(); iter++) {
                        shapes.emplace_back(iter->GetInt());
                    }
                    inputShape.insert(std::make_pair(name, shapes));
                }
            }
        }
        if (document.HasMember("outputs")) {
            auto array = document["outputs"].GetArray();
            for (auto iter = array.begin(); iter !=array.end(); iter++) {
                std::string name = iter->GetString();
                MNN_PRINT("output: %s\n", name.c_str());
                outputNames.emplace_back(name);
            }
        }
    }

    // create session
    MNN::ScheduleConfig config;
    config.type      = MNN_FORWARD_CPU;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = 2;
    // If type not fount, let it failed
    config.backupType = MNN_FORWARD_CPU;
    BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(1);
    config.backendConfig     = &backendConfig;

    if (!backendConfigJson.empty()) {
        do {
            rapidjson::Document configDoc;
            std::ifstream configOs(backendConfigJson.c_str());
            if (configOs.fail()) {
                break;
            }
            std::ostringstream outputConfigOs;
            outputConfigOs << configOs.rdbuf();
            auto outputStr = outputConfigOs.str();
            configDoc.Parse(outputStr.c_str());
            if (configDoc.HasParseError()) {
                MNN_ERROR("Invalid json for backend config\n");
                break;
            }
            if (configDoc.HasMember("backend")) {
                config.type = (MNNForwardType)configDoc["backend"].GetInt();
            }
            if (configDoc.HasMember("mode")) {
                config.mode = configDoc["mode"].GetInt();
            }
            if (configDoc.HasMember("precision")) {
                config.backendConfig->precision = (MNN::BackendConfig::PrecisionMode)configDoc["precision"].GetInt();
            }
            if (configDoc.HasMember("memory")) {
                config.backendConfig->memory = (MNN::BackendConfig::MemoryMode)configDoc["memory"].GetInt();
            }
            if (configDoc.HasMember("power")) {
                config.backendConfig->power = (MNN::BackendConfig::PowerMode)configDoc["power"].GetInt();
            }
        } while (false);
    }

    MNN::Express::Module::Config mConfig;
    mConfig.shapeMutable = true;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile("./convert_cache.mnn.weight");
    std::shared_ptr<MNN::Express::Module> net(MNN::Express::Module::load(inputNames, outputNames, defaultCacheFile.c_str(), rtmgr, &mConfig));
    std::shared_ptr<MNN::Express::Module> net2;
    net2.reset(MNN::Express::Module::clone(net.get()));
    net = net2;
    auto mInfo = net->getInfo();
    std::vector<MNN::Express::VARP> inputs(mInfo->inputs.size());
#define LOAD_DATA(TYPE)\
    if (inputInfo.find(inputName) != inputInfo.end()) {\
        auto value = inputInfo[inputName];\
        for (int i=0; i<info->size; ++i) {\
            ptr[i] = value;\
        }\
    } else {\
        std::ostringstream fileNameOs;\
        fileNameOs << directName << "/" << inputName << ".txt";\
        auto fileName = fileNameOs.str();\
        std::ifstream inputOs(fileName.c_str());\
        if (inputOs.fail()) {\
            MNN_ERROR("TESTERROR Can't open %s\n", fileName.c_str());\
            continue;\
        }\
        for (int i=0; i<info->size; ++i) {\
            double tempValue;inputOs >> tempValue;\
            ptr[i] = tempValue;\
        }\
    }
    // Load inputs
    for (int i=0; i<inputs.size(); ++i) {
        auto inputName = inputNames[i];
        // Resize
        auto shapeIter = inputShape.find(inputName);
        if (shapeIter != inputShape.end()) {
            auto s = shapeIter->second;
            inputs[i] = _Input(s, mInfo->defaultFormat, mInfo->inputs[i].type);
        } else {
            inputs[i] = _Input(mInfo->inputs[i].dim, mInfo->inputs[i].order, mInfo->inputs[i].type);
        }
        auto info = inputs[i]->getInfo();
        auto iter = inputInfo.find(inputNames[i]);
        if (iter != inputInfo.end()) {
            auto ptr = inputs[i]->writeMap<float>();
            for (int v=0; v<mInfo->inputs[i].size; ++v) {
                ptr[v] = iter->second;
            }
            continue;
        }
        if (info->type == halide_type_of<float>()){
            auto ptr = inputs[i]->writeMap<float>();
            LOAD_DATA(float)
        } else {
            auto floatVar = _Input(info->dim, info->order, halide_type_of<float>());
            auto ptr = floatVar->writeMap<float>();
            LOAD_DATA(float)
            auto temp = _Cast(floatVar, info->type);
            inputs[i]->input(temp);
        }
        inputs[i] = _Convert(inputs[i], mInfo->inputs[i].order);
    }
#undef LOAD_DATA
    bool modelError = false;
    // Module Branch
    auto outputs = net->onForward(inputs);
    for (int i=0; i<outputNames.size(); ++i) {
        auto name = outputNames[i];
        auto v = outputs[i];
        auto info = v->getInfo();
        if (nullptr == info) {
            continue;
        }
        if (info->order == MNN::Express::NC4HW4 && info->dim.size() > 1) {
            v = MNN::Express::_Convert(v, mInfo->defaultFormat);
        }
        if (info->type.code != halide_type_float) {
            v = MNN::Express::_Cast<float>(v);
        }
        v.fix(MNN::Express::VARP::CONSTANT);
        outputs[i] = v;
    }

    for (int i=0; i<outputNames.size(); ++i) {
        auto output = outputs[i];
        bool success = compareOutput(output, directName, outputNames[i], mInfo->defaultFormat, i, maxErrorRate);
        if (!success) {
            modelError = true;
            MNN_ERROR("Error for output %s\n", outputNames[i].c_str());
        }
    }

    if (modelError) {
        MNN_ERROR("Save mnn result to  .error director\n");
        for (int i=0; i<outputNames.size(); ++i) {
            auto v = outputs[i];
            auto name = outputNames[i];
            auto info = v->getInfo();
            if (nullptr == info) {
                continue;
            }
            if (info->order == MNN::Express::NC4HW4 && info->dim.size() > 1) {
                v = MNN::Express::_Convert(v, mInfo->defaultFormat);
            }
            if (info->type.code != halide_type_float) {
                v = MNN::Express::_Cast<float>(v);
            }
            v.fix(MNN::Express::VARP::CONSTANT);
            info = v->getInfo();
            std::ofstream _output((".error/" + name + ".txt").c_str());
            auto ptr = v->readMap<float>();
            for (int v=0; v<info->size; ++v) {
                _output << ptr[v] << "\n";
            }
            v->setName(name);
            outputs.emplace_back(v);
        }
        MNN::Express::Variable::save(outputs, ".Error.mnn");
        return 0;
    }
    MNN_PRINT("TEST_SUCCESS\n");
    return 0;
}

bool Cli::mnn2json(const char* modelFile, const char* jsonFile, int flag) {
    std::ifstream inputFile(modelFile, std::ios::binary);
    inputFile.seekg(0, std::ios::end);
    auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];

    inputFile.read((char*)buffer, size);
    std::ofstream output(jsonFile);

    if (flag > 3) {
        MNN_PRINT("Dont't add convweight\n");
        auto netT = MNN::UnPackNet((void*)buffer);
        auto treatFunction = [&](MNN::OpT* opParam) {
            auto type = opParam->main.type;
            if (type == MNN::OpParameter::OpParameter_Convolution2D) {
                auto param = opParam->main.AsConvolution2D();
                param->weight.clear();
                param->bias.clear();
                if (param->symmetricQuan) {
                    param->symmetricQuan->weight.clear();
                }
                if (param->quanParameter) {
                    param->quanParameter->buffer.clear();
                }
            } else if (type == MNN::OpParameter::OpParameter_Blob) {
                size_t totalSize = 1;
                auto blobT = opParam->main.AsBlob();
                for (int v=0; v<blobT->dims.size(); ++v) {
                    totalSize *= blobT->dims[v];
                }
                if (totalSize > 20) {
                    blobT->float32s.clear();
                    blobT->int8s.clear();
                    blobT->uint8s.clear();
                    blobT->int32s.clear();
                    blobT->int64s.clear();
                }
            } else if (type == MNN::OpParameter::OpParameter_Convolution2D) {
                opParam->main.AsConvolution2D()->weight.clear();
                opParam->main.AsConvolution2D()->bias.clear();
            } else if (type == MNN::OpParameter::OpParameter_MatMul) {
                opParam->main.AsMatMul()->weight.clear();
                opParam->main.AsMatMul()->bias.clear();
            } else if (type == MNN::OpParameter::OpParameter_PRelu) {
                opParam->main.AsPRelu()->slope.clear();
            } else if (type == MNN::OpParameter::OpParameter_Extra) {
                auto extra = opParam->main.AsExtra();
                extra->info.clear();
            } else if(type == MNN::OpParameter::OpParameter_LSTM){
                auto param = opParam->main.AsLSTM();
                if (param->weightH) {
                    param->weightH->float32s.clear();
                }
                if (param->weightI) {
                    param->weightI->float32s.clear();
                }
                if (param->bias) {
                    param->bias->float32s.clear();
                }
            }
        };
        for (int i = 0; i < netT->oplists.size(); ++i) {
            treatFunction(netT->oplists[i].get());
        }
        for (int i = 0; i < netT->subgraphs.size(); ++i) {
            for (int j=0; j<netT->subgraphs[i]->nodes.size(); ++j) {
                treatFunction(netT->subgraphs[i]->nodes[j].get());
            }
        }
        if (flag > 4) {
            printf("Separate dump subgraph\n");
            for (int i=0; i<netT->subgraphs.size(); ++i) {
                auto& g = netT->subgraphs[i];
                flatbuffers::FlatBufferBuilder newBuilder(1024);
                auto root = MNN::SubGraphProto::Pack(newBuilder, g.get());
                newBuilder.Finish(root);
                auto content = newBuilder.GetBufferPointer();
                char subGraphNameStr[128];
                sprintf(subGraphNameStr, "%s_%d", jsonFile, i);
                printf("Dump subgraph %s to %s\n", g->name.c_str(), subGraphNameStr);
                std::ofstream tempOutput(subGraphNameStr);
                auto s       = flatbuffers::FlatBufferToString((const uint8_t*)content, MNN::SubGraphProtoTypeTable());
                tempOutput << s;
            }
            netT->subgraphs.clear();
        }
        flatbuffers::FlatBufferBuilder newBuilder(1024);
        auto root = MNN::Net::Pack(newBuilder, netT.get());
        MNN::FinishNetBuffer(newBuilder, root);
        {
            auto content = newBuilder.GetBufferPointer();
            auto s       = flatbuffers::FlatBufferToString((const uint8_t*)content, MNN::NetTypeTable());
            output << s;
        }
    } else {
        auto s = flatbuffers::FlatBufferToString((const uint8_t*)buffer, MNN::NetTypeTable());
        output << s;
    }

    delete[] buffer;
    return true;
}

#define VECTOR_EXTRACT(FLATBUFFER_TYPE, CPP_TYPE, JSON_TYPE)\
case flatbuffers::ET_##FLATBUFFER_TYPE:\
{\
    std::vector<CPP_TYPE> data(array.Size());\
    for (int i=0; i<array.Size(); ++i) {\
        data[i] = array[i].JSON_TYPE();\
    }\
    indexes[pos].second = builder.CreateVector(data).Union();\
    break;\
}\

#define SCALAR_EXTRACT(FLATBUFFER_TYPE, CPP_TYPE, JSON_TYPE)\
case flatbuffers::ET_##FLATBUFFER_TYPE:\
{\
builder.AddElement(field, (CPP_TYPE)(iter->value.JSON_TYPE()), (CPP_TYPE)0);\
break;\
}
static flatbuffers::Offset<void> _writeJsonToFlatbuffer(const flatbuffers::TypeTable * table, flatbuffers::FlatBufferBuilder& builder, const rapidjson::GenericObject<false, rapidjson::GenericValue<rapidjson::UTF8<>>>& object) {
    std::vector<std::pair<int, flatbuffers::Offset<void>>> indexes;
    // Load union type for easy to use
    std::map<std::string, int> unionNames;
    for (int i=0; i<table->num_elems; ++i) {
        if (table->type_codes[i].sequence_ref == -1) {
            continue;
        }
        const flatbuffers::TypeTable *ref = table->type_refs[table->type_codes[i].sequence_ref]();
        if (ref->st == flatbuffers::ST_UNION) {
            unionNames.insert(std::make_pair(std::string(table->names[i]) + "_type", i));
        }
    }
    // Find index and cache
    std::map<int, int> unionTypes;
    for (auto iter = object.begin(); iter !=object.end(); iter++) {
        auto name = iter->name.GetString();
        int index = -1;
        for (int i=0; i<table->num_elems; ++i) {
            if (0 == ::strcmp(table->names[i], name)) {
                index = i;
                break;
            }
        }
        auto uiter = unionNames.find(name);
        if (uiter != unionNames.end()) {
            // Find union type id
            auto value = iter->value.GetString();
            int typePos = -1;
            auto unionIndex = uiter->second;
            auto ref = table->type_refs[table->type_codes[unionIndex].sequence_ref]();
            for (int j=0; j<ref->num_elems; ++j) {
                if (0 == ::strcmp(ref->names[j], value)) {
                    typePos = j;
                    break;
                }
            }
            if (-1 == typePos) {
                MNN_ERROR("Can't find union type\n");
                continue;
            }
            if (typePos > 0) {
                // First is None
                unionTypes.insert(std::make_pair(unionIndex, typePos-1));
            }
        }
        if (index == -1) {
            MNN_PRINT("Invalid: %s, Skip it\n", name);
        }
        indexes.emplace_back(std::make_pair(index, 0));
    }

    // resolve single object
    int pos = 0;
    for (auto iter = object.begin(); iter !=object.end(); iter++, pos++) {
        int index = indexes[pos].first;
        if (-1 == index) {
            continue;
        }
        auto code = table->type_codes[index];
        if (code.is_vector) {
            continue;
        }
        if (code.sequence_ref != -1 && code.base_type == flatbuffers::ET_SEQUENCE) {
            const flatbuffers::TypeTable *ref = table->type_refs[code.sequence_ref]();
            if (ref->st == flatbuffers::ST_TABLE) {
                indexes[pos].second = _writeJsonToFlatbuffer(ref, builder, iter->value.GetObject());
            } else if (ref->st == flatbuffers::ST_UNION) {
                auto unionInd = unionTypes.find(index)->second;
                ref = ref->type_refs[unionInd]();
                indexes[pos].second = _writeJsonToFlatbuffer(ref, builder, iter->value.GetObject());
            }
        }
    }

    // Resolve Vector and String
    pos = 0;
    for (auto iter = object.begin(); iter !=object.end(); iter++, pos++) {
        int index = indexes[pos].first;
        if (-1 == index) {
            continue;
        }
        auto code = table->type_codes[index];
        if (!code.is_vector) {
            if (code.base_type == flatbuffers::ET_STRING) {
                indexes[pos].second = builder.CreateString(iter->value.GetString()).Union();
            }
            continue;
        }
        auto array = iter->value.GetArray();
        if (code.sequence_ref != -1) {
            const flatbuffers::TypeTable *ref = table->type_refs[code.sequence_ref]();
            std::vector<flatbuffers::Offset<void>> offsets(array.Size());
            for (int i=0; i<array.Size(); ++i) {
                offsets[i] = _writeJsonToFlatbuffer(ref, builder, array[i].GetObject());
            }
            indexes[pos].second = builder.CreateVector(offsets.data(), offsets.size()).Union();
            continue;
        }
        switch (code.base_type) {
                VECTOR_EXTRACT(BOOL, bool, GetBool);
                VECTOR_EXTRACT(CHAR, char, GetInt);
                VECTOR_EXTRACT(UCHAR, uint8_t, GetInt);
                VECTOR_EXTRACT(SHORT, int16_t, GetInt);
                VECTOR_EXTRACT(USHORT, uint16_t, GetInt);
                VECTOR_EXTRACT(INT, int, GetInt);
                VECTOR_EXTRACT(UINT, uint32_t, GetUint);
                VECTOR_EXTRACT(LONG, int64_t, GetInt64);
                VECTOR_EXTRACT(ULONG, uint64_t, GetUint64);
                VECTOR_EXTRACT(FLOAT, float, GetFloat);
                VECTOR_EXTRACT(DOUBLE, double, GetDouble);
            case flatbuffers::ET_STRING:
            {
                std::vector<std::string> data(array.Size());
                for (int i=0; i<array.Size(); ++i) {
                    data[i] = array[i].GetString();
                }
                indexes[pos].second = builder.CreateVectorOfStrings(data).Union();
                break;
            }
            default:
                break;
        }
    }

    // Resolve Others
    pos = 0;
    auto start = builder.StartTable();
    for (auto iter = object.begin(); iter !=object.end(); iter++, pos++) {
        int index = indexes[pos].first;
        if (-1 == index) {
            continue;
        }
        auto field = 4 + index * 2;
        if (indexes[pos].second.o != 0) {
            builder.AddOffset(field, indexes[pos].second);
            continue;
        }
        auto code = table->type_codes[index];
        if (code.sequence_ref != -1) {
            const flatbuffers::TypeTable *ref = table->type_refs[code.sequence_ref]();
            int value = -1;
            if (ref->st == flatbuffers::ST_UNION || ref->st == flatbuffers::ST_ENUM) {
                auto type = iter->value.GetString();
                for (int i=0; i<ref->num_elems; ++i) {
                    if (0 == ::strcmp(type, ref->names[i])) {
                        if (nullptr == ref->values) {
                            value = i;
                        } else {
                            value = ref->values[i];
                        }
                    }
                }
                switch (code.base_type) {
                    case flatbuffers::ET_UTYPE:
                    case flatbuffers::ET_UINT:
                        builder.AddElement(field, (uint32_t)value, (uint32_t)0);
                        break;
                    case flatbuffers::ET_INT:
                        builder.AddElement(field, (int32_t)value, (int32_t)-1);
                        break;
                    case flatbuffers::ET_UCHAR:
                        builder.AddElement(field, (uint8_t)value, (uint8_t)0);
                        break;
                    case flatbuffers::ET_CHAR:
                        builder.AddElement(field, (int8_t)value, (int8_t)0);
                        break;
                    default:
                        break;
                }
                continue;
            }
        }
        switch (code.base_type) {
                SCALAR_EXTRACT(BOOL, bool, GetBool);
                SCALAR_EXTRACT(CHAR, char, GetInt);
                SCALAR_EXTRACT(UCHAR, uint8_t, GetInt);
                SCALAR_EXTRACT(SHORT, int16_t, GetInt);
                SCALAR_EXTRACT(USHORT, uint16_t, GetInt);
                SCALAR_EXTRACT(INT, int, GetInt);
                SCALAR_EXTRACT(UINT, uint32_t, GetUint);
                SCALAR_EXTRACT(LONG, int64_t, GetInt64);
                SCALAR_EXTRACT(ULONG, uint64_t, GetUint64);
                SCALAR_EXTRACT(FLOAT, float, GetFloat);
                SCALAR_EXTRACT(DOUBLE, double, GetDouble);
            default:
                break;
        }
    }
    return builder.EndTable(start);
}
bool Cli::json2mnn(const char* jsonFile, const char* modelFile) {
    rapidjson::Document document;
    {
        std::ifstream fileNames(jsonFile);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
    }
    auto object = document.GetObject();
    flatbuffers::FlatBufferBuilder builder;
    builder.ForceDefaults(true);
    auto table = MNN::NetTypeTable();
    auto offset = _writeJsonToFlatbuffer(table, builder, object);
    builder.Finish(offset);
    std::ofstream outputOs(modelFile, std::ios::binary);
    outputOs.write((char*)builder.GetBufferPointer(), builder.GetSize());
    return true;
}

};


bool CommonKit::FileIsExist(std::string path) {
#if defined(_MSC_VER)
    if (INVALID_FILE_ATTRIBUTES != GetFileAttributes(path.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
        return true;
    }
#else
    if ((access(path.c_str(), F_OK)) != -1) {
        return true;
    }
#endif
    return false;
}

bool CommonKit::json2protobuf(const char* jsonFile, const char* protoFile, MNN::Compression::Pipeline* pipeline) {
    rapidjson::Document document;
    {
        std::ifstream fileNames(jsonFile);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
    }
    if (!document.HasMember("pipeline")) {
        MNN_ERROR("Error||Invalid json file: missing pipeline member.\n");
        return 0;
    }
    auto pipelineInfo = document["pipeline"].GetObject();
    std::string version = pipelineInfo["version"].GetString();
    pipeline->set_version(version);

    auto algos = pipelineInfo["algo"].GetArray();
    for (auto iter = algos.begin(); iter != algos.end(); ++iter) {
        auto algoInfo = iter->GetObject();
        auto compressionType = (MNN::Compression::CompressionAlgo_CompressionType)algoInfo["type"].GetInt();
        std::unique_ptr<MNN::Compression::QuantizeParams> quant_params(new MNN::Compression::QuantizeParams());
        auto quantParamsInfo = algoInfo["quant_params"].GetObject();
        auto round_mode = quantParamsInfo["round_mode"].GetInt();
        quant_params->set_round_mode((MNN::Compression::QuantizeParams_RoundMode)round_mode);

        auto layer = quantParamsInfo["layer"].GetArray();
        for (auto ly = layer.begin(); ly != layer.end(); ++ly) {
            auto layerInfo = ly->GetObject();
            auto newLayer = quant_params->add_layer();
            if (layerInfo.HasMember("method")) {
                newLayer->set_method((MNN::Compression::LayerQuantizeParams_QuantMethod)layerInfo["method"].GetInt());
            }

            // Weight.
            auto weights_ = layerInfo["weight"].GetArray();
            for (auto w = weights_.begin(); w != weights_.end(); ++w) {
                // Get weight info.
                int bits = w->GetObject()["bits"].GetInt();
                auto name = w->GetObject()["name"].GetString();
                auto scale = w->GetObject()["scales"].GetArray();
                auto zeropoint = w->GetObject()["zero_point"].GetInt();
                auto clamp_min = w->GetObject()["clamp_min"].GetInt();
                auto clamp_max = w->GetObject()["clamp_max"].GetInt();
                // Write to newLayer
                auto weight = newLayer->add_weight();
                weight->set_bits(bits);
                weight->set_name(name);
                weight->set_clamp_max(clamp_max);
                weight->set_clamp_min(clamp_min);
                for (int k = 0; k < scale.Size(); ++k) {
                    weight->add_scales(scale[k].GetFloat());
                }
            }

            // Input.
            auto inputs_ = layerInfo["input"].GetArray();
            for (auto w = inputs_.begin(); w != inputs_.end(); ++w) {
                // Get weight info.
                int bits = w->GetObject()["bits"].GetInt();
                auto name = w->GetObject()["name"].GetString();
                auto scale = w->GetObject()["scales"].GetArray();
                auto zeropoint = w->GetObject()["zero_point"].GetInt();
                auto clamp_min = w->GetObject()["clamp_min"].GetInt();
                auto clamp_max = w->GetObject()["clamp_max"].GetInt();
                // Write to newLayer
                auto input = newLayer->add_input();
                input->set_bits(bits);
                input->set_name(name);
                input->set_clamp_max(clamp_max);
                input->set_clamp_min(clamp_min);
                for (int k = 0; k < scale.Size(); ++k) {
                    input->add_scales(scale[k].GetFloat());
                }
            }

            // Output.
            auto outputs_ = layerInfo["output"].GetArray();
            for (auto w = outputs_.begin(); w != outputs_.end(); ++w) {
                // Get weight info.
                int bits = w->GetObject()["bits"].GetInt();
                auto name = w->GetObject()["name"].GetString();
                auto scale = w->GetObject()["scales"].GetArray();
                auto zeropoint = w->GetObject()["zero_point"].GetInt();
                auto clamp_min = w->GetObject()["clamp_min"].GetInt();
                auto clamp_max = w->GetObject()["clamp_max"].GetInt();
                // Write to newLayer
                auto output = newLayer->add_output();
                output->set_bits(bits);
                output->set_name(name);
                output->set_clamp_max(clamp_max);
                output->set_clamp_min(clamp_min);
                for (int k = 0; k < scale.Size(); ++k) {
                    output->add_scales(scale[k].GetFloat());
                }
            }
        }
        MNN::Compression::CompressionAlgo* algo = pipeline->add_algo();
        algo->set_type(compressionType);
        auto params = algo->quant_params();
        params.CopyFrom(*quant_params.get());
    }
    // Write protobuf.bin
    if (protoFile) {
        std::ofstream output(protoFile, std::ios::out | std::ios::binary);
        if (!pipeline->SerializeToOstream(&output)) {
            MNN_ERROR("->Error: Fail saving Json file to protobuf file\n");
            return 0;
        }
        MNN_PRINT("Finish convert json file to protobuf binary file\n");
    }
    return 1;
}

