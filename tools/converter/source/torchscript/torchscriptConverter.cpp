//
//  torchscriptConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>

#include "MNN_generated.h"
#include "logkit.h"

#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"

#include "TorchScriptDialect.hpp"
#include "MLIRGen.hpp"
#include "Passes.hpp"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include <torch/script.h>

#include <sstream>



std::string getName(mlir::Value v) {
    std::string output_name;
    llvm::raw_string_ostream rso(output_name);
    v.print(rso);
    auto name = rso.str();
    return name.substr(1, name.find('=')-1);
}

std::vector<int> getShape(mlir::Type t) {
    std::vector<int> res;
    if (auto shapedTy = t.dyn_cast_or_null<mlir::ShapedType>()) {
        auto shape = shapedTy.getShape();
        for (auto dim : shape) {
            res.push_back(dim);
        }
    }
    return res;
}

int getInt(mlir::Attribute a) {
    if (!a) return 0;
    return static_cast<int>(a.cast<mlir::IntegerAttr>().getInt());
}
float getFloat(mlir::Attribute a) {
    if (!a) return 0.f;
    return static_cast<int>(a.cast<mlir::FloatAttr>().getValueAsDouble());
}

bool getBool(mlir::Attribute a) {
    if (!a) return false;
    auto boolAttr = a.cast<mlir::BoolAttr>();
    return boolAttr.getValue();
}

std::vector<float> getFloatVector(mlir::Attribute a) {
    std::vector<float> res;
    if (!a) return res;
    mlir::DenseFPElementsAttr elems = a.cast<mlir::DenseFPElementsAttr>();
    for (auto elem : elems.getFloatValues()) {
        res.push_back(elem.convertToFloat());
    }
    return res;
}

std::vector<int> getIntVector(mlir::Attribute a) {
    std::vector<int> res;
    if (!a) return res;
    mlir::DenseIntElementsAttr elems = a.cast<mlir::DenseIntElementsAttr>();
    for (auto elem : elems.getIntValues()) {
        res.push_back(elem.getSExtValue());
    }
    return res;
}

int torchscript2MNNNet(const std::string inputModel, const std::string bizCode,
                       std::unique_ptr<MNN::NetT>& netT) {
    printf("TorchScript Converter!\n");
    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::torchscript::TorchScriptDialect>();

    torch::jit::Module torchModule;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        torchModule = torch::jit::load(inputModel);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    mlir::OwningModuleRef module = torchscript::mlirGen(context, torchModule);
    if (!module) {
        return 1;
    }
    mlir::PassManager pm(&context);
    applyPassManagerCLOptions(pm);
    pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
    pm.addPass(mlir::torchscript::createConvertToMNNPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
    if (mlir::failed(pm.run(*module))) {
        return 4;
    }
    // module->dump();
    mlir::FuncOp main_fn = module->lookupSymbol<mlir::FuncOp>("main_graph");
    auto& block = main_fn.getBlocks().front();
    const mlir::Block::OpListType& operations = block.getOperations();
    std::map<std::string, int> tensors;
    for (auto iter = main_fn.args_begin(); iter != main_fn.args_end(); iter++) {
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = "";
        MNNOp->type      = MNN::OpType_Input;
        MNNOp->main.type = MNN::OpParameter_Input;
        auto param  = new MNN::InputT;
        param->dims = getShape(iter->getType());
        param->dtype = MNN::DataType_DT_FLOAT;
        param->dformat = MNN::MNN_DATA_FORMAT_NCHW;
        MNNOp->main.value = param;
        netT->tensorName.emplace_back("input");
        netT->oplists.emplace_back(MNNOp);
        MNNOp->outputIndexes.push_back(tensors.size());
        tensors[getName(*iter)] = tensors.size();
    }
    std::stringstream ss;
    int idx = 0;
    for (const mlir::Operation& _operation : operations) {
        mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);
        std::string op = operation.getName().getStringRef().str().substr(4, -1);
        if (op == "return") {
            for (auto in = operation.operand_begin(); in != operation.operand_end(); in++) {
                netT->outputName.push_back(getName(*in));
            }
            break;
        }
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = op + std::to_string(idx++);
        // TODO: replace the below code with table-drive-format
        if (op == "Const") {
            auto attr = operation.getAttr("value");
            auto param = new MNN::BlobT;
            param->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;
            param->dims = getShape(attr.getType());
            if (attr.isa<mlir::DenseFPElementsAttr>()) {
                param->dataType = MNN::DataType_DT_FLOAT;
                param->float32s = getFloatVector(attr);
            } else if (attr.isa<mlir::DenseIntElementsAttr>()) {
                param->dataType = MNN::DataType_DT_INT32;
                param->int32s = getIntVector(attr);
            }
            MNNOp->type      = MNN::OpType_Const;
            MNNOp->main.type  = MNN::OpParameter_Blob;
            MNNOp->main.value = param;
        } else if (op == "binary") {
            auto param = new MNN::BinaryOpT;
            param->opType     = static_cast<MNN::BinaryOpOperation>(getInt(operation.getAttr("type")));
            param->T          = MNN::DataType_DT_FLOAT;
            MNNOp->type      = MNN::OpType_BinaryOp;
            MNNOp->main.type = MNN::OpParameter_BinaryOp;
            MNNOp->main.value = param;
        } else if (op == "convolution") {
            auto param = new MNN::Convolution2DT;
            param->weight = getFloatVector(operation.getAttr("weight"));
            param->bias = getFloatVector(operation.getAttr("bias"));
            param->common.reset(new MNN::Convolution2DCommonT);
            auto common = param->common.get();
            common->dilateX = getInt(operation.getAttr("dilateX"));
            common->dilateY = getInt(operation.getAttr("dilateY"));
            common->strideX = getInt(operation.getAttr("strideX"));
            common->strideY = getInt(operation.getAttr("strideY"));
            common->kernelX = getInt(operation.getAttr("kernelX"));
            common->kernelX = getInt(operation.getAttr("kernelX"));
            common->kernelY = getInt(operation.getAttr("kernelY"));
            common->padX = getInt(operation.getAttr("padX"));
            common->padY = getInt(operation.getAttr("padY"));
            common->group = getInt(operation.getAttr("group"));
            common->outputCount = getInt(operation.getAttr("outputCount"));
            common->relu = getBool(operation.getAttr("relu"));
            common->padMode = static_cast<MNN::PadMode>(getInt(operation.getAttr("padMode")));
            common->relu6 = getBool(operation.getAttr("relu6"));
            common->inputCount = getInt(operation.getAttr("inputCount"));
            MNNOp->type      = MNN::OpType_Convolution;
            MNNOp->main.type = MNN::OpParameter_Convolution2D;
            MNNOp->main.value = param;
        } else if (op == "flatten") {
            auto param = new MNN::FlattenT;
            param->axis = getInt(operation.getAttr("start_dim"));
            param->endAxis = getInt(operation.getAttr("end_dim"));
            MNNOp->type      = MNN::OpType_Flatten;
            MNNOp->main.type = MNN::OpParameter_Flatten;
            MNNOp->main.value = param;
        } else if (op == "relu") {
            auto param = new MNN::ReluT;
            param->slope = getFloat(operation.getAttr("slope"));
            MNNOp->type      = MNN::OpType_ReLU;
            MNNOp->main.type = MNN::OpParameter_Relu;
            MNNOp->main.value = param;
        } else if (op == "batch_norm") {
            auto param = new MNN::BatchNormT;
            param->channels = getInt(operation.getAttr("channel"));
            param->slopeData = getFloatVector(operation.getAttr("slopeData"));
            param->meanData = getFloatVector(operation.getAttr("meanData"));
            param->varData = getFloatVector(operation.getAttr("varData"));
            param->biasData = getFloatVector(operation.getAttr("biasData"));
            param->Adata = getFloatVector(operation.getAttr("Aata"));
            param->Bdata = getFloatVector(operation.getAttr("Bata"));
            param->epsilon = getFloat(operation.getAttr("epsilon"));
            MNNOp->type      = MNN::OpType_BatchNorm;
            MNNOp->main.type = MNN::OpParameter_BatchNorm;
            MNNOp->main.value = param;
        } else if (op == "transpose") {
            // auto param = new MNN::PermuteT;
            // param->dims = getIntVector(operation.getAttr("dims"));
            auto param = new MNN::TransposeT;
            param->Tperm = MNN::DataType_DT_FLOAT;
            MNNOp->type      = MNN::OpType_Transpose;
            MNNOp->main.type = MNN::OpParameter_Transpose;
            MNNOp->main.value = param;
        } else if (op == "matmul") {
            auto param = new MNN::MatMulT;
            param->transposeA = getBool(operation.getAttr("transposeA"));
            param->transposeB = getBool(operation.getAttr("transposeB"));
            param->weight = getFloatVector(operation.getAttr("weight"));
            param->bias = getFloatVector(operation.getAttr("bias"));
            MNNOp->type      = MNN::OpType_MatMul;
            MNNOp->main.type = MNN::OpParameter_MatMul;
            MNNOp->main.value = param;
        } else if (op == "pool") {
            auto param = new MNN::PoolT;
            param->kernelX = getInt(operation.getAttr("kernelX"));
            param->kernelY = getInt(operation.getAttr("kernelY"));
            param->strideX = getInt(operation.getAttr("strideX"));
            param->strideY = getInt(operation.getAttr("strideY"));
            param->padX = getInt(operation.getAttr("padX"));
            param->padY = getInt(operation.getAttr("padY"));
            param->isGlobal = getBool(operation.getAttr("isGlobal"));
            param->type = static_cast<MNN::PoolType>(getInt(operation.getAttr("type")));
            param->padType = static_cast<MNN::PoolPadType>(getInt(operation.getAttr("padType")));
            param->ceilModel = getInt(operation.getAttr("ceilMode"));
            MNNOp->type      = MNN::OpType_Pooling;
            MNNOp->main.type = MNN::OpParameter_Pool;
            MNNOp->main.value = param;
        }
        // set input indexes
        for (auto in = operation.operand_begin(); in != operation.operand_end(); in++) {
            MNNOp->inputIndexes.push_back(tensors[getName(*in)]);
        }
        // set output indexes
        for (auto out = operation.result_begin(); out != operation.result_end(); out++) {
            MNNOp->outputIndexes.push_back(tensors.size());
            tensors[getName(*out)] = tensors.size();
            netT->tensorName.emplace_back(getName(*out));
        }
        netT->oplists.emplace_back(MNNOp);
    }
    netT->bizCode    = bizCode;
    return 0;
}
