//
//  TorchScriptMLIRGen.cpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "MLIRGen.hpp"
#include "TorchScriptDialect.hpp"
#include <torch/csrc/jit/passes/inliner.h>

#include <numeric>

using namespace mlir::torchscript;
using namespace torchscript;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context) : context(context), builder(&context) {}

    /// Public API: convert the AST for a Toy module (source file) to an MLIR
    /// Module operation.
    mlir::ModuleOp mlirGen(TorchModule &torchModule) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    // get main method graph
    auto graph = torchModule.get_methods()[0].graph();
    // inline the graph so that don't deal other methods
    Inline(*(graph.get()));
    // graph->dump();

    // 1. set function type and declare all args
    llvm::SmallVector<mlir::Type, 4> argTypes;
    std::vector<const std::string> argNames;
    if (graph->inputs()[0]->type()->is_module()) {
       addModule(graph->inputs()[0]->debugName(), torchModule);
    }
    for (const auto &input : graph->inputs()) {
        argTypes.emplace_back(getType(input->type()));
        argNames.push_back(input->debugName());
    }
    auto funcType = builder.getFunctionType(argTypes, llvm::None);
    auto mainFunc = mlir::FuncOp::create(UnknownLoc(), "main_graph", funcType);

    auto &entryBlock = *mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    for (const auto &name_value :
         llvm::zip(argNames, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(name_value),
                         std::get<1>(name_value))))
        return nullptr;
    }

    // 2. add all nodes in function body
    for (const auto &node : graph->nodes()) {
        ImportNode(node, torchModule);
    }

    // 3. add ret values and reset function type
    llvm::SmallVector<mlir::Type, 4> retTypes;
    llvm::SmallVector<mlir::Value, 4> retValues;
    for (const auto &output : graph->outputs()) {
        auto varName = output->debugName();
        if (auto variable = lookup(varName)) {
            retTypes.emplace_back(getTensorType({}));
            retValues.emplace_back(std::move(variable));
        }
    }
    builder.create<ReturnOp>(UnknownLoc(), retValues);
    funcType = builder.getFunctionType(argTypes, retTypes);
    mainFunc.setType(funcType);

    // 4. push function to module
    theModule.push_back(mainFunc);

    // Create an MLIR function for the given prototype.
    // mlir::FuncOp function(mlirGen(*funcAST.getProto()));

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
    /// A "module" matches a Toy source file: containing a list of functions.
    mlir::ModuleOp theModule;

    /// MLIR Context
    mlir::MLIRContext& context;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    /// The symbol table maps a variable name to a value in the current scope.
    /// Entering a function creates a new scope, and the function arguments are
    /// added to the mapping. When the processing of a function is terminated, the
    /// scope is destroyed and the mappings created in this scope are dropped.
    std::unordered_map<std::string, mlir::Value> symbolTable;
    std::unordered_map<std::string, TorchModule> moduleTable;


    mlir::LogicalResult addModule(const std::string& var, const TorchModule& module) {
        if (moduleTable.count(var))
            return mlir::failure();
        moduleTable[var] = module;
        return mlir::success();
    }

    // c10::IValue getModule(std::string name) {
    //     if (moduleTable.count(name)) {
    //         return moduleTable[name];
    //     }
    //     return nullptr;
    // }

    /// Declare a variable in the current scope, return success if the variable
    /// wasn't declared yet.
    mlir::LogicalResult declare(const std::string& var, mlir::Value value) {
        if (symbolTable.count(var))
            return mlir::failure();
        symbolTable[var] = value;
        return mlir::success();
    }

    mlir::Value lookup(std::string name) {
        if (symbolTable.count(name)) {
            return symbolTable[name];
        }
        return nullptr;
    }

    mlir::Type getType(c10::TypePtr type) {
        auto typeStr = type->str();
        if (type->is_module()) {
            return mlir::OpaqueType::get(mlir::Identifier::get("torchscript", &context), typeStr, &context);
        }
        std::unordered_map<std::string, mlir::Type> strTypes;
        strTypes["float"] = mlir::Float32Type::get(&context);
        strTypes["int"] = mlir::IntegerType::get(32, &context);
        strTypes["bool"] = mlir::IntegerType::get(1, &context);
        strTypes["None"] = mlir::NoneType::get(&context);
        strTypes["Tensor"] = getTensorType({1, 3, 224, 224});
        strTypes["Function"] = mlir::FunctionType::get(mlir::TypeRange(), mlir::TypeRange(), &context);
        if (strTypes.find(typeStr) != strTypes.end()) {
            return strTypes[typeStr];
        }
        return nullptr;
    }

    /// Build a tensor type from a list of shape dimensions.
    mlir::Type getTensorType(ArrayRef<int64_t> shape, mlir::Type dtype) {
      // If the shape is empty, then this type is unranked.
      if (shape.empty())
        return mlir::UnrankedTensorType::get(dtype);

      // Otherwise, we use the given shape.
      return mlir::RankedTensorType::get(shape, dtype);
    }
    mlir::Type getTensorType(ArrayRef<int64_t> shape) {
        return getTensorType(shape, builder.getF32Type());
    }

    mlir::Location UnknownLoc() {
        return mlir::UnknownLoc::get(&context);
    }

    void buildTensorAttribute(const at::Tensor& tensor, mlir::Attribute& dataAttribute,
                              mlir::Type& outputDataType) {
        auto shape = tensor.sizes().vec();
        auto scalarType = tensor.scalar_type();
        #define BUILD_TENSOR(T, X)\
        {\
            std::vector<T> data;\
            data.resize(tensor.numel());\
            for (int i = 0; i < tensor.numel(); i++) {\
                data[i] = tensor.data_ptr<T>()[i];\
            }\
            auto elemType = builder.get##X##Type();\
            auto shapeType = mlir::RankedTensorType::get(shape, elemType);\
            outputDataType = shapeType;\
            dataAttribute = mlir::DenseElementsAttr::get(shapeType, llvm::makeArrayRef(data));\
        }
        switch (scalarType) {
            case at::ScalarType::Int:
                BUILD_TENSOR(int32_t, I32)
                break;
            case at:: ScalarType::Long:
                BUILD_TENSOR(int64_t, I64)
                break;
            case at::ScalarType::Float:
                BUILD_TENSOR(float, F32)
                break;
            case at::ScalarType::Double:
                BUILD_TENSOR(double, F64)
                break;
            default:
                printf("Not support Type!\n");
                break;
        }
        #undef BUILD_TENSOR
    }
    void ImportNode(torch::jit::Node* node, TorchModule& torchModule) {
        mlir::Value value = nullptr;
        auto output = node->outputs()[0];
        auto outputName = output->debugName();
        std::string opType = node->kind().toUnqualString();
        const std::string& type = output->type()->str();
        mlir::Type outputDataType = getType(output->type());
        // TODO: replace the below code with table-drive-format
        if (opType == "Constant") {
            if (type == "None") {
                value = builder.create<ConstantOp>(UnknownLoc(), outputDataType, mlir::None);
            } else {
                auto attr = node->attributeNames()[0];
                auto kind = node->kindOf(attr);
                // std::cout << torch::jit::toString(kind) << std::endl;
                mlir::Attribute dataAttribute;
                switch (kind) {
                    case torch::jit::AttributeKind::f:
                        dataAttribute = mlir::FloatAttr::get(outputDataType, node->f(attr));
                        break;
                    case torch::jit::AttributeKind::i:
                        dataAttribute = mlir::IntegerAttr::get(outputDataType, node->i(attr));
                        break;
                    case torch::jit::AttributeKind::s:
                        dataAttribute = mlir::StringAttr::get(node->s(attr), &context);
                        break;
                    case torch::jit::AttributeKind::t: {
                        auto tensor = node->t(attr);
                        buildTensorAttribute(tensor, dataAttribute, outputDataType);
                        break;
                    }
                    default:
                        break;
                }
                value = builder.create<ConstantOp>(UnknownLoc(), outputDataType, dataAttribute);
            }
        }
        if (opType == "tensor") {
            auto arg0 = lookup(node->inputs()[0]->debugName());
            auto arg1 = lookup(node->inputs()[1]->debugName());
            auto arg2 = lookup(node->inputs()[2]->debugName());
            auto arg3 = lookup(node->inputs()[3]->debugName());
            if (arg0 && arg1 && arg2 && arg3) {
                auto shape = arg0.getType().dyn_cast<mlir::VectorType>().getShape();
                auto dtype = arg0.getType().dyn_cast<mlir::VectorType>().getElementType();
                auto tensorType = getTensorType(shape, dtype);
                value = builder.create<TensorOp>(UnknownLoc(), tensorType, arg0, arg1, arg2, arg3);
            } else {
                emitError(UnknownLoc(), "TensorOp arg is WRONG!");
            }
        }

        #define ARGS_OP(NAME, OP)\
        if (opType == #NAME) {\
            llvm::SmallVector<mlir::Value, 4> inValues;\
            for (const auto input : node->inputs()) {\
                if (auto variable = lookup(input->debugName())) {\
                    inValues.emplace_back(std::move(variable)); \
                }\
            }\
            value = builder.create<OP>(UnknownLoc(), inValues);\
        }
        ARGS_OP(addmm, AddmmOp)
        ARGS_OP(adaptive_avg_pool2d, AdaptiveAvgPool2dOp)
        ARGS_OP(flatten, FlattenOp)
        ARGS_OP(t, TransposeOp)
        ARGS_OP(ListConstruct, ListOp)
        ARGS_OP(add, AddOp)
        ARGS_OP(add_, AddOp)
        ARGS_OP(mul, MulOp)
        ARGS_OP(_convolution, ConvolutionOp)
        ARGS_OP(batch_norm, BatchNormOp)
        ARGS_OP(max_pool2d, MaxPool2dOp)
        ARGS_OP(relu_, ReluOp)
        #undef ARGS_OP

        if (opType == "GetAttr") {
            std::string objName = node->inputs()[0]->debugName();
            auto obj = lookup(objName);
            if (outputDataType && obj) {
                auto attr = node->attributeNames()[0];
                auto attrStr = node->s(attr);
                auto dataAttribute = mlir::StringAttr::get(attrStr, &context);
                c10::IValue submodule;
                if (objName == "self.1") {
                    submodule = torchModule.attr(attrStr);
                } else {
                    if (!moduleTable.count(objName)) {
                        return;
                    }
                    auto module = moduleTable[objName];
                    submodule = module.attr(attrStr);
                }
                if (submodule.isModule()) {
                    addModule(outputName, submodule.toModule());
                } else if (submodule.isTensor()){
                    at::Tensor tensor = submodule.toTensor();
                    // mlir::Attribute dataAttribute;
                    buildTensorAttribute(tensor, dataAttribute, outputDataType);
                    // std::cout << "submodule : " << submodule.tagKind() << std::endl;
                }
                value = builder.create<GetOp>(UnknownLoc(), outputDataType, dataAttribute, obj);
            } else {
                return;
            }
        }
        if (!value) {
            std::cout << "Don't support type = " << opType << std::endl;
            return;
        }
        // add to symTable
        if (failed(declare(outputName, value))) {
            emitError(UnknownLoc(), "Declare var failed!");
        }
    }
};

} // namespace

namespace torchscript {

// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              TorchModule &torchModule) {
  return MLIRGenImpl(context).mlirGen(torchModule);
}

} // namespace torchscript
