//
//  TorchScriptDialect.cpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TorchScriptDialect.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::torchscript;

//===----------------------------------------------------------------------===//
// TorchScriptDialect
//===----------------------------------------------------------------------===//

TorchScriptDialect::TorchScriptDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<TorchScriptDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "TorchScriptOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TorchScript Operations
//===----------------------------------------------------------------------===//
void ListOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   ArrayRef<mlir::Value> args) {
    int elemNum = args.size();
    auto elemType = args[0].getType();
    mlir::Type listType = elemType;
    if (elemType.isa<VectorType>()) {
        auto vecType = elemType.dyn_cast<VectorType>();
        auto shape = vecType.getShape();
        auto newShape(shape.vec());
        newShape.insert(newShape.begin(), elemNum);
        listType = mlir::VectorType::get(newShape, vecType.getElementType());
    } else {
        listType = mlir::VectorType::get({elemNum}, elemType);
    }
    state.addTypes(listType);
    state.addOperands(args);
}
void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  ArrayRef<mlir::Value> args) {
  auto arrayTy = args[0].getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  state.addTypes(RankedTensorType::get(dims, arrayTy.getElementType()));
  state.addOperands(args);
}
//===----------------------------------------------------------------------===//
// Arg-list Op

#define BUILD_ARGS(NAME)\
void NAME::build(mlir::OpBuilder &builder, mlir::OperationState &state,\
                  ArrayRef<mlir::Value> args) {\
  state.addTypes(args[0].getType());\
  state.addOperands(args);\
}\

BUILD_ARGS(AdaptiveAvgPool2dOp)
BUILD_ARGS(AddOp)
BUILD_ARGS(AddmmOp)
BUILD_ARGS(BatchNormOp)
BUILD_ARGS(ConvolutionOp)
BUILD_ARGS(FlattenOp)
BUILD_ARGS(MaxPool2dOp)
BUILD_ARGS(MulOp)
BUILD_ARGS(ReluOp)
#undef BUILD_ARGS

#define GET_OP_CLASSES
#include "TorchScriptOps.cpp.inc"
