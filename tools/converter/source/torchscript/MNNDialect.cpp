//
//  MNNDialect.cpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNDialect.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::mnn;

//===----------------------------------------------------------------------===//
// MNNDialect
//===----------------------------------------------------------------------===//

MNNDialect::MNNDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<MNNDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "MNNOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MNN Operations
//===----------------------------------------------------------------------===//
#define BUILD_ARGS(NAME)\
void NAME::build(mlir::OpBuilder &builder, mlir::OperationState &state,\
                 ArrayRef<mlir::Value> args) {\
  state.addTypes(args[0].getType());\
  state.addOperands(args);\
}\

BUILD_ARGS(ReturnOp)
BUILD_ARGS(ReluOp)
BUILD_ARGS(TransposeOp)
BUILD_ARGS(MatMulOp)
#undef BUILD_ARGS
#define GET_OP_CLASSES
#include "MNNOps.cpp.inc"
