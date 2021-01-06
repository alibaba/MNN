//
//  TorchScriptDialect.hpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TORCHSCRIPTDIALECT_HPP
#define TORCHSCRIPTDIALECT_HPP

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace torchscript {

class TorchScriptDialect : public mlir::Dialect {
public:
  explicit TorchScriptDialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "torchscript"; }
};

} // end namespace torchscript
} // end namespace mlir

#define GET_OP_CLASSES
#include "TorchScriptOps.h.inc"

#endif // TORCHSCRIPTDIALECT_HPP
