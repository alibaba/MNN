//
//  MNNDialect.hpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDIALECT_HPP
#define MNNDIALECT_HPP

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace mnn {

class MNNDialect : public mlir::Dialect {
public:
  explicit MNNDialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "mnn"; }
};

enum PoolType {
  MAXPOOL = 0,
  AVGPOOL
};

enum PoolPadType {
  CAFFE=0,
  VALID,
  SAME
};

} // end namespace mnn
} // end namespace mlir

#define GET_OP_CLASSES
#include "MNNOps.h.inc"

#endif // MNNDIALECT_HPP
