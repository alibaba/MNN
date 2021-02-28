//
//  torchscriptConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MLIRGEN_HPP
#define MLIRGEN_HPP

#include <torch/script.h>
#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

using TorchModule = torch::jit::Module;

namespace torchscript {
class ModuleAST;

mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, TorchModule &torchModule);
} // namespace torchscript

#endif // MLIRGEN_HPP
