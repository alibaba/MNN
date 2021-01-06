//
//  Passes.hpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PASSES_HPP
#define PASSES_HPP

#include <memory>

namespace mlir {
class Pass;

namespace torchscript {

std::unique_ptr<mlir::Pass> createConvertToMNNPass();

} // end namespace torchscript
} // end namespace mlir

#endif // PASSES_HPP
