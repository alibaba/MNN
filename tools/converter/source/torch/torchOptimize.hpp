//
//  torchOptimize.hpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TORCHOPTIMIZE_HPP
#define TORCHOPTIMIZE_HPP
#include <torch/script.h>
namespace torch {
namespace jit {

std::shared_ptr<Graph> torchOptPass(Module& module);

}
}

#endif // TORCHOPTIMIZE_HPP
