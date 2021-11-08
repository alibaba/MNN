//
//  TestPassManager.cpp
//  MNNConverter
//
//  Created by MNN on b'2020/12/07'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <string>

#include "converter/source/optimizer/passes/Pass.hpp"
#include "converter/source/optimizer/passes/PassRegistry.hpp"
#include "MNN_generated.h"
#include <iostream>

namespace MNN {
namespace passes {

using namespace Express;

// Register the `FuseExpandDimsAndConstant` pass which could fold the
// constant axis input of `ExpandDims`.
REGISTER_REWRITE_PASS(FuseExpandDimsAndConstant)
    .Verify([](PassContext* context) {
         EXPRP expr = context->node;
         // Check the current op is `ExpandDims` op.
         if (!expr->get() || expr->get()->type() != OpType_ExpandDims ||
             expr->inputs().size() < 2) {
             return false;
         }
         // Check the second input is `Constant` op.
         VARP axis = expr->inputs().at(1);
         if (axis->expr().first->inputType() != VARP::CONSTANT) {
             return false;
         }
         return true;
     })
    .Rewrite([](PassContext* context) {
         VARP axis = context->node->inputs().at(1);
         int axis_val = axis->readMap<int>()[0];
         std::unique_ptr<OpT> expand_dims_op(context->node->get()->UnPack());
         expand_dims_op->main.AsExpandDims()->axis = axis_val;

         auto expand_dims = Expr::create(expand_dims_op.get(),  // NOLINT
                                         {context->node->inputs().at(0)});
         Expr::replace(context->node, expand_dims);
         return true;
     });

std::unique_ptr<NetT> LoadModel(const char* modelFile) {
    std::ifstream inputFile(modelFile, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cout << "file not found: " << modelFile << std::endl;
        return nullptr;
    }
    inputFile.seekg(0, std::ios::end);
    auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    inputFile.read((char*)buffer, size);
    inputFile.close();
    return MNN::UnPackNet(buffer);
}

void DumpModel(std::unique_ptr<NetT>& net, const char* dumpFile) {
    flatbuffers::FlatBufferBuilder builder(1024);
    builder.ForceDefaults(true);
    auto len = MNN::Net::Pack(builder, net.get());
    builder.Finish(len);

    auto buffer = builder.GetBufferPointer();
    int size = builder.GetSize();
    std::ofstream output(dumpFile, std::ofstream::binary);
    output.write((const char*)buffer, size);
}

void TestPassManager(const char* source_modelfile, const char* target_modelfile) {
    std::unique_ptr<PassContext> ctx(new PassContext);
    PassManager pm(ctx.get());
    pm.AddPass("FuseExpandDimsAndConstant");

    auto net = LoadModel(source_modelfile);
    if (net == nullptr) {
        std::cout << "error load source mnn model" << std::endl;
        return;
    }
    auto optimized_net = pm.Run(net);

    // Dump model.
    DumpModel(optimized_net, target_modelfile);
    std::cout << "Optimized, file saved to: " << target_modelfile << std::endl;
}

}  // namespace passes
}  // namespace MNN

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./TestPassManager mnn_model_to_be_optimized.mnn optimized_mnn_model.mnn" << std::endl;
        return 0;
    }

    char *source_modelfile = argv[1];
    char *target_modelfile = argv[2];

    MNN::passes::TestPassManager(source_modelfile, target_modelfile);
    return 0;
}
