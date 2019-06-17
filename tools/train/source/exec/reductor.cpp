//
//  reductor.cpp
//  MNN
//
//  Created by MNN on 2019/05/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include "MNNDefine.h"
#include "Macro.h"
#include "OpConverter.hpp"
#include "OpGrad.hpp"
#include "Tensor.hpp"
#include "converter/source/IR/MNN_generated.h"
using namespace MNN;
using namespace std;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./reductor.out train.bin dst.bin\n");
        return 0;
    }

    unique_ptr<NetT> net;
    {
        const char* fileName = argv[1];
        FUNC_PRINT_ALL(fileName, s);
        std::ifstream fs(fileName, std::ifstream::in | std::ifstream::binary);
        std::ostringstream os;
        os << fs.rdbuf();
        net = UnPackNet(os.str().c_str());
    }

    // crop extra op, get actual tensors
    std::vector<std::unique_ptr<OpT>> originOpLists;
    int originTensorNameSize;

    MNN_ASSERT(net->oplists.size() >= 3);
    for (int i = 0; i < net->oplists.size() - 2; i++) {
        auto& op     = net->oplists[i];
        auto& lossOp = net->oplists[i + 2];

        originOpLists.emplace_back(std::move(op));

        std::string name = lossOp->name;
        if ("Loss" == name) {
            auto& compareOp      = net->oplists[i + 1];
            originTensorNameSize = compareOp->outputIndexes[0];
            break;
        }
    }

    net->oplists = std::move(originOpLists);

    // combine wight bias(conv)
    std::vector<int> deleteOpIndexes;
    {
        for (int i = 0; i < net->oplists.size(); i++) {
            auto& op = net->oplists[i];

            auto reductor = OpConverter::get(op->type);
            if (nullptr == reductor) {
                continue;
            }

            auto result = reductor->onReduct(i, op.get(), net.get());
            deleteOpIndexes.insert(deleteOpIndexes.end(), result.needDeleteOpIndexes.begin(),
                                   result.needDeleteOpIndexes.end());

            //            for (auto idx : result.needDeleteOpIndexes) {
            //                auto& op = net->oplists[idx];
            //                MNN_PRINT("%s, %d\n", op.get()->name.c_str(), idx);
            //            }
        }

        std::vector<std::unique_ptr<OpT>> newOpLists;
        for (int i = 0; i < net->oplists.size() - 1; i++) {
            auto iter = std::find(deleteOpIndexes.begin(), deleteOpIndexes.end(), i);
            if (iter == deleteOpIndexes.end()) {
                auto& op = net->oplists[i];
                newOpLists.emplace_back(std::move(op));
            }
        }

        net->oplists = std::move(newOpLists);
    }

    {
        const char* outputName = argv[2];
        FUNC_PRINT_ALL(outputName, s);
        net->tensorNumber = originTensorNameSize;

        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = Net::Pack(builder, net.get());
        builder.Finish(offset);
        ofstream os(outputName);
        os.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    }

    return 0;
}
