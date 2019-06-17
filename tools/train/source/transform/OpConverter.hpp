//
//  OpConverter.hpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpConverter_hpp
#define OpConverter_hpp
#include "MNNDefine.h"
#include "converter/source/IR/MNN_generated.h"

class MNN_PUBLIC OpConverter {
public:
    struct Result {
        std::vector<std::unique_ptr<MNN::OpT>> opLists;
        std::vector<std::string> tensorNames;
        int newTensorOffset = 1000;
    };
    OpConverter() = default;

    virtual ~OpConverter()                                             = default;
    virtual Result onConvert(const MNN::OpT* op, const MNN::NetT* net) = 0;

    static OpConverter* get(MNN::OpType type);
    static void insert(MNN::OpType type, OpConverter* converter);

    struct ReductResult {
        std::vector<int> needDeleteOpIndexes;
    };
    virtual ReductResult onReduct(int opIndex, MNN::OpT* op, MNN::NetT* net) = 0;

    static void merge(MNN::NetT* net, OpConverter::Result& result);
};
#endif
