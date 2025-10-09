//
//  RemoveTestNoUseOps.hpp
//  MNNConverter
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include "../PostTreatUtils.hpp"
#include "MNN/MNNDefine.h"
using namespace MNN;

class RemoveTestNoUseOps : public PostConverter {
public:
    /* The Op's output set as input */
    virtual bool shouldDeleteJudge(const MNN::OpT* op, const MNN::NetT* const netPtr) const = 0;

    virtual bool shouldRemoveUnusefulInputs(const MNN::OpT* op) const = 0;

    virtual bool shouldDeleteOutput(const MNN::OpT* op) const = 0;

    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override;
};
