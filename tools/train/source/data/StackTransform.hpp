//
//  StackTransform.hpp
//  MNN
//
//  Created by MNN on 2019/11/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef StackTransform_hpp
#define StackTransform_hpp

#include <MNN/expr/ExprCreator.hpp>
#include "Transform.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC StackTransform : public BatchTransform {
    std::vector<Example> transformBatch(std::vector<Example> batch) override {
        std::vector<std::vector<VARP>> batchData(batch[0].first.size());
        std::vector<std::vector<VARP>> batchTarget(batch[0].second.size());
        for (int i = 0; i < batch.size(); i++) {
            for (int j = 0; j < batchData.size(); j++) {
                batchData[j].emplace_back(batch[i].first[j]);
            }
        }

        for (int i = 0; i < batch.size(); i++) {
            for (int j = 0; j < batchTarget.size(); j++) {
                batchTarget[j].emplace_back(batch[i].second[j]);
            }
        }

        Example example;
        for (int i = 0; i < batchData.size(); i++) {
            example.first.emplace_back(_Stack(batchData[i], 0));
        }
        for (int i = 0; i < batchTarget.size(); i++) {
            example.second.emplace_back(_Stack(batchTarget[i], 0));
        }

        return {example};
    }
};

} // namespace Train
} // namespace MNN

#endif // StackTransform_hpp
