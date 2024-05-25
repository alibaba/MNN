//
//  ReIndexTensor.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <set>
#include <sstream>
#include "../PostTreatUtils.hpp"
using namespace MNN;
class ReIndexTensor : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& mNet = net;
        std::map<std::string, int> tensorNameIdx;
        std::map<int, int> usefulTensorIndexMap;
        std::vector<std::string> usefulTensorName;
        // extraTensorDescribe reindex
        for (int i = 0; i < mNet->tensorName.size(); i++) {
            tensorNameIdx.insert(std::make_pair(mNet->tensorName[i], i));
        }
        for (int i = 0; i < mNet->extraTensorDescribe.size(); i++) {
            auto name = mNet->extraTensorDescribe[i]->name;
            auto iter = tensorNameIdx.find(name);
            if (iter == tensorNameIdx.end()) {
                mNet->extraTensorDescribe[i]->index = -1;
            } else {
                mNet->extraTensorDescribe[i]->index = iter->second;
            }
        }

        std::vector<bool> tensorValid(mNet->tensorName.size(), false);
        for (auto& op : mNet->oplists) {
            for (auto index : op->inputIndexes) {
                if (index < 0) {
                    continue; // optional input, ignore it
                }
                tensorValid[index] = true;
            }
            for (auto index : op->outputIndexes) {
                tensorValid[index] = true;
            }
        }

        for (int i = 0; i < tensorValid.size(); ++i) {
            if (tensorValid[i]) {
                usefulTensorIndexMap.insert(std::make_pair(i, usefulTensorName.size()));
                usefulTensorName.push_back(mNet->tensorName[i]);
            }
        }

        // Re index
        for (auto& op : mNet->oplists) {
            for (int i = 0; i < op->inputIndexes.size(); ++i) {
                if (op->inputIndexes[i] < 0) {
                    continue;
                }
                auto iter = usefulTensorIndexMap.find(op->inputIndexes[i]);
                DCHECK(iter != usefulTensorIndexMap.end()) << "ERROR";
                op->inputIndexes[i] = iter->second;
            }
            for (int i = 0; i < op->outputIndexes.size(); ++i) {
                auto iter = usefulTensorIndexMap.find(op->outputIndexes[i]);
                DCHECK(iter != usefulTensorIndexMap.end()) << "ERROR";
                op->outputIndexes[i] = iter->second;
            }
        }

        mNet->tensorName = usefulTensorName;
        for (auto iter = mNet->extraTensorDescribe.begin(); iter != mNet->extraTensorDescribe.end();) {
            auto index = (*iter)->index;
            if (usefulTensorIndexMap.find(index) == usefulTensorIndexMap.end()) {
                iter = mNet->extraTensorDescribe.erase(iter);
                continue;
            }
            (*iter)->index = usefulTensorIndexMap.find(index)->second;
            iter++;
        }
        // Check dup name and modify
        std::set<std::string> names;
        std::set<std::string> tensorNames;
        for (int i = 0; i < mNet->oplists.size(); ++i) {
            auto& op    = mNet->oplists[i];
            auto opName = op->name;
            if (opName.empty() || names.find(opName) != names.end()) {
                std::ostringstream defaultName;
                defaultName << EnumNameOpType(op->type);
                defaultName << i;
                op->name = defaultName.str();
#ifdef DEBUG
                MNN_PRINT("%d op name is empty or dup, set to %s\n", i, op->name.c_str());
#endif
                opName = op->name;
            }
            names.insert(opName);
            for (auto output : op->outputIndexes) {
                auto origin = net->tensorName[output];
                if (origin.empty() || tensorNames.find(origin) != tensorNames.end()) {
                    std::ostringstream defaultName;
                    defaultName << output;
                    origin                  = defaultName.str();
                    net->tensorName[output] = origin;
                }
                tensorNames.insert(origin);
            }
        }
        return true;
    }
};
static PostConverterRegister<ReIndexTensor> __l("ReIndexTensor");
