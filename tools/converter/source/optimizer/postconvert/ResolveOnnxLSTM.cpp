//
//  ResolveOnnxLSTM.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"

class ResolveOnnxLSTM : public PostConverter{
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT> &net) const override{
        if(net->sourceType != MNN::NetSource_ONNX){
            return true;
        }
        
        std::set<MNN::OpT*> readyToDelete;
        const int size = net->oplists.size();
        for(int i = 0; i < size; ++i){
            auto& op = net->oplists[i];
            if(op->type != MNN::OpType_LSTM) continue;
            
            const int outputIndex = op->outputIndexes[0];
            auto referenceLstmOps = PostTreatUtils::_findOpByInputIndex(outputIndex, net.get());
            DCHECK(referenceLstmOps.size() == 1) << "TODO ==> support biLSTM!";
            
            if(referenceLstmOps[0]->type != MNN::OpType_Squeeze) continue;
            
            // change Squeeze axis 1 to be 2
            auto squeezeParam = referenceLstmOps[0]->main.AsSqueezeParam();
            squeezeParam->squeezeDims.clear();
            squeezeParam->squeezeDims.push_back(2);
            
            auto squeezeOutIndex = referenceLstmOps[0]->outputIndexes[0];
            auto referenceSqueezeOps = PostTreatUtils::_findOpByInputIndex(squeezeOutIndex, net.get());
            DCHECK(referenceSqueezeOps.size() == 1) << "size should be 1";
            
            // referenceSqueezeOps(size == 1) is Transpose or LSTM
            const int oldIndex = referenceSqueezeOps[0]->outputIndexes[0];
            auto referenceTransposeOps = PostTreatUtils::_findOpByInputIndex(oldIndex, net.get());
            for(auto op : referenceTransposeOps){
                DCHECK(PostTreatUtils::_replace(op->inputIndexes, squeezeOutIndex, oldIndex)) << "index error!";
            }
            if(referenceSqueezeOps[0]->type == MNN::OpType_Permute){
                readyToDelete.insert(referenceSqueezeOps[0]);
            }
            
        }
        
        for (auto op : readyToDelete) {
            PostTreatUtils::_removeOpInNet(op, net.get());
        }
        
        return true;
    }
};

static PostConverterRegister<ResolveOnnxLSTM> __rlstm("ResolveOnnxLSTM");
