//
//  TransformShuffleChannel.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace MNN;
class TransformShuffleChannel : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto op = iter->get();
            if (op->type == OpType_Plugin) {
                auto plugin = op->main.AsPlugin();
                if (plugin->type == "ShuffleChannel") {
                    int currentTensorCount = (int)net->tensorName.size();
                    std::unique_ptr<OpT> convertTo(new OpT);
                    convertTo->type                               = OpType_ConvertTensor;
                    convertTo->main.type                          = OpParameter_TensorConvertInfo;
                    convertTo->main.value                         = new TensorConvertInfoT;
                    convertTo->main.AsTensorConvertInfo()->source = MNN_DATA_FORMAT_NC4HW4;
                    convertTo->main.AsTensorConvertInfo()->dest   = MNN_DATA_FORMAT_NHWC;
                    convertTo->inputIndexes                       = op->inputIndexes;
                    convertTo->name                               = op->name + "_ConvertToNHWC";
                    convertTo->outputIndexes                      = {currentTensorCount + 0};
                    net->tensorName.emplace_back(convertTo->name);

                    auto group = plugin->attr[0]->tensor->int32s[0];
                    std::unique_ptr<OpT> reshape(new OpT);
                    reshape->type                      = OpType_Reshape;
                    reshape->name                      = op->name + "_Reshape";
                    reshape->main.value                = new ReshapeT;
                    reshape->main.type                 = OpParameter_Reshape;
                    reshape->main.AsReshape()->dimType = MNN_DATA_FORMAT_NHWC;
                    reshape->main.AsReshape()->dims    = {0, 0, 0, group, -1};
                    reshape->inputIndexes              = {currentTensorCount + 0};
                    reshape->outputIndexes             = {currentTensorCount + 1};
                    net->tensorName.emplace_back(reshape->name);

                    std::unique_ptr<OpT> constOp(new OpT);
                    constOp->type          = OpType_Const;
                    auto blob              = new BlobT;
                    blob->int32s           = {0, 1, 2, 4, 3};
                    blob->dataFormat       = MNN_DATA_FORMAT_NHWC;
                    blob->dataType         = DataType_DT_INT32;
                    blob->dims             = {5};
                    constOp->main.value    = blob;
                    constOp->main.type     = OpParameter_Blob;
                    constOp->name          = op->name + "_Const";
                    constOp->outputIndexes = {currentTensorCount + 2};
                    net->tensorName.emplace_back(constOp->name);

                    std::unique_ptr<OpT> permute(new OpT);
                    permute->type                      = OpType_Transpose;
                    permute->name                      = op->name + "_Transpose";
                    permute->main.value                = new TransposeT;
                    permute->main.type                 = OpParameter_Transpose;
                    permute->main.AsTranspose()->Tperm = DataType_DT_INT32;
                    permute->inputIndexes              = {currentTensorCount + 1, currentTensorCount + 2};
                    permute->outputIndexes             = {currentTensorCount + 3};
                    net->tensorName.emplace_back(permute->name);

                    std::unique_ptr<OpT> reshapeR(new OpT);
                    reshapeR->type                      = OpType_Reshape;
                    reshapeR->name                      = op->name + "_ReshapeR";
                    reshapeR->main.value                = new ReshapeT;
                    reshapeR->main.type                 = OpParameter_Reshape;
                    reshapeR->main.AsReshape()->dimType = MNN_DATA_FORMAT_NHWC;
                    reshapeR->main.AsReshape()->dims    = {0, 0, 0, -1};
                    reshapeR->inputIndexes              = {currentTensorCount + 3};
                    reshapeR->outputIndexes             = {currentTensorCount + 4};
                    net->tensorName.emplace_back(reshapeR->name);

                    std::unique_ptr<OpT> convertFrom(new OpT);
                    convertFrom->type                               = OpType_ConvertTensor;
                    convertFrom->main.type                          = OpParameter_TensorConvertInfo;
                    convertFrom->main.value                         = new TensorConvertInfoT;
                    convertFrom->main.AsTensorConvertInfo()->source = MNN_DATA_FORMAT_NHWC;
                    convertFrom->main.AsTensorConvertInfo()->dest   = MNN_DATA_FORMAT_NC4HW4;
                    convertFrom->inputIndexes                       = {currentTensorCount + 4};
                    convertFrom->outputIndexes                      = op->outputIndexes;
                    convertFrom->name                               = op->name + "_ConvertToNC4HW4";

                    iter = net->oplists.erase(iter);
                    iter = net->oplists.insert(iter, std::move(convertFrom));
                    iter = net->oplists.insert(iter, std::move(reshapeR));
                    iter = net->oplists.insert(iter, std::move(permute));
                    iter = net->oplists.insert(iter, std::move(constOp));
                    iter = net->oplists.insert(iter, std::move(reshape));
                    iter = net->oplists.insert(iter, std::move(convertTo));
                    iter++;
                    iter++;
                    iter++;
                    iter++;
                    continue;
                }
            }
            iter++;
        }
        return true;
    }
};
static PostConverterRegister<TransformShuffleChannel> __l("TransformShuffleChannel");
