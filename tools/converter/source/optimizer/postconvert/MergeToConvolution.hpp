//
//  MergeToConvolution.hpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <set>
#include "../PostTreatUtils.hpp"
using namespace MNN;

class MergeToConvolution : public PostConverter {
public:
    virtual bool merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const = 0;

    virtual bool merge2Convolution3D(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const = 0;

    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        // Merge Layer
        std::vector<MNN::OpT*> readyToDelete;
        std::set<std::string> outputNames;
        for (auto n : net->outputName) {
            outputNames.insert(n);
        }
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            MNN::OpT& currentOp = *(iter->get());
            if (currentOp.type != MNN::OpType_Convolution
                && currentOp.type != MNN::OpType_Deconvolution
                && currentOp.type != MNN::OpType_ConvolutionDepthwise
                && currentOp.type != MNN::OpType_Convolution3D) {
                continue;
            }
            DCHECK(currentOp.outputIndexes.size() == 1) << "Conv output ERROR!";
            if (outputNames.find(net->tensorName[currentOp.outputIndexes[0]]) != outputNames.end()) {
                continue;
            }

            // merge Batchnorm/Relu/Relu6 to Convolution
            std::vector<MNN::OpT*> nextOp = PostTreatUtils::_findOpByInputIndex(currentOp.outputIndexes[0], net.get());
            while (1) {
                if (nextOp.size() != 1) {
                    break;
                }
                const int nextOutputIndex = nextOp[0]->outputIndexes[0];
                bool succ;
                if (currentOp.type == MNN::OpType_Convolution3D) {
                    succ = merge2Convolution3D(nextOp[0], &currentOp);
                } else {
                    succ = merge2Convolution(nextOp[0], &currentOp);
                }
                if (PostTreatUtils::_isSingleInputOutput(nextOp[0]) && succ) {
                    // LOG(INFO) << "Merge " << nextOp[0]->name.c_str()<< " into convolution: " <<
                    // currentOp.name.c_str();
                    currentOp.outputIndexes[0] = nextOp[0]->outputIndexes[0];
                    readyToDelete.push_back(nextOp[0]);
                    nextOp = PostTreatUtils::_findOpByInputIndex(nextOutputIndex, net.get());
                } else {
                    break;
                }
            }
        }
        for (auto op : readyToDelete) {
            PostTreatUtils::_removeOpInNet(op, net.get());
        }
        return true;
    }
};
