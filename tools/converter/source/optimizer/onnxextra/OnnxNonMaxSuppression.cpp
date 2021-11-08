//
//  OnnxNonMaxSuppression.cpp
//  MNNConverter
//
//  Created by MNN on 2020/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <flatbuffers/util.h>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxNonMaxSuppressionTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);

        VARP result(nullptr);
        // NonMaxSuppression(boxes, scores, max_output_size, iou_threshold, score_threshold)
        // boxes and scores are required, the last 3 parameters are optional.
        // onnx boxes is 3D [num_batches, boxes_num, 4] with num_batches = 1, while tf boxes
        // is 2D [boxes_num, 4].
        // onnx scores is 3D [num_batches, num_classes, boxes_num] with num_batches = 1,
        // while tf scores is 1D [boxes_num].
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() >= 2);
        auto input0Info = inputs[0]->getInfo();
        auto input1Info = inputs[1]->getInfo();
        if (nullptr == input0Info || nullptr == input1Info) {
            return nullptr;
        }
        MNN_ASSERT(inputs[0]->getInfo()->dim.size() == 3);
        MNN_ASSERT(inputs[1]->getInfo()->dim.size() == 3);

        for (int batch = 0; batch < inputs[0]->getInfo()->dim[0]; ++batch) {
            VARP boxes  = _Gather(inputs[0], _Scalar<int>(batch)); // [boxes_num, 4]
            VARP scores = _Gather(inputs[1], _Scalar<int>(batch)); // [num_classes, boxes_num]

            int num_classes = scores->getInfo()->dim[0];
            for (int cls = 0; cls < num_classes; ++cls) {
                VARP scores_per_class = _Gather(scores, _Scalar<int>(cls)); // [boxes_num]

                std::unique_ptr<MNN::OpT> nonMaxSuppressionOp(new OpT);
                std::string name                = op->name()->str() + "/" + flatbuffers::NumToString(cls);
                nonMaxSuppressionOp->name       = name;
                nonMaxSuppressionOp->type       = OpType_NonMaxSuppressionV2;
                nonMaxSuppressionOp->main.type  = OpParameter_NonMaxSuppressionV2;
                nonMaxSuppressionOp->main.value = nullptr;

                std::vector<VARP> newInputs{boxes, scores_per_class};
                for (int i = 2; i < inputs.size(); ++i) {
                    newInputs.push_back(inputs[i]);
                }
                auto nonMaxSupp = Expr::create(nonMaxSuppressionOp.get(), newInputs, 1 /*output size*/);
                nonMaxSupp->setName(expr->name() + "/" + flatbuffers::NumToString(cls));

                // Tensorflow's output is [num_selected_boxes], while onnx requires
                // [num_selected_boxes, 3], and the meaning of last dim is
                // [batch_index, class_index, box_index].
                VARP output = _Unsqueeze(Variable::create(nonMaxSupp), {0}); // [1, num_selected_boxes]
                output      = _Transpose(output, {1, 0});                    // [num_selected_boxes, 1]
                std::vector<int> paddings{0, 0, 1, 0};
                auto data_type = halide_type_of<int32_t>();
                VARP pad       = _Const(paddings.data(), {2, 2}, NCHW, data_type);

                // TODO(houjiang): Support padding with non-zero constant value.
                output = _Pad(_Pad(output, pad /*, cls*/), pad /*, batch*/); // [num_selected_boxes, 3]
                if (result.get() != nullptr) {
                    result = _Concat({result, output}, 0);
                } else {
                    result = output;
                }
            }
        }

        return result->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("NonMaxSuppression",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxNonMaxSuppressionTransformer));
    return true;
}();

} // namespace Express
} // namespace MNN
