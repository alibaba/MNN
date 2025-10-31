#include "MTKEXT.hpp"
#include "core/TensorUtils.hpp"
#include "flatbuffers/flexbuffers.h"
namespace MNN {
ConvertTflite::CommandBuffer MTKEXT::onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) {
    ConvertTflite::CommandBuffer res;
    if (op->type() == OpType_LayerNorm) {
        res.op = op;
        ConvertTflite::Command cmd;
        cmd.op.reset(new tflite::OperatorT());
        auto layernorm = op->main_as_LayerNorm();
        cmd.inputs = inputs;
        cmd.outputs = outputs;
        if (!layernorm->useRMSNorm()) {
            MNN_ERROR("Don't support not rms norm\n");
        }
        cmd.op->opcode_index = root->getCustomOpIndex("MTKEXT_RMS_NORMALIZATION");
        flexbuffers::Builder builder;
        auto start = builder.StartMap();
        builder.Float("epsilon", layernorm->epsilon());
        builder.EndMap(start);
        builder.Finish();
        cmd.op->custom_options = builder.GetBuffer();
        std::vector<int> axises;
        if (nullptr != layernorm->axis()) {
            axises.resize(layernorm->axis()->size());
            ::memcpy(axises.data(), layernorm->axis()->data(), layernorm->axis()->size() * sizeof(int));
            for (int i=0; i<axises.size(); ++i) {
                if (axises[i] < 0) {
                    axises[i] = inputs[0]->dimensions() + axises[i];
                }
            }
        }
        auto axisTensor = ConvertTflite::getIntArrayTensor(axises);
        bool hasGammaBeta = (layernorm->gamma() && layernorm->beta());
        int gammasize = 0;
        if (hasGammaBeta) {
            MNN_ASSERT(layernorm->gamma()->size() == layernorm->beta()->size());
            gammasize = layernorm->gamma()->size();
        }
        std::shared_ptr<MNN::Tensor> gamma;
        std::shared_ptr<MNN::Tensor> beta;
        if (hasGammaBeta) {
            // Use uint8_t to avoid lowp reduce float bytes
            gamma.reset(Tensor::create<float>({gammasize}));
            beta.reset(Tensor::create<float>({gammasize}));
            cmd.inputs = {inputs[0], axisTensor.get(), gamma.get(), beta.get()};
            TensorUtils::getDescribe(gamma.get())->usage = Tensor::InsideDescribe::CONSTANT;
            TensorUtils::getDescribe(beta.get())->usage = Tensor::InsideDescribe::CONSTANT;
            const float* gamma_data = layernorm->gamma()->data();
            memcpy(gamma->host<float>(), gamma_data, gammasize * sizeof(float));
            const float* beta_data = layernorm->beta()->data();
            memcpy(beta->host<float>(), beta_data, gammasize * sizeof(float));
            res.extraConst.emplace_back(gamma);
            res.extraConst.emplace_back(beta);
        } else {
            cmd.inputs = {inputs[0], axisTensor.get()};
        }
        res.extraConst.emplace_back(axisTensor);
        res.commands.emplace_back(std::move(cmd));
    }
    return res;
}

};
