#include <cmath>
#include "AttentionConverter.hpp"
#include "core/TensorUtils.hpp"
#include "backend/NeuropilotBackend.hpp"
namespace MNN {
/**
 Q = _Reshape(Q, {batch, seqLength, kvNumHead,group, headDim});
 Q = _Transpose(Q, {0, 2, 3, 1, 4});
 K = _Reshape(K, {batch, seqLength, kvNumHead, 1, headDim});
 K = _Transpose(K, {0, 2, 3, 1, 4});

 auto scale = 1.0f / sqrtf(headDim);
 K = K * _Scalar<float>(scale);
 K.fix(VARP::CONSTANT);
 auto QK = _MatMul(Q, K, false, true); // [batch, kvNumHead, group , seq_len, seq_len]
 QK = QK + mask;
 auto QKPast = _MatMul(Q, cache.pastK, false, true);
 QKPast = QKPast + cache.pastMask;
 QK = _Concat({QKPast, QK}, -1);
 QK = _Softmax(QK, -1);
 V = _Reshape(V, {batch, seqLength, kvNumHead, 1, headDim});
 V = _Transpose(V, {0, 2, 3, 1, 4});
 V.fix(VARP::CONSTANT);
 auto totalV = _Concat({cache.pastV, V}, 3);
 auto QKV = _MatMul(QK, totalV, false, false);
 auto info = QKV->getInfo();
 auto O = _Transpose(QKV, {0, 3, 1, 2, 4});
 O = _Reshape(O, {batch, seqLength, -1});
 O.fix(VARP::CONSTANT);
 
 */
ConvertTflite::CommandBuffer AttentionConverter::onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) {
    auto kvMaxSize = root->pBackend->getRuntime()->hint().kvcacheSizeLimit;
    ConvertTflite::CommandBuffer res;
    auto Q = inputs[0];
    auto K = inputs[1];
    auto V = inputs[2];
    auto mask = inputs[3];
    auto seqLength = Q->length(1);
    auto numHead = Q->length(2);
    auto headDim = Q->length(3);
    auto kvNumHead = K->length(2);
    auto batch = Q->length(0);
    auto group = numHead / kvNumHead;
    auto scale = 1.0f / sqrtf(headDim);
    auto attn = op->main_as_AttentionParam();
    bool needState = false;
    if (nullptr != attn && attn->kv_cache()) {
        needState = true;
    }
    MNN_ASSERT(batch == 1);
    if (1 != batch) {
        MNN_ERROR("Don't support batch > 1 for mtk npu attention\n");
        return res;
    }

    //    Q = _Reshape(Q, {seqLength, kvNumHead,group, headDim});
    //    Q = _Transpose(Q, {1, 2, 0, 3});
    //    K = _Reshape(K, {batch, seqLength, kvNumHead, 1, headDim});
    //    K = _Transpose(K, {1, 2, 0, 3});
    
    Q = root->makeReshape(res, Q, {seqLength, kvNumHead,group, headDim});
    Q = root->makeTranspose(res, Q, {1, 2, 0, 3});
    Q = root->makeReshape(res, Q, {1, kvNumHead * group, seqLength, headDim});
    K = root->makeReshape(res, K, {seqLength, kvNumHead,1, headDim});
    K = root->makeTranspose(res, K, {2, 1, 0, 3}); // {1, kvNumHead, seqLength, headDim}
    Tensor* pastK = nullptr;
    Tensor* pastV = nullptr;
    Tensor* stateMask = nullptr;
    if (needState) {
        stateMask = root->pBackend->getStateMask(kvMaxSize);
        // Create pk, pv
        std::shared_ptr<Tensor> pastKWrap(Tensor::createDevice<float>({1, kvNumHead, kvMaxSize, headDim}));
        pastK = pastKWrap.get();
        root->pBackend->insertExtraInput(pastKWrap.get());
        res.extraConst.emplace_back(pastKWrap);
        std::shared_ptr<Tensor> pastVWrap(Tensor::createDevice<float>({1, kvNumHead, kvMaxSize, headDim}));
        pastV = pastVWrap.get();
        root->pBackend->insertExtraInput(pastVWrap.get());
        res.extraConst.emplace_back(pastVWrap);
    }
    // K = K * scale
    {
        std::shared_ptr<Tensor> scaleTensor(Tensor::create<float>({}));
        TensorUtils::getDescribe(scaleTensor.get())->usage = Tensor::InsideDescribe::Usage::CONSTANT;
        scaleTensor->host<float>()[0] = scale;
        res.extraConst.emplace_back(scaleTensor);
        K = root->makeBinary(res, K, scaleTensor.get(), tflite::BuiltinOperator_MUL);
        if (needState) {
            root->pBackend->insertExtraOutput(K);
        }
    }

    //  auto QK = _MatMul(Q, K, false, true); // [batch, kvNumHead, group , seq_len, seq_len]
    Tensor* QK = nullptr;
    {
        // Tile firstly
        if (1 != group) {
            K = root->makeTile(res, K, {1, 1, group, 1});
            K = root->makeReshape(res, K, {1, kvNumHead * group, seqLength, headDim});
        }
        // Matmul
        std::shared_ptr<Tensor> qktensor(Tensor::createDevice<float>({1, kvNumHead * group, seqLength, seqLength}));
        root->makeMatMul(res, Q, K, false, true, qktensor.get());
        QK = qktensor.get();
        res.extraConst.emplace_back(qktensor);
    }
    //    QK = QK + mask;
    if (seqLength != 1) {
        // For decode don't need mask, if add mask will cause crash for neuropilot
        QK = root->makeBinary(res, QK, mask, tflite::BuiltinOperator_ADD);
    }
    if (needState) {
        // auto QKPast = _MatMul(Q, pastK, false, true);
        // Tile firstly
        if (1 != group) {
            pastK = root->makeTile(res, pastK, {1, 1, group, 1});
            pastK = root->makeReshape(res, pastK, {1, kvNumHead * group, kvMaxSize, headDim});
        }
        // Matmul
        std::shared_ptr<Tensor> qktensor(Tensor::createDevice<float>({1, kvNumHead * group, seqLength, kvMaxSize}));
        root->makeMatMul(res, Q, pastK, false, true, qktensor.get());
        res.extraConst.emplace_back(qktensor);
        auto QKPastMask = root->makeBinary(res, qktensor.get(), stateMask, tflite::BuiltinOperator_ADD);
        QK = root->makeConcat(res, {QKPastMask, QK}, 3);
    }
//    QK = _Softmax(QK, -1);
    {
        QK = root->makeSoftmax(res, QK);
    }
//    V = _Reshape(V, {seqLength, kvNumHead, 1, headDim});
//    V = _Transpose(V, {1, 2, 0, 3});
    V = root->makeReshape(res, V, {seqLength, kvNumHead,1, headDim});
    V = root->makeTranspose(res, V, {2, 1, 0, 3}); // 1, kvNumHead, seqLength, headDim
    if (needState) {
        root->pBackend->insertExtraOutput(V);
        V = root->makeConcat(res, {pastV, V}, 2);
    }

//    auto QKV = _MatMul(QK, V, false, false);
    Tensor* QKV = nullptr;
    {
        // Tile firstly
        if (1 != group) {
            V = root->makeTile(res, V, {1, 1, group, 1});
            V = root->makeReshape(res, V, {1, kvNumHead * group, seqLength + kvMaxSize, headDim});
        }
        // Matmul
        std::shared_ptr<Tensor> qkvtensor(Tensor::createDevice<float>({1, kvNumHead * group, seqLength, headDim}));
        root->makeMatMul(res, QK, V, false, false, qkvtensor.get());
        QKV = qkvtensor.get();
        res.extraConst.emplace_back(qkvtensor);
    }
//    auto O = _Transpose(QKV, {0, 3, 1, 2, 4});
//    O = _Reshape(O, {batch, seqLength, -1});
    auto O = root->makeTranspose(res, QKV, {0, 2, 1, 3});
    root->makeReshape(res, O, {1, seqLength, -1}, outputs[0]);
    return res;
}

};
