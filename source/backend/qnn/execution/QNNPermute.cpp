#include "QNNPermute.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNPermute::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor * input = inputs[0];
    int dim = input->dimensions();
    Tensor::DimensionType inputDimType = inputs[0]->getDimensionType();
    Tensor::DimensionType outputDimType = outputs[0]->getDimensionType();
    MNN_ASSERT(inputDimType == outputDimType);

    #ifdef QNN_VERBOSE
    MNN_PRINT("QNN Permute: %s input0:", mNodeName.c_str());
    auto shape0 = inputs[0]->shape();
    for(int i = 0; i < shape0.size(); i++) {
        MNN_PRINT("%d x ", shape0[i]);
    }

    MNN_PRINT("\noutput:");
    auto outShape = outputs[0]->shape();
    for(int i = 0; i < outShape.size(); i++) {
        MNN_PRINT("%d x ", outShape[i]);
    }
    MNN_PRINT("\n");

    #endif

    mNodeType = "Transpose";
    std::vector<uint32_t> mapRaw(dim, 0);
    if (mOp->type() == OpType_Permute) {
        auto param = mOp->main_as_Permute();
        auto axis = param->dims();
        int size = (int) param->dims()->size();
        MNN_ASSERT(size == dim);
        for (int i = 0; i < dim; i++) {
            int index = axis->Get(i);
            mapRaw[i] = (uint32_t)index;
        }
    } else {
        auto permutation = inputs[1]->host<int32_t>();
        if (permutation) {
            for (int i = 0; i < dim; i++) {
                mapRaw[i] = (uint32_t)permutation[i];
            }
        } else {
            for (int i = 0; i < dim; i++) {
                mapRaw[i] = (uint32_t)i;
            }
        }
    }

    // When tensor is NC4HW4, QNN sees NHWC layout via getNHWCShape.
    // The permutation from MNN is in NCHW order and must be remapped.
    // NCHW dim mapping to NHWC: 0->0, 1->dim-1, k(2..dim-1)->k-1
    auto dataFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    if (dataFormat == MNN_DATA_FORMAT_NC4HW4 && dim > 2) {
        // nchw2nhwc[i] = where NCHW dim i appears in NHWC
        std::vector<int> nchw2nhwc(dim);
        nchw2nhwc[0] = 0;
        nchw2nhwc[1] = dim - 1;
        for (int i = 2; i < dim; i++) {
            nchw2nhwc[i] = i - 1;
        }
        // nhwc2nchw is the inverse
        std::vector<int> nhwc2nchw(dim);
        for (int i = 0; i < dim; i++) {
            nhwc2nchw[nchw2nhwc[i]] = i;
        }
        // Convert: nhwcPerm[nhwcPos] = nchw2nhwc[nchwPerm[nhwc2nchw[nhwcPos]]]
        std::vector<uint32_t> nhwcPerm(dim);
        for (int i = 0; i < dim; i++) {
            int ncPos = nhwc2nchw[i];
            int ncDst = (int)mapRaw[ncPos];
            nhwcPerm[i] = (uint32_t)nchw2nhwc[ncDst];
        }
        mapRaw = nhwcPerm;
    }

    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) dim}, mapRaw.data());
    this->addNodeCommon(inputs, outputs, 1, 1);
    return NO_ERROR;
}

class QNNPermuteCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNPermute(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNPermuteCreator, OpType_Permute)
REGISTER_QNN_OP_CREATOR(QNNPermuteCreator, OpType_Transpose)
#endif
} // end namespace QNN
} // end namespace MNN
