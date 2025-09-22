#include "QNNCast.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

static DataType _mapDataType(DataType src) {
    if (DataType_DT_BOOL == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_INT64 == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_DOUBLE == src) {
        return DataType_DT_FLOAT;
    }
    return src;
}

ErrorCode QNNCast::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);

    mNodeType = "Cast";

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

class QNNCastCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
    auto cast = op->main_as_CastParam();

    // cast param srcT is invalid
    // auto srcT = _mapDataType(cast->srcT());
    auto dstT = _mapDataType(cast->dstT());

    const auto &inputDataType = inputs[0]->getType();

    bool flag0 = ((halide_type_of<int32_t>() == inputDataType) || (halide_type_of<float>() == inputDataType));
    bool flag1 = ((dstT == MNN::DataType_DT_FLOAT) || (dstT == MNN::DataType_DT_INT32));

    // Currently, only support int and float cast.
    if (!flag0 || !flag1) {
        MNN_ERROR("Cast src:%d dst:%d\n", inputDataType, dstT);
        return nullptr;
    }

    if (inputs.size() > 1) {
        MNN_ERROR("QNN Cast inputs size: %d, error\n", inputs.size());
        return nullptr;
    }

    return new QNNCast(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNCastCreator, OpType_Cast)
#endif
} // end namespace QNN
} // end namespace MNN
