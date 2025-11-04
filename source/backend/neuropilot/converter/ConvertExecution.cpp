#include "ConvertExecution.hpp"
#include "backend/NeuropilotBackend.hpp"
namespace MNN {
ConvertExecution::~ConvertExecution() {
    if (nullptr != mHostPtr) {
        delete mHostPtr;
    }
}
ConvertExecution::ConvertExecution(Backend* bn, const Op* op) : Execution(bn) {
    mOp = op;
    if (op->main_type() == OpParameter_LayerNorm || op->main_type() == OpParameter_Scale) {
        // Copy Op because it may has exteranl
        std::unique_ptr<OpT> opt(op->UnPack());
        flatbuffers::FlatBufferBuilder builder;
        size_t size, offset;
        builder.Finish(Op::Pack(builder, opt.get()));
        mHostPtr = builder.ReleaseRaw(size, offset);
        mOp = flatbuffers::GetRoot<Op>((mHostPtr+offset));
    }
}

ErrorCode ConvertExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    NeuropilotBackend::ExecuteInfo info;
    info.op = mOp;
    info.inputs = inputs;
    info.outputs = outputs;
    static_cast<NeuropilotBackend*>(backend())->mInfos.emplace_back(std::move(info));
    return NO_ERROR;
}
};
