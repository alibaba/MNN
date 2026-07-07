//
//  RopeBufExecution.cpp
//  MNN
//
//  OpenCL buffer-path implementation of RoPE (Rotary Positional Embedding).
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "RopeBufExecution.hpp"
#include "MNN_generated.h"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

static std::shared_ptr<cl::Buffer> makeRopeNormGamma(OpenCLBackend* backend, const LayerNorm* layerNorm) {
    if (nullptr == layerNorm || nullptr == layerNorm->gamma()) {
        return nullptr;
    }
    int size = layerNorm->gamma()->size();
    if (size <= 0) {
        return nullptr;
    }
    std::shared_ptr<cl::Buffer> gamma(new cl::Buffer(backend->getOpenCLRuntime()->context(),
                                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                     ALIGN_UP4(size) * sizeof(float)));
    auto error = CL_SUCCESS;
    auto ptr = backend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        *gamma, true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * sizeof(float), nullptr, nullptr, &error);
    if (ptr == nullptr || error != CL_SUCCESS) {
        return nullptr;
    }
    ::memset(ptr, 0, ALIGN_UP4(size) * sizeof(float));
    ::memcpy(ptr, layerNorm->gamma()->data(), size * sizeof(float));
    backend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*gamma, ptr);
    return gamma;
}

static bool validRopeC4Input(const Tensor* q, const Tensor* k, int numHead, int kvNumHead, int headDim) {
    if (q == nullptr || k == nullptr || numHead <= 0 || kvNumHead <= 0 || headDim <= 0) {
        return false;
    }
    if (TensorUtils::getDescribe(q)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
        TensorUtils::getDescribe(k)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return false;
    }
    if (q->dimensions() < 2 || k->dimensions() < 2) {
        return false;
    }
    return q->length(1) == numHead * headDim && k->length(1) == kvNumHead * headDim;
}

RopeBufExecution::RopeBufExecution(const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);

    auto param = op == nullptr ? nullptr : op->main_as_RoPEParam();
    if (param != nullptr) {
        mRopeCutHeadDim = param->rope_cut_head_dim();
        mNumHead = param->num_head();
        mKvNumHead = param->kv_num_head();
        mHeadDim = param->head_dim();
        auto qNorm = param->q_norm();
        auto kNorm = param->k_norm();
        if (qNorm != nullptr) {
            mQEps = qNorm->epsilon();
            mQGamma = makeRopeNormGamma(mOpenCLBackend, qNorm);
        }
        if (kNorm != nullptr) {
            mKEps = kNorm->epsilon();
            mKGamma = makeRopeNormGamma(mOpenCLBackend, kNorm);
        }
    }
}

RopeBufExecution::RopeBufExecution(const MNN::Op* op, Backend* backend, int ropeCutHeadDim, int numHead, int kvNumHead,
                                   int headDim, std::shared_ptr<cl::Buffer> qGamma, float qEps,
                                   std::shared_ptr<cl::Buffer> kGamma, float kEps)
    : CommonExecution(backend, op),
      mRopeCutHeadDim(ropeCutHeadDim),
      mNumHead(numHead),
      mKvNumHead(kvNumHead),
      mHeadDim(headDim),
      mQGamma(qGamma),
      mKGamma(kGamma),
      mQEps(qEps),
      mKEps(kEps) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
}

bool RopeBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst =
        new RopeBufExecution(op, bn, mRopeCutHeadDim, mNumHead, mKvNumHead, mHeadDim, mQGamma, mQEps, mKGamma, mKEps);
    return true;
}

ErrorCode RopeBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 4);
    MNN_ASSERT(outputs.size() == 2);

    auto q = inputs[0];
    auto k = inputs[1];
    if (!validRopeC4Input(q, k, mNumHead, mKvNumHead, mHeadDim)) {
        MNN_ERROR("RopeBufExecution: invalid C4 input, numHead=%d, kvNumHead=%d, headDim=%d.\n", mNumHead, mKvNumHead,
                  mHeadDim);
        return NOT_SUPPORT;
    }

    int batch = 1;
    int seqLen = q->length(0);
    int numHead = mNumHead;
    int headDim = mHeadDim;
    int kvNumHead = mKvNumHead;

    int halfD = headDim / 2;
    int ropeDim = mRopeCutHeadDim;
    if (ropeDim <= 0 || ropeDim > headDim) {
        ropeDim = headDim;
    }
    ropeDim = (ropeDim / 2) * 2;
    int ropeHalfD = ropeDim / 2;
    if (ropeHalfD > halfD) {
        ropeHalfD = halfD;
    }

    int outerSize = batch * seqLen;
    int fullHead = numHead + kvNumHead;

    mUnits.resize(1);
    auto& unit = mUnits[0];

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    std::set<std::string> buildOptions;
    if (mQGamma) {
        buildOptions.emplace("-DQ_NORM");
    }
    if (mKGamma) {
        buildOptions.emplace("-DK_NORM");
    }
    unit.kernel = runtime->buildKernel("rope_buf", "rope_buf", buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(unit.kernel);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    if (mQGamma || mKGamma) {
        mGlobalWorkSize = {1, static_cast<uint32_t>(outerSize), static_cast<uint32_t>(fullHead)};
    } else {
        mGlobalWorkSize = {static_cast<uint32_t>(halfD), static_cast<uint32_t>(outerSize),
                           static_cast<uint32_t>(fullHead)};
    }

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[0]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[1]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[2]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[3]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[0]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[1]));
    ret |= unit.kernel->get().setArg(idx++, outerSize);
    ret |= unit.kernel->get().setArg(idx++, halfD);
    ret |= unit.kernel->get().setArg(idx++, ropeHalfD);
    ret |= unit.kernel->get().setArg(idx++, headDim);
    ret |= unit.kernel->get().setArg(idx++, numHead);
    ret |= unit.kernel->get().setArg(idx++, kvNumHead);
    if (mQGamma) {
        ret |= unit.kernel->get().setArg(idx++, *mQGamma);
        ret |= unit.kernel->get().setArg(idx++, mQEps);
    }
    if (mKGamma) {
        ret |= unit.kernel->get().setArg(idx++, *mKGamma);
        ret |= unit.kernel->get().setArg(idx++, mKEps);
    }
    MNN_CHECK_CL_SUCCESS(ret, "setArg RopeBufExecution");

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runtime, "rope_buf", unit.kernel,
                                      mOpenCLBackend->getCLTuneLevel(), "rope_buf")
                         .first;

    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};

    return NO_ERROR;
}

class RopeBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new RopeBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(RopeBufCreator, OpType_RoPE, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
