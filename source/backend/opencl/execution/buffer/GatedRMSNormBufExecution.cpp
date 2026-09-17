//
//  GatedRMSNormBufExecution.cpp
//  MNN
//
//  OpenCL (buffer mode) execution for OpType_GatedRMSNorm:
//  out = (RMSNorm(x) * gamma + beta) * silu(z).
//
//  Why keep the op whole: the geometry decomposition emits LayerNorm + a raster
//  (the flat view of the normalized result) + SILU + MUL, i.e. four dispatches
//  and three full round-trips through DRAM for what is one pass over the data.
//  In the linear-attention blocks this op sits per layer, so those launches are
//  paid per token. One kernel does the whole thing from x and z.
//
//  The decomposition stays the fallback for anything gatedRMSNormOpenCLOk
//  rejects; that predicate is shared with the geometry gate, which must not keep
//  an op whole that this creator would then refuse.
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && defined(MNN_GATED_RMS_NORM)

#include "backend/opencl/execution/buffer/GatedRMSNormBufExecution.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace OpenCL {

// gamma / beta upload, mirroring LayerNormBufExecution: padded to 4 so the
// vector kernel can read them as FLOAT4, and honouring the mmap weight cache.
static std::shared_ptr<cl::Buffer> _uploadParam(OpenCLBackend* backend, const float* data, int size) {
    const bool isHalf = backend->getPrecision() != BackendConfig::Precision_High;
    const size_t unitSize = isHalf ? sizeof(half_float::half) : sizeof(float);
    const size_t bytes = ALIGN_UP4(size) * unitSize;
    auto staticMapAlloc = backend->getStaticAllocatorMMap();
    std::shared_ptr<cl::Buffer> buffer;
    if (backend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr) {
        buffer = staticMapAlloc.get()->allocBuffer(bytes);
    } else {
        buffer.reset(
            new cl::Buffer(backend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes));
    }
    if (nullptr == buffer || nullptr == buffer->get()) {
        return nullptr;
    }
    if (backend->getRuntime()->hint().useCachedMmap > 1) {
        // Already filled from the cache file.
        return buffer;
    }
    auto error = CL_SUCCESS;
    auto ptr = backend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*buffer, true, CL_MAP_WRITE, 0, bytes,
                                                                            nullptr, nullptr, &error);
    if (nullptr == ptr || error != CL_SUCCESS) {
        MNN_ERROR("GatedRMSNormBufExecution: map param buffer failed\n");
        if (nullptr != ptr) {
            backend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*buffer, ptr);
        }
        return nullptr;
    }
    ::memset(ptr, 0, bytes);
    if (isHalf) {
        for (int i = 0; i < size; ++i) {
            ((half_float::half*)ptr)[i] = (half_float::half)(data[i]);
        }
    } else {
        ::memcpy(ptr, data, size * sizeof(float));
    }
    backend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*buffer, ptr);
    return buffer;
}

GatedRMSNormBufExecution::GatedRMSNormBufExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                                   Backend* backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto param = op->main_as_LayerNorm();
    mResource.reset(new GatedRMSNormResource);
    mResource->epsilon = param->epsilon();
    mResource->hasGammaBeta = (nullptr != param->gamma() && nullptr != param->beta());
    if (mResource->hasGammaBeta) {
        const int size = (int)param->gamma()->size();
        mResource->mGammaBuffer = _uploadParam(mOpenCLBackend, param->gamma()->data(), size);
        mResource->mBetaBuffer = _uploadParam(mOpenCLBackend, param->beta()->data(), size);
        if (nullptr == mResource->mGammaBuffer || nullptr == mResource->mBetaBuffer) {
            mValid = false;
            return;
        }
    }
}

GatedRMSNormBufExecution::GatedRMSNormBufExecution(std::shared_ptr<GatedRMSNormResource> resource, const MNN::Op* op,
                                                   Backend* backend)
    : CommonExecution(backend, op) {
    mResource = resource;
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
}

bool GatedRMSNormBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new GatedRMSNormBufExecution(mResource, op, bn);
    return true;
}

ErrorCode GatedRMSNormBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    int outside = 0, inside = 0, heads = 0;
    if (!OpCommonUtils::gatedRMSNormOpenCLOk(mOp, inputs, outputs, &outside, &inside, &heads)) {
        // The geometry gate asked the same question at build time, so this can
        // only trip if the shapes changed under us.
        MNN_ERROR("GatedRMSNormBufExecution: unsupported shape\n");
        return NOT_SUPPORT;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    // One workgroup per row of x, cooperating over the row's channels.
    const bool align4 = (inside % 4) == 0;
    const int reduceSize = align4 ? (inside / 4) : inside;
    const uint32_t maxLocalSize =
        std::min(std::min(runtime->getMaxWorkItemSizes()[0], runtime->MaxWorkGroupSize()), (uint32_t)256);
    int localSize = 1;
    while (localSize * 2 <= (int)maxLocalSize && localSize * 2 <= reduceSize) {
        localSize *= 2;
    }
    const std::string kernelName = align4 ? "gated_rms_norm_c4_buf" : "gated_rms_norm_buf";

    mUnits.resize(1);
    auto& unit = mUnits[0];
    while (true) {
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(localSize));
        if (mResource->hasGammaBeta) {
            buildOptions.emplace("-DGAMMA_BETA");
        }
        unit.kernel =
            runtime->buildKernel("gated_rms_norm_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
        OPENCL_CHECK_KERNEL(unit.kernel);
        const uint32_t kernelMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        if (kernelMaxWorkGroupSize == 0) {
            return NOT_SUPPORT;
        }
        if ((uint32_t)localSize <= kernelMaxWorkGroupSize) {
            break;
        }
        do {
            localSize /= 2;
        } while (localSize > 1 && (uint32_t)localSize > kernelMaxWorkGroupSize);
        if ((uint32_t)localSize > kernelMaxWorkGroupSize) {
            return NOT_SUPPORT;
        }
    }
    const std::vector<uint32_t> gws{(uint32_t)localSize, (uint32_t)outside};
    const std::vector<uint32_t> lws{(uint32_t)localSize, 1};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, gws[0]);
    ret |= unit.kernel->get().setArg(idx++, gws[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[0]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[1]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[0]));
    if (mResource->hasGammaBeta) {
        ret |= unit.kernel->get().setArg(idx++, *mResource->mGammaBuffer.get());
        ret |= unit.kernel->get().setArg(idx++, *mResource->mBetaBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inside));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(heads));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(outside / heads));
    ret |= unit.kernel->get().setArg(idx++, mResource->epsilon);
    MNN_CHECK_CL_SUCCESS(ret, "setArg GatedRMSNormBufExecution");

    mOpenCLBackend->recordKernel2d(unit.kernel, gws, lws);
    unit.globalWorkSize = {gws[0], gws[1]};
    unit.localWorkSize = {lws[0], lws[1]};
    return NO_ERROR;
}

class GatedRMSNormBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~GatedRMSNormBufCreator() = default;
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        // Must match the geometry keep-whole gate exactly: an op the geometry
        // keeps whole but this refuses would fail session creation.
        if (!OpCommonUtils::gatedRMSNormOpenCLOk(op, inputs, outputs)) {
            return nullptr;
        }
        // The kernels address x, z and out as plain C4 buffers, like the
        // LayerNorm / Binary chain they replace.
        for (auto t : inputs) {
            TensorUtils::setTensorSupportPack(t, false);
        }
        for (auto t : outputs) {
            TensorUtils::setTensorSupportPack(t, false);
        }
        OPENCL_CREATOR_CHECK(new GatedRMSNormBufExecution(inputs, op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(GatedRMSNormBufCreator, OpType_GatedRMSNorm, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE && MNN_GATED_RMS_NORM */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
