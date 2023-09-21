//
//  OneDNNConvInt8.cpp
//
//
#ifdef MNN_USE_ONEDNN
#include "backend/cpu/OneDNNConvInt8.hpp"
#include "core/ConvolutionCommon.hpp"
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace MNN {
OneDNNConvInt8::~OneDNNConvInt8() {
    // Do nothing
}

Execution* OneDNNConvInt8::create(Backend* backend, const MNN::Convolution2D* convParam, const std::vector<Tensor*>& inputs, const std::vector<Tensor *> &outputs) {
    std::shared_ptr<OneDNNConvInt8::Resource> resource(new OneDNNConvInt8::Resource);
    resource->backend = backend;
    const auto convCommon             = convParam->common();
    const auto kw                     = convCommon->kernelX();
    const auto kh                     = convCommon->kernelY();
    const auto ic                     = convCommon->inputCount();
    const auto oc                     = convCommon->outputCount();
    const auto strideX                = convCommon->strideX();
    const auto strideY                = convCommon->strideY();
    auto weights                      = convParam->symmetricQuan()->weight()->data();
    auto bias                         = convParam->symmetricQuan()->bias()->data();
    std::vector<float> scale(oc);
    for (auto i = 0; i < scale.size(); i++) {
        scale[i] = convParam->symmetricQuan()->scale()->data()[i];
    }
    const int conv_mask = 2;
    resource->conv_attr.set_output_scales(conv_mask, scale);
    if (convCommon->relu() || convCommon->relu6()) {
        post_ops ops;
        ops.append_eltwise(1.0f, algorithm::eltwise_relu, 0.0f, 0.0f);
        resource->conv_attr.set_post_ops(ops);
    }
    auto eng = engine(engine::kind::cpu, 0);
    resource->eng = eng;
    auto stm = stream(eng);
    memory::dims conv_weights_tz = {oc, ic, kh, kw};
    memory::dims conv_bias_tz = {oc};
    memory::dims conv_strides = {strideX, strideY};
    memory::dims conv_src_tz = {1, ic, convCommon->strideY() + (kh - 1) * convCommon->dilateY() + 1, (kw - 1) * convCommon->dilateX() + 1 + convCommon->strideX()};
    memory::dims conv_dst_tz = {1, oc, 2, 2};
    memory::dims conv_padding = {0, 0};

    auto user_weights_md = memory::desc({conv_weights_tz}, dt::s8, tag::oihw);

    auto conv_src_md = memory::desc({conv_src_tz}, dt::s8, tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::s8, tag::any);
    auto conv_bias_md = memory::desc({conv_bias_tz}, dt::s32, tag::a);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::s8, tag::any);

    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_auto, conv_src_md, conv_weights_md, conv_bias_md,
        conv_dst_md, conv_strides, conv_padding, conv_padding);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, resource->conv_attr, eng);
    auto weightSrc = convParam->symmetricQuan()->weight()->data();
    resource->mWeight.reset(Tensor::createDevice<int8_t>({(int)conv_pd.weights_desc().get_size()}));
    resource->mBias.reset(Tensor::createDevice<int32_t>({(int)convParam->symmetricQuan()->bias()->size()}));
    auto res = backend->onAcquireBuffer(resource->mWeight.get(), Backend::STATIC);
    res = res && backend->onAcquireBuffer(resource->mBias.get(), Backend::STATIC);
    if (!res) {
        return nullptr;
    }
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (convParam->quanParameter() != nullptr) {
        quanCommon = ConvolutionCommon::load(convParam, backend(), false);
        weightSrc = quanCommon->weight.get();
    }
    auto user_weights = memory(user_weights_md, eng, (int8_t*)weightSrc);
    auto conv_weights = memory(conv_pd.weights_desc(), eng, resource->mWeight->host<int8_t>());
    auto r_pd = reorder::primitive_desc(user_weights, conv_weights);
    reorder(r_pd).execute(stm, user_weights, conv_weights);
    ::memcpy(resource->mBias->host<int32_t>(), convParam->symmetricQuan()->bias()->data(), convParam->symmetricQuan()->bias()->size() * sizeof(int32_t));
    resource->conv_bias = memory(conv_bias_md, eng, resource->mBias->host<int32_t>());
    resource->conv_weights = conv_weights;
    return new OneDNNConvInt8(resource, convCommon, backend);
}

OneDNNConvInt8::OneDNNConvInt8(std::shared_ptr<OneDNNConvInt8::Resource> resource, const MNN::Convolution2DCommon* common, Backend* bn) : CPUConvolution(common, bn) {
    mResource = resource;
    stm = stream(mResource->eng);
}

bool OneDNNConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new OneDNNConvInt8(mResource, op->main_as_Convolution2D()->common(), bn);
    *dst = dstExe;
    return true;
}

ErrorCode OneDNNConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto convCommon             = mCommon;
    const auto kw                     = convCommon->kernelX();
    const auto kh                     = convCommon->kernelY();
    const auto ic                     = convCommon->inputCount();
    const auto oc                     = convCommon->outputCount();
    const auto strideX                = convCommon->strideX();
    const auto strideY                = convCommon->strideY();
    const auto ih                     = inputs[0]->height();
    const auto iw                     = inputs[0]->width();
    const auto oh                     = outputs[0]->height();
    const auto ow                     = outputs[0]->width();
    auto pads = ConvolutionCommon::convolutionPadFull(inputs[0], outputs[0], mCommon);

    memory::dims conv_src_tz = {inputs[0]->batch(), ic, ih, iw};
    memory::dims conv_weights_tz = {oc, ic, kh, kw};
    memory::dims conv_bias_tz = {oc};
    memory::dims conv_dst_tz = {outputs[0]->batch(), oc, oh, ow};
    memory::dims conv_strides = {strideX, strideY};

    auto user_src_md = memory::desc({conv_src_tz}, dt::s8, tag::nChw4c);
    auto user_weights_md = memory::desc({conv_weights_tz}, dt::s8, tag::oihw);
    auto user_dst_md = memory::desc({conv_dst_tz}, dt::s8, tag::nChw4c);

    auto conv_src_md = memory::desc({conv_src_tz}, dt::s8, tag::any);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::s8, tag::any);

    user_src = memory(user_src_md, mResource->eng, inputs[0]->host<int8_t>());
    user_dst = memory(user_dst_md, mResource->eng, outputs[0]->host<int8_t>());
    mSrcTemp = nullptr;
    mDstTemp = nullptr;

    // Fix weight desc and bias desc
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_auto, conv_src_md, mResource->conv_weights.get_desc(), mResource->conv_bias.get_desc(),
                                               conv_dst_md, conv_strides, {std::get<1>(pads), std::get<0>(pads)}, {std::get<3>(pads), std::get<2>(pads)});
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, mResource->conv_attr, mResource->eng);
    conv = convolution_forward(conv_pd);
    mSrcTemp = nullptr;
    mDstTemp = nullptr;
    if (conv_pd.src_desc() != user_src.get_desc()) {
        auto needSize = conv_pd.src_desc().get_size();
        mSrcTemp.reset(Tensor::createDevice<int8_t>({(int)needSize}));
        auto res = backend()->onAcquireBuffer(mSrcTemp.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        conv_src = memory(conv_pd.src_desc(), mResource->eng, mSrcTemp->host<int8_t>());
    }
    if (conv_pd.dst_desc() != user_dst.get_desc()) {
        auto needSize = conv_pd.dst_desc().get_size();
        mDstTemp.reset(Tensor::createDevice<int8_t>({(int)needSize}));
        auto res = backend()->onAcquireBuffer(mDstTemp.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        conv_dst = memory(conv_pd.dst_desc(), mResource->eng, mDstTemp->host<int8_t>());
    }
    if (nullptr != mSrcTemp) {
        backend()->onReleaseBuffer(mSrcTemp.get(), Backend::DYNAMIC);
    }
    if (nullptr != mDstTemp) {
        backend()->onReleaseBuffer(mDstTemp.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode OneDNNConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    memory conv_src_temp = user_src;
    if (nullptr != mSrcTemp) {
        auto r_pd = reorder::primitive_desc(user_src, conv_src);
        reorder(r_pd).execute(stm, user_src, conv_src);
        conv_src_temp = conv_src;
    }
    memory conv_dst_temp = user_dst;
    if (nullptr != mDstTemp) {
        conv_dst_temp = conv_dst;
    }
    conv.execute(stm, {{DNNL_ARG_SRC, conv_src_temp},
                       {DNNL_ARG_WEIGHTS, mResource->conv_weights},
                       {DNNL_ARG_BIAS, mResource->conv_bias},
                       {DNNL_ARG_DST, conv_dst_temp}});
    if (nullptr != mDstTemp) {
        auto r_pd = reorder::primitive_desc(conv_dst, user_dst);
        reorder(r_pd).execute(stm, conv_dst, user_dst);
    }
    return NO_ERROR;
}
} // namespace MNN
#endif
