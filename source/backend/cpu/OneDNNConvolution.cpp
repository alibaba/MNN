#ifdef MNN_USE_ONEDNN
#include "OneDNNConvolution.hpp"
#include "CPUConvolution.hpp"
#include "dnnl.hpp"
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace MNN {
namespace OneDNN {
class OneDNNConvolution : public Execution {
public:
    OneDNNConvolution(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                      size_t originWeightSize, const float *bias, size_t biasSize) : Execution(b) {
        mCommon = common;
        const auto convCommon             = common;
        const auto kw                     = convCommon->kernelX();
        const auto kh                     = convCommon->kernelY();
        auto ic                           = convCommon->inputCount();
        const auto oc                     = convCommon->outputCount();
        const auto strideX                = convCommon->strideX();
        const auto strideY                = convCommon->strideY();
        if (0 == ic) {
            ic = originWeightSize / oc / kw / kh;
        }
        eng = engine(engine::kind::cpu, 0);
        stm = stream(eng);
        memory::dims conv_weights_tz = {oc, ic, kh, kw};
        memory::dims conv_bias_tz = {oc};
        memory::dims conv_strides = {strideX, strideY};
        int defaultOw = 10;
        int defaultOh = 10;
        memory::dims conv_src_tz = {1, ic, mCommon->strideY() * (defaultOh - 1) + (kh - 1) * mCommon->dilateY() + 1, (kw - 1) * mCommon->dilateX() + 1 + mCommon->strideX() * (defaultOw - 1)};
        memory::dims conv_dst_tz = {1, oc, defaultOh, defaultOw};
        memory::dims conv_padding = {0, 0};
        if (mCommon->relu()) {
            post_ops ops;
            ops.append_eltwise(1.0f, algorithm::eltwise_relu, 0.0f, 0.0f);
            conv_attr.set_post_ops(ops);
        }
        if (mCommon->relu6()) {
            post_ops ops;
            ops.append_eltwise(1.0f, algorithm::eltwise_clip, 0.0f, 6.0f);
            conv_attr.set_post_ops(ops);
        }
        auto user_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::oihw);

        auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
        auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
        auto conv_bias_md = memory::desc({conv_bias_tz}, dt::f32, tag::a);
        auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::any);

        auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_auto, conv_src_md, conv_weights_md, conv_bias_md,
            conv_dst_md, conv_strides, conv_padding, conv_padding);
        auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, eng);
        const auto* weightSrc = originWeight;
        mWeight.reset(Tensor::createDevice<int8_t>({(int)conv_pd.weights_desc().get_size()}));
        auto res = b->onAcquireBuffer(mWeight.get(), Backend::STATIC);
        if (!res) {
            mValid = false;
            return;
        }
        auto user_weights = memory(user_weights_md, eng, (float*)weightSrc);
        conv_weights = memory(conv_pd.weights_desc(), eng, mWeight->host<float>());
        auto r_pd = reorder::primitive_desc(user_weights, conv_weights);
        reorder(r_pd).execute(stm, user_weights, conv_weights);

        conv_bias = memory(conv_bias_md, eng);
        {
            auto ptr = conv_bias.map_data();
            ::memcpy(ptr, bias, biasSize * sizeof(float));
            conv_bias.unmap_data(ptr);
        }
    }
    virtual ~OneDNNConvolution() {
        if (nullptr != mWeight) {
            backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
        }
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        const auto convCommon             = mCommon;
        const auto kw                     = convCommon->kernelX();
        const auto kh                     = convCommon->kernelY();
        const auto ic                     = inputs[0]->channel();
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

        auto user_src_md = memory::desc({conv_src_tz}, dt::f32, tag::nChw4c);
        auto user_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::oihw);
        auto user_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::nChw4c);

        auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
        auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::any);

        user_src = memory(user_src_md, eng, inputs[0]->host<float>());
        user_dst = memory(user_dst_md, eng, outputs[0]->host<float>());
        mSrcTemp = nullptr;
        mDstTemp = nullptr;

        // Fix weight desc and bias desc
        auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_auto, conv_src_md, conv_weights.get_desc(), conv_bias.get_desc(),
                                                   conv_dst_md, conv_strides, {std::get<1>(pads), std::get<0>(pads)}, {std::get<3>(pads), std::get<2>(pads)});
        auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, eng);
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
            conv_src = memory(conv_pd.src_desc(), eng, mSrcTemp->host<float>());
        }
        if (conv_pd.dst_desc() != user_dst.get_desc()) {
            auto needSize = conv_pd.dst_desc().get_size();
            mDstTemp.reset(Tensor::createDevice<int8_t>({(int)needSize}));
            auto res = backend()->onAcquireBuffer(mDstTemp.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
            conv_dst = memory(conv_pd.dst_desc(), eng, mDstTemp->host<float>());
        }
        if (nullptr != mSrcTemp) {
            backend()->onReleaseBuffer(mSrcTemp.get(), Backend::DYNAMIC);
        }
        if (nullptr != mDstTemp) {
            backend()->onReleaseBuffer(mDstTemp.get(), Backend::DYNAMIC);
        }        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
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
                           {DNNL_ARG_WEIGHTS, conv_weights},
                           {DNNL_ARG_BIAS, conv_bias},
                           {DNNL_ARG_DST, conv_dst_temp}});
        if (nullptr != mDstTemp) {
            auto r_pd = reorder::primitive_desc(conv_dst, user_dst);
            reorder(r_pd).execute(stm, conv_dst, user_dst);
        }
        return NO_ERROR;
    }
private:
    engine eng;
    stream stm;
    convolution_forward conv;
    memory conv_weights;
    memory conv_bias;
    primitive_attr conv_attr;
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mSrcTemp;
    std::shared_ptr<Tensor> mDstTemp;
    memory user_src;
    memory user_dst;
    memory conv_src;
    memory conv_dst;
    const Convolution2DCommon* mCommon;
};


Execution* createConvolution(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                             size_t originWeightSize, const float *bias, size_t biasSize) {
    return new OneDNNConvolution(common, b, originWeight, originWeightSize, bias, biasSize);
}


}
};
#endif
