//
//  OneDNNConvInt8.hpp
//
//

#ifndef OneDNNConvInt8_hpp
#define OneDNNConvInt8_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "dnnl.hpp"

using namespace dnnl;
namespace MNN {

class OneDNNConvInt8 : public CPUConvolution {
public:
    struct Resource : public CPUConvolution::Resource {
        memory conv_weights;
        memory conv_bias;
        primitive_attr conv_attr;
        engine eng;
    };
    static Execution* create(Backend *backend, const MNN::Convolution2D *convOp, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    OneDNNConvInt8(std::shared_ptr<OneDNNConvInt8::Resource> resource, const MNN::Convolution2DCommon* common, Backend* bn);
    virtual ~OneDNNConvInt8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<OneDNNConvInt8::Resource> mResource;
    stream stm;
    convolution_forward conv;
    std::shared_ptr<Tensor> mSrcTemp;
    std::shared_ptr<Tensor> mDstTemp;
    memory user_src;
    memory user_dst;
    memory conv_src;
    memory conv_dst;
};
} // namespace MNN
#endif /* OneDNNConvInt8_hpp */
