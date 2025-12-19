#ifndef CONVTMAC_HPP
#define CONVTMAC_HPP
#include "backend/cpu/CPUConvolution.hpp"
#include "Int8FunctionsOpt.h"
#include "CommonOptFunction.h"

namespace MNN {
//struct TMacResource;
struct TMacCache;
class ConvTMac : public CPUConvolution {
public:
    virtual ~ ConvTMac();
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    static ConvTMac* create(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon);
private:
    ConvTMac(const Convolution2DCommon *convOp, Backend *b);
    std::shared_ptr<TMacResource> mResource;
    std::shared_ptr<TMacCache> mCache;
    std::vector<float> mParameters;
    // functions
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, ssize_t)> mQuantFunc;
};
};

#endif
