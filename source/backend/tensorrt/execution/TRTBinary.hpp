//
//  TRTBinary.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTBinary_HPP
#define MNN_TRTBinary_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTBinary : public TRTCommonExecution {
public:
    TRTBinary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTBinary() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;

private:
    IConstantLayer *const_layer;
};

class TRTNormalPlugin : public TRTCommonExecution {
public:
    TRTNormalPlugin(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                    const std::vector<Tensor *> &outputs);
    virtual ~TRTNormalPlugin() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
};

class TRTRaster : public TRTCommonExecution {
public:
    TRTRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTRaster() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
};

class TRTScatterNd : public TRTCommonExecution {
public:
    TRTScatterNd(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTScatterNd() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;

private:
    IConstantLayer *const_layer;
};

class TRTInterp : public TRTCommonExecution {
public:
    TRTInterp(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTInterp() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
};

class TRTGather : public TRTCommonExecution {
public:
    TRTGather(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTGather() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
};

} // namespace MNN

#endif // MNN_TRTBinary_HPP
