//
//  GeometryConvUtils.hpp
//  MNN
//
//  Created by MNN on 2020/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GeometryConvUtils_hpp
#define GeometryConvUtils_hpp
#include "core/ConvolutionCommon.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
class GeometryConvUtils {
public:
    static void im2Col3d(Tensor* im2Col, Tensor* input, int ic, int kd, int kh, int kw, int batch, int od, int oh, int ow, int id, int ih, int iw,
                         int sd, int sh, int sw, int dd, int dh, int dw, int pd, int ph, int pw, int srcKernelOffset = 0);
    static std::shared_ptr<Tensor> im2Col(Tensor* im2Col, Tensor* input, int ic, int kh, int kw, int batch, int oh, int ow, int ih, int iw,
                       int sh, int sw, int dh, int dw, std::pair<int, int> pads, int srcKernelOffset = 0, Tensor* padVal = nullptr);
    static bool computeSingle(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       GeometryComputer::Context& context, CommandBuffer& res);
    static flatbuffers::Offset<Op> makeRelu6(flatbuffers::FlatBufferBuilder& builder, float minValue, float maxValue);
};
} // namespace MNN
#endif
