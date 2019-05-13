//
//  MetalLSTM.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalLSTM.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalLSTM::MetalLSTM(Backend *backend, const LSTM *lstm) : Execution(backend), mLSTM(lstm) {
    // nothing to do
}

ErrorCode MetalLSTM::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // divide weight & bias if they are united in weightI.
    auto input   = inputs[0];
    auto output  = outputs[0];
    auto weightI = mLSTM->weightI();
    auto weightH = mLSTM->weightH();
    auto bias    = mLSTM->bias();
    auto iw      = input->width();
    auto ow      = output->width();
    auto size    = weightI->dims()->data()[0];
    auto united  = weightI && !weightH && size == 4 * ow * (iw + ow + 2);
    if (united) {
        auto data = weightI->float32s()->data();
        mWeightI  = [context newDeviceBuffer:iw * ow * 4 * sizeof(metal_float) access:CPUWriteOnly];
        {
            auto to   = (metal_float *)mWeightI.contents;
            auto step = iw * ow;
#pragma clang loop vectorize(enable) unroll(enable)
            for (int i = 0; i < step; i++, to += 4) {
                to[0] = data[i];                      // I
                to[1] = data[i + step];               // F
                to[2] = data[i + step + step + step]; // O
                to[3] = data[i + step + step];        // G
            }
            data += 4 * step;
        }
        mWeightH = [context newDeviceBuffer:ow * ow * 4 * sizeof(metal_float) access:CPUWriteOnly];
        {
            // convert from III...FFF...GGG...OOO to IFOG...IFOG
            auto to   = (metal_float *)mWeightH.contents;
            auto step = ow * ow;
#pragma clang loop vectorize(enable) unroll(enable)
            for (int i = 0; i < step; i++, to += 4) {
                to[0] = data[i];                      // I
                to[1] = data[i + step];               // F
                to[2] = data[i + step + step + step]; // O
                to[3] = data[i + step + step];        // G
            }
            data += 4 * step;
        }
        mBias = [context newDeviceBuffer:ow * 4 * sizeof(metal_float) access:CPUWriteOnly];
        { // bias
            // convert from III...FFF...GGG...OOO to IFOG...IFOG
            auto to   = (metal_float *)mBias.contents;
            auto step = ow;
#pragma clang loop vectorize(enable) unroll(enable)
            for (int i = 0; i < step; i++, to += 4) {
                to[0] = data[i];                      // I
                to[1] = data[i + step];               // F
                to[2] = data[i + step + step + step]; // O
                to[3] = data[i + step + step];        // G
            }
        }
    } else {
        mWeightI = [context newDeviceBuffer:iw * ow * 4 * sizeof(metal_float) access:CPUWriteOnly];
        {
            // convert from III...FFF...OOO...GGG to IFOG...IFOG
            auto from = weightI->float32s()->data();
            auto to   = (metal_float *)mWeightI.contents;
            auto sect = iw * ow;
#pragma clang loop vectorize(enable) unroll(enable)
            for (int i = 0; i < sect; i++, to += 4, from++) {
                to[0] = from[0];
                to[1] = from[0 + sect];
                to[2] = from[0 + sect + sect];
                to[3] = from[0 + sect + sect + sect];
            }
        }
        mWeightH = [context newDeviceBuffer:ow * ow * 4 * sizeof(metal_float) access:CPUWriteOnly];
        {
            // convert from III...FFF...OOO...GGG to IFOG...IFOG
            auto from = weightH->float32s()->data();
            auto to   = (metal_float *)mWeightH.contents;
            auto sect = ow * ow;
#pragma clang loop vectorize(enable) unroll(enable)
            for (int i = 0; i < sect; i++, to += 4, from++) {
                to[0] = from[0];
                to[1] = from[0 + sect];
                to[2] = from[0 + sect + sect];
                to[3] = from[0 + sect + sect + sect];
            }
        }
        mBias = [context newDeviceBuffer:ow * 4 * sizeof(metal_float) access:CPUWriteOnly];
        {
            // convert from III...FFF...OOO...GGG to IFOG...IFOG
            auto from = bias->float32s()->data();
            auto to   = (metal_float *)mBias.contents;
            auto sect = ow;
#pragma clang loop vectorize(enable) unroll(enable)
            for (int i = 0; i < sect; i++, to += 4, from++) {
                to[0] = from[0];
                to[1] = from[0 + sect];
                to[2] = from[0 + sect + sect];
                to[3] = from[0 + sect + sect + sect];
            }
        }
    }

    return NO_ERROR;
}

ErrorCode MetalLSTM::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    int iw = input->width(), ow = output->width(), c = input->channel(), z = UP_DIV(c, 4);
    
    auto constBuffer                 = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    ((int *)constBuffer.contents)[0] = ow;
    ((int *)constBuffer.contents)[1] = iw;
    ((int *)constBuffer.contents)[2] = c;
    ((int *)constBuffer.contents)[3] = z;

    // calc gates
    auto encoder    = [context encoder];
    auto gateBuffer = [context newHeapBuffer:4 * ow * z * 4 * sizeof(metal_float) access:CPUTransparent];
    auto bandwidth  = [context load:@"lstm_gate" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)inputs[0]->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:mWeightI offset:0 atIndex:1];
    [encoder setBuffer:gateBuffer offset:0 atIndex:2];
    [encoder setBuffer:constBuffer offset:0 atIndex:3];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) ow, (NSUInteger)z, 1 } bandwidth:bandwidth];

    // calc output
    auto cont = inputs.size() > 1 ? (__bridge id<MTLBuffer>)(void *)inputs[1]->deviceId() : NULL;
    bandwidth = [context load:inputs.size() > 1 ? @"lstm_cont" : @"lstm" encoder:encoder];
    [encoder setBuffer:gateBuffer offset:0 atIndex:0];
    [encoder setBuffer:mWeightH offset:0 atIndex:1];
    [encoder setBuffer:mBias offset:0 atIndex:2];
    [encoder setBuffer:constBuffer offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:4];
    [encoder setBuffer:cont offset:0 atIndex:5];
    [encoder setThreadgroupMemoryLength:ow * sizeof(metal_float) atIndex:0];
    [context dispatchEncoder:encoder
        threads:{ (NSUInteger) ow, 1, 1 }
        threadsPerGroup:{ (NSUInteger) ow, 1, 1 }
        bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);

    // clean up
    [context releaseHeapBuffer:gateBuffer];
    return NO_ERROR;
}

class MetalLSTMCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalLSTM(backend, op->main_as_LSTM());
    }
};
REGISTER_METAL_OP_CREATOR(MetalLSTMCreator, OpType_LSTM);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
