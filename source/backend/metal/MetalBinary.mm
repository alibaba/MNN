//
//  MetalBinary.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalBinary.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalBinary::MetalBinary(Backend *backend, int binarytype) : Execution(backend) {
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    mBinaryType  = [context newDeviceBuffer:sizeof(int) bytes:&binarytype access:CPUWriteOnly];
}

ErrorCode MetalBinary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
    const int input0_data_count = (int)input0->elementSize();
    const int input1_data_count = (int)input1->elementSize();

    // scalar support
    int iw0 = input0->width(), ih0 = input0->height();
    int iw1 = input1->width(), ih1 = input1->height();
    int ow = output->width(), oh = output->height(), oc = output->channel(), ob = output->batch();
    iw0 = iw0 == 0 ? 1 : iw0;
    ih0 = ih0 == 0 ? 1 : ih0;
    iw1 = iw1 == 0 ? 1 : iw1;
    ih1 = ih1 == 0 ? 1 : ih1;
    ow  = ow == 0 ? 1 : ow;
    oh  = oh == 0 ? 1 : oh;
    oc  = oc == 0 ? 1 : oc;
    ob  = ob == 0 ? 1 : ob;

    bool same_shape = true;
    // scalar input
    if (inputs[0]->buffer().dimensions == 0 || inputs[1]->buffer().dimensions == 0) {
        // do nothing
    }
    // same shape
    else if (inputs[0]->buffer().dimensions == inputs[1]->buffer().dimensions) {
        for (int i = 0; i < inputs[0]->buffer().dimensions; i++) {
            if (inputs[0]->buffer().dim[i].extent != inputs[1]->buffer().dim[i].extent) {
                same_shape = false;
                break;
            }
        }
    }
    // different shape
    else {
        same_shape = false;
    }

    // encode
    auto output_dimensions = output->buffer().dimensions;
    auto shape             = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
    auto encoder           = [context encoder];
    if (same_shape == false) {
        // dim
        auto dimsIn0Buffer = [context newDeviceBuffer:sizeof(int) * output_dimensions access:CPUWriteOnly];
        auto dimsIn1Buffer = [context newDeviceBuffer:sizeof(int) * output_dimensions access:CPUWriteOnly];
        int *dims0         = (int *)dimsIn0Buffer.contents;
        int *dims1         = (int *)dimsIn1Buffer.contents;
        for (int i = 0; i < output_dimensions; i++) {
            dims0[i] = dims1[i] = 1;
        }
        for (int i = input0->buffer().dimensions - 1, j = output_dimensions - 1; i >= 0; i--, j--) {
            dims0[j] = input0->buffer().dim[i].extent;
        }
        for (int i = input1->buffer().dimensions - 1, j = output_dimensions - 1; i >= 0; i--, j--) {
            dims1[j] = input1->buffer().dim[i].extent;
        }

        // strides & shape
        auto stridesIn0Buffer = [context newDeviceBuffer:sizeof(int) * output_dimensions access:CPUWriteOnly];
        auto stridesIn1Buffer = [context newDeviceBuffer:sizeof(int) * output_dimensions access:CPUWriteOnly];
        auto stridesOutBuffer = [context newDeviceBuffer:sizeof(int) * output_dimensions access:CPUWriteOnly];
        int *input0_strides   = (int *)stridesIn0Buffer.contents;
        int *input1_strides   = (int *)stridesIn1Buffer.contents;
        int *output_strides   = (int *)stridesOutBuffer.contents;
        int input_data_count0 = 1, input_data_count1 = 1;
        int output_data_count = 1;
        for (int i = output_dimensions - 1; i >= 0; i--) {
            input0_strides[i] = input_data_count0;
            input_data_count0 *= dims0[i];
            input1_strides[i] = input_data_count1;
            input_data_count1 *= dims1[i];
            output_strides[i] = output_data_count;
            output_data_count *= output->buffer().dim[i].extent;
        }
        ((int *)shape.contents)[0] = input0_data_count;
        ((int *)shape.contents)[1] = input1_data_count;
        ((int *)shape.contents)[2] = output_data_count;
        ((int *)shape.contents)[3] = ow;
        ((int *)shape.contents)[4] = ow * oh;
        ((int *)shape.contents)[5] = output_dimensions;

        // encode
        auto bandwidth = [context load:@"binary_notshape" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
        [encoder setBuffer:shape offset:0 atIndex:3];
        [encoder setBuffer:mBinaryType offset:0 atIndex:4];
        [encoder setBuffer:dimsIn0Buffer offset:0 atIndex:5];
        [encoder setBuffer:dimsIn1Buffer offset:0 atIndex:6];
        [encoder setBuffer:stridesIn0Buffer offset:0 atIndex:7];
        [encoder setBuffer:stridesIn1Buffer offset:0 atIndex:8];
        [encoder setBuffer:stridesOutBuffer offset:0 atIndex:9];
        [context dispatchEncoder:encoder threads:{ (NSUInteger) output_data_count, 1, 1 } bandwidth:bandwidth];
    } else {
        int outdatacount = 0;
        if (input0_data_count == input1_data_count) {
            outdatacount = input0_data_count;
        } else {
            outdatacount = input0_data_count > input1_data_count ? input0_data_count : input1_data_count;
        }
        ((int *)shape.contents)[0] = input0_data_count;
        ((int *)shape.contents)[1] = input1_data_count;
        ((int *)shape.contents)[2] = outdatacount;
        ((int *)shape.contents)[3] = ow;
        ((int *)shape.contents)[4] = ow * oh;
        ((int *)shape.contents)[5] = output_dimensions;

        auto bandwidth = [context load:@"binary_normal" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
        [encoder setBuffer:shape offset:0 atIndex:3];
        [encoder setBuffer:mBinaryType offset:0 atIndex:4];
        [context dispatchEncoder:encoder threads:{ (NSUInteger) outdatacount, 1, 1 } bandwidth:bandwidth];
    }

    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalBinaryCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto binaryop = op->main_as_BinaryOp();
        return new MetalBinary(backend, binaryop->opType());
    }
};
REGISTER_METAL_OP_CREATOR(MetalBinaryCreator, OpType_BinaryOp);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
