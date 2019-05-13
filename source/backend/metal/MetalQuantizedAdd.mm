//
//  MetalQuantizedAdd.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalQuantizedAdd.hpp"
#import "CPUQuantizationUtils.hpp"
#import "MNNMetalContext.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalQuantizedAdd::MetalQuantizedAdd(Backend *backend, const MNN::QuantizedAdd *add) : Execution(backend) {
    // offset
    int input1_offset = -add->input1QuantizedParam()->zeroPoint();
    int input2_offset = -add->input2QuantizedParam()->zeroPoint();
    int output_offset = add->outputQuantizedParam()->zeroPoint();

    // multiplier & shift
    const int left_shift = 20;
    const double twice_max_inputScale =
        2 * std::max(add->input1QuantizedParam()->scale(), add->input2QuantizedParam()->scale());
    const double real_input1_multiplier = add->input1QuantizedParam()->scale() / twice_max_inputScale;
    const double real_input2_multiplier = add->input2QuantizedParam()->scale() / twice_max_inputScale;
    const double real_output_multiplier =
        twice_max_inputScale / ((1 << left_shift) * add->outputQuantizedParam()->scale());
    int input1_multiplier = 0, input1_shift = 0;
    int input2_multiplier = 0, input2_shift = 0;
    int output_multiplier = 0, output_shift = 0;
    QuantizeMultiplierSmallerThanOne(real_input1_multiplier, &input1_multiplier, &input1_shift);
    QuantizeMultiplierSmallerThanOne(real_input2_multiplier, &input2_multiplier, &input2_shift);
    QuantizeMultiplierSmallerThanOne(real_output_multiplier, &output_multiplier, &output_shift);

    int reverse_shift_result_1 = -input1_shift;
    int reverse_shift_result_2 = -input2_shift;
    int left_shift_1           = reverse_shift_result_1 > 0 ? reverse_shift_result_1 : 0;
    int right_shift_1          = reverse_shift_result_1 > 0 ? 0 : -reverse_shift_result_1;
    int left_shift_2           = reverse_shift_result_2 > 0 ? reverse_shift_result_2 : 0;
    int right_shift_2          = reverse_shift_result_2 > 0 ? 0 : -reverse_shift_result_2;

    const int input1_left_shift  = (1 << left_shift) * ((1 << left_shift_1));
    const int input2_left_shift  = (1 << left_shift) * ((1 << left_shift_2));
    const int output_left_shift  = -output_shift > 0 ? -output_shift : 0;
    const int output_right_shift = -output_shift > 0 ? 0 : output_shift;

    // activation
    int outputActivationMin = 0, outputActivationMax = 0;
    CalculateActivationRangeUint8(add->activationType(), add->outputQuantizedParam()->zeroPoint(),
                                  add->outputQuantizedParam()->scale(), &outputActivationMin, &outputActivationMax);

    // write buffer
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    mConstBuffer = [context newDeviceBuffer:14 * sizeof(int) access:CPUWriteOnly];
    auto buffer  = (int *)mConstBuffer.contents;
    buffer[0]    = input1_offset;
    buffer[1]    = input2_offset;
    buffer[2]    = output_offset;
    buffer[3]    = input1_multiplier;
    buffer[4]    = input2_multiplier;
    buffer[5]    = output_multiplier;
    buffer[6]    = right_shift_1;
    buffer[7]    = right_shift_2;
    buffer[8]    = input1_left_shift;
    buffer[9]    = input2_left_shift;
    buffer[10]   = output_left_shift;
    buffer[11]   = output_right_shift;
    buffer[12]   = outputActivationMin;
    buffer[13]   = outputActivationMax;
}

ErrorCode MetalQuantizedAdd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
    NSUInteger count = output->elementSize();

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"quantized_add" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
    [context dispatchEncoder:encoder threads:{ count, 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalQuantizedAddCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalQuantizedAdd(backend, op->main_as_QuantizedAdd());
    }
};
REGISTER_METAL_OP_CREATOR(MetalQuantizedAddCreator, OpType_QuantizedAdd);
} // namespace MNN

#endif /* MNN_METAL_ENABLED */
