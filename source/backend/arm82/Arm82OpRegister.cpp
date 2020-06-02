namespace MNN {

// TODO -> generate this file automatically

#ifdef MNN_CODEGEN_REGISTER

// extern void ___CPUArgMaxCreator__OpType_ArgMax__();

// void ___##creator##__##type##__()
extern void ___Arm82ConcatCreator__OpType_Concat__();
extern void ___Arm82ConvolutionCreator__OpType_Convolution__();
extern void ___Arm82ConvolutionDepthwiseCreator__OpType_ConvolutionDepthwise__();
extern void ___Arm82EltwiseCreator__OpType_Eltwise__();
extern void ___Arm82PoolingCreator__OpType_Pooling__();
extern void ___Arm82ReluCreator__OpType_ReLU__();
extern void ___Arm82ReluCreator__OpType_PReLU__();
extern void ___Arm82PaddingCreator__OpType_Padding__();
extern void ___Arm82InterpCreator__OpType_Interp__();
// extern void ___Arm82TensorConverterCreator__OpType_ConvertTensor();


void registerArm82Ops(){
    ___Arm82ConcatCreator__OpType_Concat__();
    ___Arm82ConvolutionCreator__OpType_Convolution__();
    ___Arm82ConvolutionDepthwiseCreator__OpType_ConvolutionDepthwise__();
    ___Arm82EltwiseCreator__OpType_Eltwise__();
    ___Arm82PoolingCreator__OpType_Pooling__();
    ___Arm82ReluCreator__OpType_ReLU__();
    ___Arm82ReluCreator__OpType_PReLU__();
    ___Arm82PaddingCreator__OpType_Padding__();
    ___Arm82InterpCreator__OpType_Interp__();
    // ___Arm82TensorConverterCreator__OpType_ConvertTensor();

}

#endif

} // namespace MNN
