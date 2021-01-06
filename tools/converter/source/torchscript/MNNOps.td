//
//  MNNOps.td
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNOPS_TD
#define MNNOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MNN_Dialect : Dialect {
  let name = "mnn";
  let cppNamespace = "::mlir::mnn";
}

class MNN_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MNN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// MNN Operations
//===----------------------------------------------------------------------===//

def MNNTensor : AnyTypeOf<[AnyTensor, AnyInteger, AnyFloat]>;

def ConstOp : MNN_Op<"Const", [NoSideEffect]> {
  let summary = "tensor const";
  let description = [{
  }];

  let arguments = (ins ElementsAttr:$value);

  let results = (outs AnyTensor);
}

def ReturnOp : MNN_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyTensor>:$input);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def BinaryOp : MNN_Op<"binary", [NoSideEffect]> {
  let summary = "element-wise binary operation";
  let description = [{
  }];

  let arguments = (ins MNNTensor:$lhs, MNNTensor:$rhs, I32Attr:$type);
  let results = (outs MNNTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def F32ElementsAttr : FloatElementsAttr<32>;

def ConvolutionOp : MNN_Op<"convolution"> {
  let summary = "convolution";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, I32Attr:$dilateX, I32Attr:$dilateY,
                       I32Attr:$strideX, I32Attr:$strideY, I32Attr:$kernelX, I32Attr:$kernelY,
                       I32Attr:$padX, I32Attr:$padY, I32Attr:$group, I32Attr:$outputCount,
                       BoolAttr:$relu, I32Attr:$padMode, BoolAttr:$relu6, I32Attr:$inputCount,
                       F32ElementsAttr:$weight, F32ElementsAttr:$bias);

  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Attribute> attrs, Value arg">
  ];
}

def MAXPOOL : I32EnumAttrCase<"MAXPOOL", 0>;
def AVGPOOL : I32EnumAttrCase<"AVGPOOL", 1>;
def PoolType : I32EnumAttr<"mlir::mnn::PoolType", "", [MAXPOOL, AVGPOOL]>;

def CAFFE : I32EnumAttrCase<"CAFFE", 0>;
def VALID : I32EnumAttrCase<"VALID", 0>;
def SAME : I32EnumAttrCase<"SAME", 0>;
def PoolPadType : I32EnumAttr<"mlir::mnn::PoolPadType", "", [CAFFE, VALID, SAME]>;

def PoolOp : MNN_Op<"pool"> {
  let summary = "max pool 2d";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, I32Attr:$kernelX, I32Attr:$kernelY,
                       I32Attr:$strideX, I32Attr:$strideY, I32Attr:$padX, I32Attr:$padY,
                       BoolAttr:$isGlobal, PoolType:$type, PoolPadType:$padType,
                       BoolAttr:$ceilModel);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Attribute> attrs, Value arg">
  ];
}

def FlattenOp : MNN_Op<"flatten", [NoSideEffect]> {
  let summary = "flatten";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, I32Attr:$start_dim, I32Attr:$end_dim);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def ReluOp : MNN_Op<"relu"> {
  let summary = "relu";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, DefaultValuedAttr<F32Attr, "0.">:$slope);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def BatchNormOp : MNN_Op<"batch_norm"> {
  let summary = "batch norm";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, I32Attr:$channel,
                   F32ElementsAttr:$slopeData, F32ElementsAttr:$meanData,
                   F32ElementsAttr:$varData, F32ElementsAttr:$biasData,
                   DefaultValuedAttr<F32ElementsAttr, "0.">:$Adata,
                   DefaultValuedAttr<F32ElementsAttr, "0.">:$Bdata,
                   DefaultValuedAttr<F32Attr, "0.001">:$epsilon);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def TransposeOp : MNN_Op<"transpose"> {
  let summary = "transpose";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, I32Tensor:$perm);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def MatMulOp : MNN_Op<"matmul"> {
  let summary = "matmul";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$inputA, AnyTensor:$inputB,
                       DefaultValuedAttr<BoolAttr, "false">:$transposeA,
                       DefaultValuedAttr<BoolAttr, "false">:$transposeB,
                       DefaultValuedAttr<F32ElementsAttr, "0.">:$weight,
                       DefaultValuedAttr<F32ElementsAttr, "0.">:$bias);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}
/*
def AddmmOp : MNN_Op<"addmm"> {
  let summary = "addmm";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$mat1, AnyTensor:$mat2,
                       AnyTypeOf<[AnyInteger, AnyFloat]>:$beta, AnyTypeOf<[AnyInteger, AnyFloat]>:$alpha);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

*/
#endif // MNNOPS_TD
