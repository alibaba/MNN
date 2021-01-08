//
//  TorchScriptOps.td
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TORCHSCRIPTOPS_TD
#define TORCHSCRIPTOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TorchScript_Dialect : Dialect {
  let name = "torchscript";
  let cppNamespace = "::mlir::torchscript";
}

class TorchScript_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TorchScript_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TorchScript Operations
//===----------------------------------------------------------------------===//
def ConstantOp : TorchScript_Op<"constant", [NoSideEffect]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = torchscript.constant () {value=1.0 : f32} : () -> f32
    ```
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$value);

  let results = (outs AnyType);
}

def ListOp : TorchScript_Op<"list", [NoSideEffect]> {
  let summary = "list";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$args);

  let results = (outs AnyType);

  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def TensorOp : TorchScript_Op<"tensor", [NoSideEffect]> {
  let summary = "tensor";
  let description = [{
      tensor(data, dtype=None, device=None, requires_grad=False)
  }];

  let arguments = (ins AnyType:$value, AnyType:$dtype, AnyType:$device, I1:$req_grad);

  let results = (outs AnyTensor);

  let hasCanonicalizer = 1;

  let builders = [
  ];
}

def MulOp : TorchScript_Op<"mul", [NoSideEffect]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def AddOp : TorchScript_Op<"add", [NoSideEffect]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, I32:$alpha);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def GetOp : TorchScript_Op<"get", [NoSideEffect]> {
  let summary = "get attribute";
  let description = [{
  }];

  // let arguments = (ins StrAttr:$attr, OptionalAttr<AnyAttr>:$data, AnyType:$obj);
  let arguments = (ins AnyAttr:$attr, AnyType:$obj);
  let results = (outs AnyType);
  let hasCanonicalizer = 1;
}

def ConvolutionOp : TorchScript_Op<"convolution", [NoSideEffect]> {
  let summary = "convolution";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$weight, AnyTypeOf<[AnyTensor, NoneType]>:$bias,
                       VectorOf<[AnyInteger]>:$stride, VectorOf<[AnyInteger]>:$padding, VectorOf<[AnyInteger]>:$dialation,
                       I1:$transposed, VectorOf<[AnyInteger]>:$output_padding, AnyInteger:$groups,
                       I1:$benchmark, I1:$deterministic, I1:$cudnn_enable);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def BatchNormOp : TorchScript_Op<"batch_norm", [NoSideEffect]> {
  let summary = "batch norm";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, AnyTypeOf<[AnyTensor, NoneType]>:$weight, AnyTypeOf<[AnyTensor, NoneType]>:$bias,
                       AnyTypeOf<[AnyTensor, NoneType]>:$running_mean, AnyTypeOf<[AnyTensor, NoneType]>:$running_var,
                       I1:$training, AnyFloat:$momentum, AnyFloat:$eps, I1:$cudnn_enable);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def MaxPool2dOp : TorchScript_Op<"max_pool2d", [NoSideEffect]> {
  let summary = "max pool 2d";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, VectorOf<[AnyInteger]>:$kernel_size, VectorOf<[AnyInteger]>:$stride,
                       VectorOf<[AnyInteger]>:$padding, VectorOf<[AnyInteger]>:$dialation, I1:$ceil_mode);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def ReluOp : TorchScript_Op<"relu", [NoSideEffect]> {
  let summary = "relu";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def TransposeOp : TorchScript_Op<"transpose", [NoSideEffect]> {
  let summary = "transpose";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def AdaptiveAvgPool2dOp : TorchScript_Op<"adaptive_avg_pool2d", [NoSideEffect]> {
  let summary = "adaptive_avg_pool2d";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, VectorOf<[AnyInteger]>:$output_size);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def AddmmOp : TorchScript_Op<"addmm", [NoSideEffect]> {
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

def FlattenOp : TorchScript_Op<"flatten", [NoSideEffect]> {
  let summary = "flatten";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$input, AnyInteger:$start_dim, AnyInteger:$end_dim);
  let results = (outs AnyTensor);

  let builders = [
    OpBuilder<"ArrayRef<Value> args">
  ];
}

def ReturnOp : TorchScript_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation.
  }];

  let arguments = (ins Variadic<AnyTensor>:$input);

  let builders = [
    OpBuilder<"", [{ build($_builder, $_state, llvm::None); }]>
  ];
}
#endif // TORCHSCRIPTOPS_TD
