//
//  ConvertToMNN.cpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNDialect.hpp"
#include "TorchScriptDialect.hpp"
#include "Passes.hpp"
#include "MNN_generated.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;


namespace {
static std::map<std::string, MNN::BinaryOpOperation> gMaps{
    {"add", MNN::BinaryOpOperation_ADD}, {"sum", MNN::BinaryOpOperation_ADD},
    {"sub", MNN::BinaryOpOperation_SUB}, {"div", MNN::BinaryOpOperation_REALDIV},
    {"mul", MNN::BinaryOpOperation_MUL}, {"pow", MNN::BinaryOpOperation_POW},
    {"equal", MNN::BinaryOpOperation_EQUAL}, {"less", MNN::BinaryOpOperation_LESS},
    {"greater", MNN::BinaryOpOperation_GREATER}, {"max", MNN::BinaryOpOperation_MAXIMUM},
    {"min", MNN::BinaryOpOperation_MINIMUM},
};
//===----------------------------------------------------------------------===//
// TorchScriptToMNN RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//
struct ConstantOpLowering : public OpRewritePattern<torchscript::ConstantOp> {
  using OpRewritePattern<torchscript::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(torchscript::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    auto val = op.value();
    if (op.getType().isa<mlir::VectorType>() || !val) {
      rewriter.eraseOp(op);
      return success();
    }
    DenseElementsAttr elemAttr = val->dyn_cast<DenseElementsAttr>();
    if (elemAttr) {
    } else if (auto intAttr = val->dyn_cast<IntegerAttr>()) {
      elemAttr = rewriter.getI32TensorAttr({static_cast<int>(intAttr.getInt())});
    } else if (auto floatAttr = val->dyn_cast<FloatAttr>()) {
      elemAttr = DenseFPElementsAttr::get(RankedTensorType::get(1, rewriter.getF32Type()), {floatAttr.getValue()});
    } else {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<mnn::ConstOp>(op, elemAttr.getType(), elemAttr);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// TorchScriptToMNN RewritePatterns: Unary operations
//===----------------------------------------------------------------------===//
template <typename SrcUnaryOp, typename DstUnaryOp>
struct UnaryOpLowering : public ConversionPattern {
  UnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(SrcUnaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<DstUnaryOp>(op, operands);
    return success();
  }
};
using ReluOpLowering = UnaryOpLowering<torchscript::ReluOp, mnn::ReluOp>;
// using TransposeOpLowering = UnaryOpLowering<torchscript::TransposeOp, mnn::TransposeOp>;
// using ReturnOpLowering = UnaryOpLowering<torchscript::ReturnOp, mnn::ReturnOp>;
// using BatchNormOpLowering = UnaryOpLowering<torchscript::BatchNormOp, mnn::BatchNormOp>;
//===----------------------------------------------------------------------===//
// TorchScriptToMNN RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//
template <typename BinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
      std::string opName = BinaryOp::getOperationName().substr(12, -1).str();
      auto opType = rewriter.getI32IntegerAttr(static_cast<int>(gMaps[opName]));
      auto lhs = operands[0], rhs = operands[1];
      rewriter.replaceOpWithNewOp<mnn::BinaryOp>(op, lhs.getType(), lhs, rhs, opType);
      return success();
  }
};
using AddOpLowering = BinaryOpLowering<torchscript::AddOp>;
using MulOpLowering = BinaryOpLowering<torchscript::MulOp>;
//===----------------------------------------------------------------------===//
// TorchScriptToMNN RewritePatterns: Return operations
//===----------------------------------------------------------------------===//
struct ReturnOpLowering : public OpRewritePattern<torchscript::ReturnOp> {
  using OpRewritePattern<torchscript::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(torchscript::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mnn::ReturnOp>(op, op.input());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// TorchScriptToMNN RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//
std::vector<Attribute> ValueToAttr(const Value& v) {
    std::vector<Attribute> attrs;
    auto constantV = v.getDefiningOp<torchscript::ConstantOp>();
    if (!constantV) {
        return attrs;
    }
    auto attr = constantV.value();
    if (!attr) {
      return attrs;
    }
    if (constantV.getType().isa<mlir::VectorType>()) {
      auto denseAttr = attr->dyn_cast<DenseElementsAttr>();
      for (auto iter = denseAttr.attr_value_begin(); iter != denseAttr.attr_value_end(); iter++) {
          attrs.push_back(*iter);
      }
      return attrs;
    } else {
      attrs.push_back(attr.getValue());
    }
    return attrs;
}

template <typename T>
T getAttr(const Value& v, int idx = 0) {
  auto attrs = ValueToAttr(v);
  if (attrs.size() <= idx) {
    emitError(v.getLoc(), "Value don't have right Attribute!");
  }
  auto res = attrs[idx].dyn_cast<T>();
  if (!res) {
    emitError(v.getLoc(), "Attribute Type is Invalid!");
  }
  return res;
}

struct ConvolutionOpLowering : public OpRewritePattern<torchscript::ConvolutionOp> {
  using OpRewritePattern<torchscript::ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(torchscript::ConvolutionOp op,
                                PatternRewriter &rewriter) const final {
    // auto weight = ValueToAttr(op.weight())[0].dyn_cast<DenseElementsAttr>();
    auto weight = getAttr<DenseFPElementsAttr>(op.weight());
    // weight format : NCHW
    auto weightDims = weight.getType().getShape();
    auto outputCount = rewriter.getI32IntegerAttr(weightDims[0]);
    auto inputCount = rewriter.getI32IntegerAttr(weightDims[1]);
    auto kernelY = rewriter.getI32IntegerAttr(weightDims[2]);
    auto kernelX = rewriter.getI32IntegerAttr(weightDims[3]);
    auto biasArray = ValueToAttr(op.bias());
    DenseElementsAttr bias;
    if (biasArray.size()) {
      bias = biasArray[0].dyn_cast<DenseElementsAttr>();
    } else {
      bias = rewriter.getZeroAttr(RankedTensorType::get(static_cast<int64_t>(weightDims[0]), rewriter.getF32Type())).dyn_cast<DenseElementsAttr>();
    }
    auto dilateX = getAttr<IntegerAttr>(op.dialation(), 0);
    auto dilateY = getAttr<IntegerAttr>(op.dialation(), 1);
    auto strideX = getAttr<IntegerAttr>(op.stride(), 0);
    auto strideY = getAttr<IntegerAttr>(op.stride(), 1);
    auto padX = getAttr<IntegerAttr>(op.padding(), 0);
    auto padY = getAttr<IntegerAttr>(op.padding(), 1);
    auto group = getAttr<IntegerAttr>(op.groups());
    auto relu = rewriter.getBoolAttr(false);
    auto relu6 = rewriter.getBoolAttr(false);
    auto padMode = rewriter.getI32IntegerAttr(mnn::PoolPadType::CAFFE);
    rewriter.replaceOpWithNewOp<mnn::ConvolutionOp>(op, op.input().getType(), op.input(),
                                                    dilateX, dilateY,
                                                    strideX, strideY, kernelX, kernelY,
                                                    padX, padY, group, outputCount,
                                                    relu, padMode, relu6, inputCount,
                                                    weight, bias);
    return success();
  }
};

template <typename PoolOp>
struct PoolOpLowering : public ConversionPattern {
  PoolOpLowering(MLIRContext *ctx)
      : ConversionPattern(PoolOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto input = operands[0];
    auto name = PoolOp::getOperationName();
    IntegerAttr kernelX, kernelY, strideX, strideY, padX, padY;
    BoolAttr ceilMode;
    if (name.find("adaptive") == StringRef::npos) {
      kernelX = getAttr<IntegerAttr>(operands[1], 0);
      kernelY = getAttr<IntegerAttr>(operands[1], 1);
      strideX = getAttr<IntegerAttr>(operands[2], 0);
      strideY = getAttr<IntegerAttr>(operands[2], 1);
      padX = getAttr<IntegerAttr>(operands[3], 0);
      padY = getAttr<IntegerAttr>(operands[3], 1);
      // auto dialations = getAttr<IntegerAttr>(operands[4], 0);
      ceilMode = getAttr<BoolAttr>(operands[5]);
    } else {
        auto outputX = getAttr<IntegerAttr>(operands[1], 0);
        auto outputY = getAttr<IntegerAttr>(operands[1], 1);
        auto ox = outputX.getInt(), oy = outputY.getInt();
        // TODO: fix adaptive_pooling
        kernelX = rewriter.getI32IntegerAttr(7);
        kernelY = rewriter.getI32IntegerAttr(7);
        strideX = rewriter.getI32IntegerAttr(1);
        strideY = rewriter.getI32IntegerAttr(1);
        padX = rewriter.getI32IntegerAttr(0);
        padY = rewriter.getI32IntegerAttr(0);
        ceilMode = rewriter.getBoolAttr(false);
    }
    auto poolType = name.find("max") == StringRef::npos ? mnn::PoolType::AVGPOOL : mnn::PoolType::MAXPOOL;
    auto isGlobal = rewriter.getBoolAttr(false);
    auto type = rewriter.getI32IntegerAttr(poolType);
    auto padType = rewriter.getI32IntegerAttr(mnn::PoolPadType::CAFFE);
    rewriter.replaceOpWithNewOp<mnn::PoolOp>(op, input.getType(), input,
                                             kernelX, kernelY, strideX, strideY,
                                             padX, padY, isGlobal, type, padType,
                                             ceilMode);
    return success();
  }
};
using MaxPoolOpLowering = PoolOpLowering<torchscript::MaxPool2dOp>;
using AdaptiveAvgPoolOpLowering = PoolOpLowering<torchscript::AdaptiveAvgPool2dOp>;
struct FlattenOpLowering : public OpRewritePattern<torchscript::FlattenOp> {
  using OpRewritePattern<torchscript::FlattenOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(torchscript::FlattenOp op,
                                PatternRewriter &rewriter) const final {
    auto start_dim = getAttr<IntegerAttr>(op.start_dim());
    auto end_dim = getAttr<IntegerAttr>(op.end_dim());
    rewriter.replaceOpWithNewOp<mnn::FlattenOp>(op, op.input().getType(), op.input(),
                                                start_dim, end_dim);
    return success();
  }
};
struct BatchNormOpLowering : public OpRewritePattern<torchscript::BatchNormOp> {
  using OpRewritePattern<torchscript::BatchNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(torchscript::BatchNormOp op,
                                PatternRewriter &rewriter) const final {
    auto slope = getAttr<DenseElementsAttr>(op.weight());
    auto bias = getAttr<DenseElementsAttr>(op.bias());
    auto mean = getAttr<DenseElementsAttr>(op.running_mean());
    auto var = getAttr<DenseElementsAttr>(op.running_var());
    auto epsilon = getAttr<FloatAttr>(op.eps());
    auto channel = rewriter.getI32IntegerAttr(slope.size());
    auto ab = rewriter.getZeroAttr(RankedTensorType::get(static_cast<int64_t>(slope.size()), rewriter.getF32Type())).dyn_cast<DenseElementsAttr>();
    rewriter.replaceOpWithNewOp<mnn::BatchNormOp>(op, op.input().getType(), op.input(),
                                                  channel, slope, mean, var, bias,
                                                  ab, ab, epsilon);
    return success();
  }
};
struct AddmmOpLowering : public OpRewritePattern<torchscript::AddmmOp> {
  using OpRewritePattern<torchscript::AddmmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(torchscript::AddmmOp op,
                                PatternRewriter &rewriter) const final {
    // out = beta * mat0 + alpha * (mat1 @ mat2)
    auto mat0 = op.input();
    auto mat1 = op.mat1();
    auto mat2 = op.mat2();
    auto beta = op.beta();
    auto alpha = op.alpha();

    auto betaVal = getAttr<IntegerAttr>(beta).getInt();
    auto alphaVal = getAttr<IntegerAttr>(alpha).getInt();
    auto mulType = rewriter.getI32IntegerAttr(static_cast<int>(gMaps["mul"]));
    if (betaVal != 1) {
      mat0 = rewriter.create<mnn::BinaryOp>(op.getLoc(), mat0.getType(), mat0, beta, mulType);
    }
    mat1 = rewriter.create<mnn::MatMulOp>(op.getLoc(), ArrayRef({mat1, mat2}));
    if (alphaVal != 1) {
      mat1 = rewriter.create<mnn::BinaryOp>(op.getLoc(), mat1.getType(), mat1, alpha, mulType);
    }
    auto addType = rewriter.getI32IntegerAttr(static_cast<int>(gMaps["add"]));
    auto addOp = rewriter.create<mnn::BinaryOp>(op.getLoc(), mat0.getType(), mat0, mat1, addType);
    rewriter.replaceOp(op, {addOp});
    return success();
  }
};
struct TransposeOpLowering : public OpRewritePattern<torchscript::TransposeOp> {
  using OpRewritePattern<torchscript::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(torchscript::TransposeOp op,
                                PatternRewriter &rewriter) const final {
    auto permAttr = rewriter.getI32TensorAttr({1, 0});
    auto perm = rewriter.create<mnn::ConstOp>(op.getLoc(), permAttr.getType(), permAttr);
    rewriter.replaceOpWithNewOp<mnn::TransposeOp>(op, op.input().getType(), op.input(), perm);
    return success();
  }
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct TorchScriptToMNNPass
    : public PassWrapper<TorchScriptToMNNPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mnn::MNNDialect>();
  }
  void runOnFunction() final;
};
} // end anonymous namespace.

void TorchScriptToMNNPass::runOnFunction() {
  auto function = getFunction();

  // We only lower the main function as we expect that all other functions have
  // been inlined.
  if (function.getName() != "main_graph")
    return;

  // remove module arguments
  for (int i = 0; i < function.getNumArguments(); i++) {
    auto arg = function.getArgument(i);
    if (arg.getType().isa<OpaqueType>()) {
      function.eraseArgument(i);
    }
  }
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<mnn::MNNDialect>();
  target.addLegalDialect<torchscript::TorchScriptDialect>();

  target.addIllegalDialect<torchscript::TorchScriptDialect>();
  target.addLegalOp<torchscript::ConstantOp>();
  // target.addLegalOp<torchscript::AddmmOp>();
  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  OwningRewritePatternList patterns;
  patterns.insert<AddOpLowering, MulOpLowering,
                  ReturnOpLowering, ReluOpLowering, TransposeOpLowering,
                  ConvolutionOpLowering, MaxPoolOpLowering, AdaptiveAvgPoolOpLowering,
                  FlattenOpLowering, BatchNormOpLowering, AddmmOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();

  // dealwith constant op
  target.addIllegalOp<torchscript::ConstantOp>();
  OwningRewritePatternList constPattern;
  constPattern.insert<ConstantOpLowering>(&getContext());
  if (failed(applyPartialConversion(getFunction(), target, constPattern)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::torchscript::createConvertToMNNPass() {
  return std::make_unique<TorchScriptToMNNPass>();
}
