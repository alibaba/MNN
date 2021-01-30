//
//  TorchScriptOpt.cpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "TorchScriptDialect.hpp"
#include <numeric>
using namespace mlir;
using namespace torchscript;

// constant propagation for ListOp: ListOp -> ConstantOp
struct ConstantPropagationForList : public mlir::OpRewritePattern<ListOp> {
  ConstantPropagationForList(mlir::MLIRContext *context)
      : OpRewritePattern<ListOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ListOp op,
                  mlir::PatternRewriter &rewriter) const override {
    std::vector<Attribute> attrs;
    for (auto input : op.getOperands()) {
        ConstantOp constantInput = input.getDefiningOp<ConstantOp>();
        if (!constantInput)
            return failure();
        auto attrVal = constantInput.value();
        if (!attrVal) {
            return failure();
        }
        if (auto denseAttr = attrVal->dyn_cast<DenseElementsAttr>()) {
            for (auto iter = denseAttr.attr_value_begin(); iter != denseAttr.attr_value_end(); iter++) {
                attrs.push_back(*iter);
            }
        } else {
            attrs.push_back(attrVal.getValue());
        }
    }
    auto type = op.getType().dyn_cast<VectorType>();
    auto newAttribute = mlir::DenseElementsAttr::get(type, llvm::makeArrayRef(attrs));
    auto newOp = rewriter.create<ConstantOp>(op.getLoc(), type, newAttribute);
    rewriter.replaceOp(op, { newOp });
    return success();
  }
};

// constant propagation for TensorOp: TensorOp -> ConstantOp
struct ConstantPropagationForTensor : public mlir::OpRewritePattern<TensorOp> {
  ConstantPropagationForTensor(mlir::MLIRContext *context)
      : OpRewritePattern<TensorOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(TensorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    ConstantOp constantInput = op.getOperand(0).getDefiningOp<ConstantOp>();
    if (!constantInput) {
        return failure();
    }
    auto attrVal = constantInput.value();
    if (!attrVal) {
        return failure();
    }
    auto type = op.getType().dyn_cast<TensorType>();
    auto dataAttribute = attrVal->dyn_cast<DenseElementsAttr>();
    auto newOp = rewriter.create<ConstantOp>(op.getLoc(), type, dataAttribute);
    rewriter.replaceOp(op, { newOp });
    return success();
  }
};

// constant propagation for GetOp: GetOp -> ConstantOp
struct ConstantPropagationForGet : public mlir::OpRewritePattern<GetOp> {
  ConstantPropagationForGet(mlir::MLIRContext *context)
      : OpRewritePattern<GetOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(GetOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto attrVal = op.attr();
    auto dataAttribute = attrVal.dyn_cast<DenseElementsAttr>();
    if (!dataAttribute) {
        return failure();
    }
    auto type = op.getType().dyn_cast<TensorType>();
    auto newOp = rewriter.create<ConstantOp>(op.getLoc(), type, dataAttribute);
    rewriter.replaceOp(op, { newOp });
    return success();
  }
};
void ListOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<ConstantPropagationForList>(context);
}

void TensorOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<ConstantPropagationForTensor>(context);
}

void GetOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<ConstantPropagationForGet>(context);
}
