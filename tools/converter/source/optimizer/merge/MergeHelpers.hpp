//
//  MergeHelpers.hpp
//  MNNConverter
//
//  Created by MNN on b'2020/07/20'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CONVERTER_SOURCE_OPTIMIZER_MERGE_MERGE_HELPERS_HPP_
#define CONVERTER_SOURCE_OPTIMIZER_MERGE_MERGE_HELPERS_HPP_

#include <vector>

#include <MNN/expr/Expr.hpp>

namespace MNN {
namespace helpers {

bool IsConstant(Express::EXPRP expr);
bool IsBinaryOp(Express::EXPRP expr);
bool IsUnaryOp(Express::EXPRP expr);

bool IsBinaryAdd(Express::EXPRP expr);
bool IsBinarySub(Express::EXPRP expr);
bool IsBinaryMul(Express::EXPRP expr);

bool IsBinarySquaredDifference(Express::EXPRP expr);

bool IsUnarySquare(Express::EXPRP expr);
bool IsUnaryRsqrt(Express::EXPRP expr);
bool IsUnaryNeg(Express::EXPRP expr);

bool IsReductionMean(Express::EXPRP expr);

bool IsConvolution(Express::EXPRP expr);

bool IsExpandDims(Express::EXPRP expr);

Express::EXPRP InputExpr(Express::EXPRP expr, int input_index);
Express::EXPRP OutputExpr(Express::EXPRP expr, int output_index);

std::vector<Express::VARP> OutputVars(Express::EXPRP expr);

Express::VARP ConvertLayout(Express::VARP input,
                            Express::Dimensionformat dest_layout,
                            Express::Dimensionformat src_layout);

}  // namespace MNN
}  // namespace helpers

#endif // CONVERTER_SOURCE_OPTIMIZER_MERGE_MERGE_HELPERS_HPP_
