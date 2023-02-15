//
//  SourceTargetCodeGen.cpp
//  MNNCodegen
//
//  Created by MNN on 2020/11/27.
//

#include "cpu/CPUAst.hpp"
#include <sstream>

using namespace AST;
std::string PrototypeAST::codegen(SourceTarget *target) {
    std::stringstream ss;
    ss << target->getIndent();
    ss << "void " << Name << "(";
    ss << "float** inputs, float** outputs";
    ss << ")\n";
    return ss.str();
}

std::string FunctionAST::codegen(SourceTarget* target) {
    std::stringstream ss;
    ss << Proto->codegen(target) << "{\n";
    target->addIndent();
    ss << Body->codegen(target);
    target->subIndent();
    ss << "}\n";
    return ss.str();
}

std::string ListExpr::codegen(SourceTarget* target) {
    std::stringstream ss;
    for (auto& expr : exprs) {
        ss << expr->codegen(target);
    }
    return ss.str();
}

std::string VarExpr::codegen(SourceTarget* target) {
}

std::string ForExpr::codegen(SourceTarget* target) {
    std::stringstream ss;
    ss << target->getIndent() << "for (int ";
    ss << VarName << " = " << Start->codegen(target) << "; ";
    ss << VarName << " < " << End->codegen(target) << "; ";
    ss << VarName << " += " << Step->codegen(target) << ") {\n";
    target->addIndent();
    ss << Body->codegen(target);
    target->subIndent();
    ss << target->getIndent() << "}\n";
    return ss.str();
}

std::string IfExpr::codegen(SourceTarget* target) {
}

std::string CallExpr::codegen(SourceTarget* target) {
}

std::string AssignExpr::codegen(SourceTarget* target) {
    std::stringstream ss;
    ss << target->getIndent() << LHS->codegen(target) << " = " << RHS->codegen(target) << ";\n";
    return ss.str();
}

std::string BinaryExpr::codegen(SourceTarget *target) {
    std::stringstream ss;
    auto l = LHS->codegen(target);
    auto r = RHS->codegen(target);
    switch (Op) {
        case MNN::BinaryOpOperation_ADD:
            ss << "(" << l << " + " << r << ")";
            break;
        case MNN::BinaryOpOperation_SUB:
            ss << "(" << l << " - " << r << ")";
            break;
        case MNN::BinaryOpOperation_MUL:
            ss << "(" << l << " * " << r << ")";
            break;
        case MNN::BinaryOpOperation_DIV:
        case MNN::BinaryOpOperation_REALDIV:
            ss << "(" << l << " / " << r << ")";
            break;
        case MNN::BinaryOpOperation_FLOORDIV:
            ss << "floor(" << l << " / " << r << ")";
            break;
        case MNN::BinaryOpOperation_POW:
            ss << "pow(" << l << ", " << r << ")";
            break;
        case MNN::BinaryOpOperation_MINIMUM:
            ss << "fmin(" << l << ", " << r << ")";
            break;
        case MNN::BinaryOpOperation_MAXIMUM:
            ss << "fmax(" << l << ", " << r << ")";
            break;
        case MNN::BinaryOpOperation_GREATER:
            ss << "(" << l << " > " << r << ")";
            break;
        case MNN::BinaryOpOperation_GREATER_EQUAL:
            ss << "(" << l << " >= " << r << ")";
            break;
        case MNN::BinaryOpOperation_LESS:
            ss << "(" << l << " < " << r << ")";
            break;
        case MNN::BinaryOpOperation_LESS_EQUAL:
            ss << "(" << l << " <= " << r << ")";
            break;
        case MNN::BinaryOpOperation_EQUAL:
            ss << "(" << l << " == " << r << ")";
            break;
        default:
            MNN_ASSERT(false);
    }
    return ss.str();
}

std::string ReluExpr::codegen(SourceTarget *target) {
    std::stringstream ss;
    auto x = Operand->codegen(target);
    if (maxVal == 0.f) {
        // slope = minVal
        // relu(x) = ((x < 0) * slope * x + (x >= 0) * x)
        ss << "((" << x << " < 0 ) * " << minVal << " * " << x << " + (" << x << " >= 0 ) * " << x << ")";
    } else {
        // relu6(x) = min(max(x, minv), maxv)
        ss << "fmin(fmax(" << x << ", " << minVal << "), " << maxVal << ")";
    }
    return ss.str();
}

std::string UnaryExpr::codegen(SourceTarget *target) {
    std::stringstream ss;
    auto x = Operand->codegen(target);
    switch (Op) {
        case MNN::UnaryOpOperation_ABS:
            ss << "abs(" << x << ")";
            break;
        case MNN::UnaryOpOperation_FLOOR:
            ss << "floor(" << x << ")";
            break;
        case MNN::UnaryOpOperation_CEIL:
            ss << "ceil(" << x << ")";
            break;
        case MNN::UnaryOpOperation_SQRT:
            ss << "sqrt(" << x << ")";
            break;
        case MNN::UnaryOpOperation_EXP:
            ss << "exp(" << x << ")";
            break;
        case MNN::UnaryOpOperation_LOG:
            ss << "log(" << x << ")";
            break;
        case MNN::UnaryOpOperation_SIN:
            ss << "sin(" << x << ")";
            break;
        case MNN::UnaryOpOperation_COS:
            ss << "cos(" << x << ")";
            break;
        case MNN::UnaryOpOperation_ROUND:
            ss << "round(" << x << ")";
            break;
        case MNN::UnaryOpOperation_NEG:
            ss << "(-" << x << ")";
            break;
        case MNN::UnaryOpOperation_SQUARE:
            ss << "(" << x << " * " << x << ")";
            break;
        case MNN::UnaryOpOperation_RSQRT:
            ss << "(1.f / sqrt(" << x << "))";
            break;
        case MNN::UnaryOpOperation_RECIPROCAL:
            ss << "(1.f / " << x << ")";
            break;
        case MNN::UnaryOpOperation_SIGMOID:
            ss << "(1.f / (1.f + exp(-" << x << ")))";
            break;
        case MNN::UnaryOpOperation_TANH:
            ss << "tanh(" << x << ")";
            break;
        default:
            MNN_ASSERT(false);
    }
    return ss.str();
}

std::string SubscriptExpr::codegen(SourceTarget *target) {
    std::stringstream ss;
    ss << Base->codegen(target) << "[" << Offset->codegen(target) << "]";
    return ss.str();
}

std::string VariableExpr::codegen(SourceTarget *target) {
    std::stringstream ss;
    ss << Name;
    return ss.str();
}

std::string NumberExpr::codegen(SourceTarget *target) {
    std::stringstream ss;
    switch (mType) {
        case FP32:
            ss << mVal.f32Val;
            break;
        case FP64:
            ss << mVal.f64Val;
            break;
        case INT32:
            ss << mVal.i32Val;
            break;
        case INT64:
            ss << mVal.i64Val;
            break;
        default:
            return nullptr;
    }
    return ss.str();
}
