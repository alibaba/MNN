
#include <cassert>
#include <map>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "PluginModule.hpp"


class SourceTarget {
public:
    SourceTarget() {}
    ~SourceTarget() {}
    void addIndent() { indent++; }
    void subIndent() { indent--; }
    std::string getIndent() {
        return std::string(4 * indent, ' ');
    }
private:
    int indent = 0;
};

class CTarget : public SourceTarget {
public:
    CTarget(std::string& name) {}
    ~CTarget() {}
};

namespace AST {

class Expr {
public:
    virtual ~Expr() = default;
    virtual std::string codegen(SourceTarget* target) = 0;
private:
    friend class PluginModule;
};

class NumberExpr : public Expr {
private:
    union Val {
        char chVal;
        float f32Val;
        double f64Val;
        int8_t  i8Val;
        int16_t i16Val;
        int32_t i32Val;
        int64_t i64Val;
        uint8_t  ui8Val;
        uint16_t ui16Val;
        uint32_t ui32Val;
        uint64_t ui64Val;
    } mVal;
    enum DataType {
        CHAR = 0,
        FP16,
        FP32,
        FP64,
        INT1,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT1,
        UINT8,
        UINT16,
        UINT32,
        UINT64
    };
    DataType mType;
public:
    NumberExpr(float Val) : mType(FP32) { mVal.f32Val = Val; }
    NumberExpr(double Val) : mType(FP64) { mVal.f64Val = Val; }
    NumberExpr(int8_t Val) : mType(INT8) { mVal.i8Val = Val; }
    NumberExpr(int16_t Val) : mType(INT16) { mVal.i16Val = Val; }
    NumberExpr(int32_t Val) : mType(INT32) { mVal.i32Val = Val; }
    NumberExpr(int64_t Val) : mType(INT64) { mVal.i64Val = Val; }
    NumberExpr(uint8_t Val) : mType(UINT8) { mVal.ui8Val = Val; }
    NumberExpr(uint16_t Val) : mType(UINT16) { mVal.ui16Val = Val;}
    NumberExpr(uint32_t Val) : mType(UINT32) { mVal.ui32Val = Val;}
    NumberExpr(uint64_t Val) : mType(UINT64) { mVal.ui64Val = Val;}
    CODEGEN_FUNCS
};

class VariableExpr : public Expr {
    std::string Name;

public:
    VariableExpr() = default;
    VariableExpr(const std::string &Name) : Name(Name) {}
    const std::string &getName() const { return Name; }
    std::string codegen(SourceTarget* target) override;
};

class SubscriptExpr : public VariableExpr {
    std::unique_ptr<Expr> Base, Offset;
public:
    SubscriptExpr(std::unique_ptr<Expr> Base, std::unique_ptr<Expr> Offset)
        : Base(std::move(Base)), Offset(std::move(Offset)) {}
    SubscriptExpr(std::unique_ptr<Expr> Base, const std::string& Offset)
        : Base(std::move(Base)), Offset(std::make_unique<VariableExpr>(Offset)) {}
    SubscriptExpr(std::unique_ptr<Expr> Base, int Offset)
        : Base(std::move(Base)), Offset(std::make_unique<NumberExpr>(Offset)) {}
    SubscriptExpr(const std::string& Base, const std::string& Offset)
        : Base(std::make_unique<VariableExpr>(Base)), Offset(std::make_unique<VariableExpr>(Offset)) {}
    SubscriptExpr(const std::string& Base, int Offset)
        : Base(std::make_unique<VariableExpr>(Base)), Offset(std::make_unique<NumberExpr>(Offset)) {}
    SubscriptExpr(const std::string& Base, std::unique_ptr<Expr> Offset)
        : Base(std::make_unique<VariableExpr>(Base)), Offset(std::move(Offset)) {}
    std::string codegen(SourceTarget* target) override;
};

class UnaryExpr : public Expr {
    MNN::UnaryOpOperation Op;
    std::unique_ptr<Expr> Operand;
public:
    UnaryExpr(MNN::UnaryOpOperation Op, std::unique_ptr<Expr> Operand)
        : Op(Op), Operand(std::move(Operand)) {}
    std::string codegen(SourceTarget* target) override;
};

class ReluExpr : public Expr {
    float minVal, maxVal;
    std::unique_ptr<Expr> Operand;
public:
    ReluExpr(float minVal, float maxVal, std::unique_ptr<Expr> Operand)
        : minVal(minVal), maxVal(maxVal), Operand(std::move(Operand)) {}
    std::string codegen(SourceTarget* target) override;
};

class BinaryExpr : public Expr {
    MNN::BinaryOpOperation Op;
    std::unique_ptr<Expr> LHS, RHS;
public:
    BinaryExpr(MNN::BinaryOpOperation Op, std::unique_ptr<Expr> LHS,
                    std::unique_ptr<Expr> RHS)
          : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    std::string codegen(SourceTarget* target) override;
};

class AssignExpr : public Expr {
    std::unique_ptr<Expr> LHS, RHS;
public:
    AssignExpr(std::unique_ptr<Expr> LHS, std::unique_ptr<Expr> RHS)
             : LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    std::string codegen(SourceTarget* target) override;
};

class CallExpr : public Expr {
    std::string Callee;
    std::vector<std::unique_ptr<Expr>> Args;
public:
    CallExpr(const std::string &Callee,
                std::vector<std::unique_ptr<Expr>> Args)
        : Callee(Callee), Args(std::move(Args)) {}
    std::string codegen(SourceTarget* target) override;
};

class IfExpr : public Expr {
    std::unique_ptr<Expr> Cond, Then, Else;
public:
    IfExpr(std::unique_ptr<Expr> Cond, std::unique_ptr<Expr> Then,
              std::unique_ptr<Expr> Else)
        : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}
    std::string codegen(SourceTarget* target) override;
};

class LoopExpr : public Expr {
    std::string VarName;
    std::unique_ptr<Expr> Start, End, Step, Body;
public:
    LoopExpr(const std::string &VarName, std::unique_ptr<Expr> Start,
             std::unique_ptr<Expr> End, std::unique_ptr<Expr> Step,
             std::unique_ptr<Expr> Body)
           : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
             Step(std::move(Step)), Body(std::move(Body)) {}
    std::string codegen(SourceTarget* target) override;
};

class VarExpr : public Expr {
    std::vector<std::pair<std::string, std::unique_ptr<Expr>>> VarNames;
    std::unique_ptr<Expr> Body;
public:
    VarExpr(std::vector<std::pair<std::string, std::unique_ptr<Expr>>> VarNames,
            std::unique_ptr<Expr> Body)
          : VarNames(std::move(VarNames)), Body(std::move(Body)) {}
    std::string codegen(SourceTarget* target) override;
};

class ListExpr : public Expr {
    std::vector<std::unique_ptr<Expr>> exprs;
public:
    ListExpr() = default;
    ListExpr(std::vector<std::unique_ptr<Expr>> exprs)
        : exprs(std::move(exprs)) {}
    void push_back(std::unique_ptr<Expr> expr) {
        exprs.emplace_back(std::move(expr));
    }
    std::string codegen(SourceTarget* target) override;
};

class PrototypeAST {
    std::string Name;
    int inputArgNum, outputArgNum;
public:
    PrototypeAST(const std::string &Name, int inputNum, int outputNum)
        : Name(Name), inputArgNum(inputNum), outputArgNum(outputNum) {}

    const std::string &getName() const { return Name; }
    std::string codegen(SourceTarget* target);
};

class FunctionAST {
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<Expr> Body;
public:
    FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                std::unique_ptr<Expr> Body)
        : Proto(std::move(Proto)), Body(std::move(Body)) {}
    std::string codegen(SourceTarget* target);
};
} // end AST namespace