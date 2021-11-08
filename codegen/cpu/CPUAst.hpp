
#include <cassert>
#include <map>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "PluginModule.hpp"

#ifdef MNN_CODEGEN_LLVM
#include "llvm/IR/Type.h"
#include "llvm/IR/Function.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
using namespace llvm;
using namespace llvm::orc;
#endif

#ifdef MNN_CODEGEN_LLVM
class LLVMTarget {
public:
    LLVMTarget(std::string name) {
        llvmContext.reset(new LLVMContext);
        llvmBuilder = std::make_unique<IRBuilder<>>(*llvmContext.get());
        llvmModule = std::make_unique<Module>(name, *llvmContext.get());
        llvmModule->setTargetTriple("x86_64-apple-macosx11.0.0");
    }
    ~LLVMTarget() {}
    Module* getModule() {
        return llvmModule.get();
    }
    LLVMContext& getContext() {
        return *llvmContext.get();
    }
    IRBuilder<>* getBuilder() {
        return llvmBuilder.get();
    }
    ThreadSafeModule getThreadSafeModule() {
        return ThreadSafeModule(std::move(llvmModule), std::move(llvmContext));
    }
private:
    std::unique_ptr<LLVMContext> llvmContext;
    std::unique_ptr<IRBuilder<>> llvmBuilder;
    std::unique_ptr<Module> llvmModule;
};
#endif

#ifdef MNN_CODEGEN_C
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
#endif

#ifdef MNN_CODEGEN_LLVM
    #define LLVM_CODEGEN Value *codegen(LLVMTarget* target) override;
#else
    #define LLVM_CODEGEN
#endif
#ifdef MNN_CODEGEN_C
    #define C_CODEGEN std::string codegen(SourceTarget* target) override;
#else
    #define C_CODEGEN
#endif

#define CODEGEN_FUNCS \
        LLVM_CODEGEN  \
        C_CODEGEN

namespace AST {
/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
    virtual ~ExprAST() = default;

#ifdef MNN_CODEGEN_LLVM
    virtual Value *codegen(LLVMTarget* target) = 0;
#endif
#ifdef MNN_CODEGEN_C
    virtual std::string codegen(SourceTarget* target) = 0;
#endif
private:
    friend class PluginModule;
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
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
    NumberExprAST(float Val) : mType(FP32) { mVal.f32Val = Val; }
    NumberExprAST(double Val) : mType(FP64) { mVal.f64Val = Val; }
    NumberExprAST(int8_t Val) : mType(INT8) { mVal.i8Val = Val; }
    NumberExprAST(int16_t Val) : mType(INT16) { mVal.i16Val = Val; }
    NumberExprAST(int32_t Val) : mType(INT32) { mVal.i32Val = Val; }
    NumberExprAST(int64_t Val) : mType(INT64) { mVal.i64Val = Val; }
    NumberExprAST(uint8_t Val) : mType(UINT8) { mVal.ui8Val = Val; }
    NumberExprAST(uint16_t Val) : mType(UINT16) { mVal.ui16Val = Val;}
    NumberExprAST(uint32_t Val) : mType(UINT32) { mVal.ui32Val = Val;}
    NumberExprAST(uint64_t Val) : mType(UINT64) { mVal.ui64Val = Val;}
    CODEGEN_FUNCS
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
    std::string Name;

public:
    VariableExprAST() = default;
    VariableExprAST(const std::string &Name) : Name(Name) {}
    const std::string &getName() const { return Name; }
#ifdef MNN_CODEGEN_LLVM
    virtual Value* getRef(LLVMTarget* target);
#endif
    CODEGEN_FUNCS
};

class SubscriptExprAST : public VariableExprAST {
    std::unique_ptr<ExprAST> Base, Offset;
public:
    SubscriptExprAST(std::unique_ptr<ExprAST> Base, std::unique_ptr<ExprAST> Offset)
        : Base(std::move(Base)), Offset(std::move(Offset)) {}
    SubscriptExprAST(std::unique_ptr<ExprAST> Base, const std::string& Offset)
        : Base(std::move(Base)), Offset(std::make_unique<VariableExprAST>(Offset)) {}
    SubscriptExprAST(std::unique_ptr<ExprAST> Base, int Offset)
        : Base(std::move(Base)), Offset(std::make_unique<NumberExprAST>(Offset)) {}
    SubscriptExprAST(const std::string& Base, const std::string& Offset)
        : Base(std::make_unique<VariableExprAST>(Base)), Offset(std::make_unique<VariableExprAST>(Offset)) {}
    SubscriptExprAST(const std::string& Base, int Offset)
        : Base(std::make_unique<VariableExprAST>(Base)), Offset(std::make_unique<NumberExprAST>(Offset)) {}
    SubscriptExprAST(const std::string& Base, std::unique_ptr<ExprAST> Offset)
        : Base(std::make_unique<VariableExprAST>(Base)), Offset(std::move(Offset)) {}
#ifdef MNN_CODEGEN_LLVM
    Value *getRef(LLVMTarget* target) override;
#endif
    CODEGEN_FUNCS
};

/// UnaryExprAST - Expression class for a unary operator.
class UnaryExprAST : public ExprAST {
    MNN::UnaryOpOperation Op;
    std::unique_ptr<ExprAST> Operand;
public:
    UnaryExprAST(MNN::UnaryOpOperation Op, std::unique_ptr<ExprAST> Operand)
        : Op(Op), Operand(std::move(Operand)) {}

    CODEGEN_FUNCS
};

class ReluExprAST : public ExprAST {
    float minVal, maxVal;
    std::unique_ptr<ExprAST> Operand;
public:
    ReluExprAST(float minVal, float maxVal, std::unique_ptr<ExprAST> Operand)
        : minVal(minVal), maxVal(maxVal), Operand(std::move(Operand)) {}

    CODEGEN_FUNCS
};


/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
    MNN::BinaryOpOperation Op;
    std::unique_ptr<ExprAST> LHS, RHS;

public:
    BinaryExprAST(MNN::BinaryOpOperation Op, std::unique_ptr<ExprAST> LHS,
                    std::unique_ptr<ExprAST> RHS)
          : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    CODEGEN_FUNCS
};

class AssignExprAST : public ExprAST {
    std::unique_ptr<ExprAST> LHS, RHS;
public:
    AssignExprAST(std::unique_ptr<ExprAST> LHS,
                  std::unique_ptr<ExprAST> RHS)
        : LHS(std::move(LHS)), RHS(std::move(RHS)) {}

    CODEGEN_FUNCS
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;

public:
    CallExprAST(const std::string &Callee,
                std::vector<std::unique_ptr<ExprAST>> Args)
        : Callee(Callee), Args(std::move(Args)) {}

    CODEGEN_FUNCS
};

/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
    std::unique_ptr<ExprAST> Cond, Then, Else;

public:
    IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then,
              std::unique_ptr<ExprAST> Else)
        : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}

    CODEGEN_FUNCS
};

/// ForExprAST - Expression class for for/in.
class ForExprAST : public ExprAST {
    std::string VarName;
    std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
    ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
               std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
               std::unique_ptr<ExprAST> Body)
        : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
            Step(std::move(Step)), Body(std::move(Body)) {}

    CODEGEN_FUNCS
};

/// VarExprAST - Expression class for var/in
class VarExprAST : public ExprAST {
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    std::unique_ptr<ExprAST> Body;

public:
    VarExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::unique_ptr<ExprAST> Body)
      : VarNames(std::move(VarNames)), Body(std::move(Body)) {}

    CODEGEN_FUNCS
};

/// ListExprAST - Expression class for expr list
class ListExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> exprs;
public:
    ListExprAST() = default;
    ListExprAST(std::vector<std::unique_ptr<ExprAST>> exprs)
        : exprs(std::move(exprs)) {}
    void push_back(std::unique_ptr<ExprAST> expr) {
        exprs.emplace_back(std::move(expr));
    }

    CODEGEN_FUNCS
};

class PrototypeAST {
    std::string Name;
    int inputArgNum, outputArgNum;
public:
    PrototypeAST(const std::string &Name, int inputNum, int outputNum)
        : Name(Name), inputArgNum(inputNum), outputArgNum(outputNum) {}

    const std::string &getName() const { return Name; }

#ifdef MNN_CODEGEN_LLVM
    Function *codegen(LLVMTarget* target);
#endif
#ifdef MNN_CODEGEN_C
    std::string codegen(SourceTarget* target);
#endif
};

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;

public:
    FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                std::unique_ptr<ExprAST> Body)
        : Proto(std::move(Proto)), Body(std::move(Body)) {}

#ifdef MNN_CODEGEN_LLVM
    Function *codegen(LLVMTarget* target);
#endif
#ifdef MNN_CODEGEN_C
    std::string codegen(SourceTarget* target);
#endif
};
} // end CodeGen namespace
