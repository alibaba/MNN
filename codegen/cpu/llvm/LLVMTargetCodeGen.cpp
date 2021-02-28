#include "cpu/CPUAst.hpp"

using namespace AST;

static std::map<std::string, AllocaInst *> NamedValues;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

std::unique_ptr<ExprAST> LogError(const char *Str) {
    fprintf(stderr, "Error: %s\n", Str);
    return nullptr;
}

Value *LogErrorV(const char *Str) {
    LogError(Str);
    return nullptr;
}
static Value* getValueByName(const std::string& name) {
    if (name.empty()) {
        LogErrorV(std::string("Variable name: " + name + "is empty!").c_str());
    }
    if (NamedValues.find(name) == NamedValues.end()) {
        LogErrorV(std::string("Unknown variable name: " + name).c_str());
    }
    return NamedValues[name];
}

static Function *getFunction(LLVMTarget* target, std::string Name) {
    // First, see if the function has already been added to the current module.
    if (auto *F = target->getModule()->getFunction(Name)) {
        return F;
    }

    // If not, check whether we can codegen the declaration from some existing
    // prototype.
    auto FI = FunctionProtos.find(Name);
    if (FI != FunctionProtos.end()) {
        return FI->second->codegen(target);
    }

    // If no existing prototype exists, return null.
    return nullptr;
}

/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
static AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          StringRef VarName, Type* type) {
    IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                     TheFunction->getEntryBlock().begin());
    return TmpB.CreateAlloca(type, nullptr, VarName);
}

Value *NumberExprAST::codegen(LLVMTarget* target) {
    switch (mType) {
        case FP32:
            return ConstantFP::get(target->getContext(), APFloat(mVal.f32Val));
        case FP64:
            return ConstantFP::get(target->getContext(), APFloat(mVal.f64Val));
        case INT32:
            return ConstantInt::get(target->getContext(), APInt(32, mVal.i32Val, true));
        case INT64:
            return ConstantInt::get(target->getContext(), APInt(64, mVal.i64Val, true));
        default:
            return nullptr;
    }
}
Value *VariableExprAST::getRef(LLVMTarget* target) {
    return getValueByName(Name);
}
Value *VariableExprAST::codegen(LLVMTarget* target) {
    return target->getBuilder()->CreateLoad(getRef(target), Name.c_str());
}

Value* SubscriptExprAST::getRef(LLVMTarget* target) {
    return target->getBuilder()->CreateGEP(Base->codegen(target), Offset->codegen(target));
}
Value* SubscriptExprAST::codegen(LLVMTarget* target) {
    return target->getBuilder()->CreateLoad(getRef(target));
}
Value *ReluExprAST::codegen(LLVMTarget* target) {
    Value *V = Operand->codegen(target);
    if (!V) {
        return nullptr;
    }
    auto builder = target->getBuilder();
    auto& llvmContext = target->getContext();
    if (maxVal == 0.f) {
        // relu(x) = (x <= 0) * slope * x + (x > 0) * x
        auto aV = builder->CreateFMul(V, ConstantFP::get(llvmContext, APFloat((float)minVal)));
        auto gtz = builder->CreateUIToFP(builder->CreateFCmpUGT(V, ConstantFP::get(llvmContext, APFloat((float)0))), Type::getFloatTy(llvmContext));
        auto ltz = builder->CreateFSub(ConstantFP::get(llvmContext, APFloat((float)1)), gtz);
        auto l = builder->CreateFMul(ltz, aV);
        auto r = builder->CreateFMul(gtz, V);
        return builder->CreateFAdd(l, r);
    }
    // relu6(x) = min(max(x, minv), maxv)
    V = builder->CreateMaxNum(V, ConstantFP::get(llvmContext, APFloat((float)minVal)));
    return builder->CreateMinNum(V, ConstantFP::get(llvmContext, APFloat((float)maxVal)));
}
Value *UnaryExprAST::codegen(LLVMTarget* target) {
    Value *V = Operand->codegen(target);
    if (!V) {
        return nullptr;
    }
    auto builder = target->getBuilder();
    auto& llvmContext = target->getContext();
    // llvm intrinsic suppport: abs, floor, ceil, sqrt, exp, log, sin, cos, round
    //             not support: neg, square, rsqrt, tan, asin, acos, atan, reciprocal, log1p,
    //                          bnll, acosh, sinh, asinh, atanh, sign, cosh, erf, erfc,
    //                          erfinv, expm1, tanh, sigmoid,
    switch (Op) {
        // llvm intrinsic
        case MNN::UnaryOpOperation_ABS:
            return builder->CreateUnaryIntrinsic(Intrinsic::abs, V);
        case MNN::UnaryOpOperation_FLOOR:
            return builder->CreateUnaryIntrinsic(Intrinsic::floor, V);
        case MNN::UnaryOpOperation_CEIL:
            return builder->CreateUnaryIntrinsic(Intrinsic::ceil, V);
        case MNN::UnaryOpOperation_SQRT:
            return builder->CreateUnaryIntrinsic(Intrinsic::sqrt, V);
        case MNN::UnaryOpOperation_EXP:
            return builder->CreateUnaryIntrinsic(Intrinsic::exp, V);
        case MNN::UnaryOpOperation_LOG:
            return builder->CreateUnaryIntrinsic(Intrinsic::log, V);
        case MNN::UnaryOpOperation_SIN:
            return builder->CreateUnaryIntrinsic(Intrinsic::sin, V);
        case MNN::UnaryOpOperation_COS:
            return builder->CreateUnaryIntrinsic(Intrinsic::cos, V);
        case MNN::UnaryOpOperation_ROUND:
            return builder->CreateUnaryIntrinsic(Intrinsic::round, V);
        // other
        case MNN::UnaryOpOperation_NEG:
            return builder->CreateFNeg(V);
        case MNN::UnaryOpOperation_SQUARE:
            return builder->CreateFMul(V, V);
        case MNN::UnaryOpOperation_RSQRT:
            V = builder->CreateUnaryIntrinsic(Intrinsic::sqrt, V);
            return builder->CreateFDiv(ConstantFP::get(llvmContext, APFloat((float)1.0)), V);
        case MNN::UnaryOpOperation_RECIPROCAL:
            return builder->CreateFDiv(ConstantFP::get(llvmContext, APFloat((float)1.0)), V);
        case MNN::UnaryOpOperation_SIGMOID:
        {
            V = builder->CreateFNeg(V);
            V = builder->CreateUnaryIntrinsic(Intrinsic::exp, V);
            V = builder->CreateFAdd(ConstantFP::get(llvmContext, APFloat((float)1.0)), V);
            return builder->CreateFDiv(ConstantFP::get(llvmContext, APFloat((float)1.0)), V);
            // Type* type = V->getType();
            // FunctionCallee func = TheModule->getOrInsertFunction("sigmoid", FunctionType::get(type, {type}, false));
            // return builder->CreateCall(func, {V});
        }
        // function call
        case MNN::UnaryOpOperation_TANH:
        {
#ifdef USE_FUNC_CALL
            Type* type = V->getType();
            FunctionCallee func = TheModule->getOrInsertFunction("tanhf", FunctionType::get(type, {type}, false));
            return builder->CreateCall(func, {V});
#else
            auto exp = builder->CreateUnaryIntrinsic(Intrinsic::exp, V);
            auto nexp = builder->CreateFDiv(ConstantFP::get(llvmContext, APFloat((float)1.0)), exp);
            return builder->CreateFDiv(builder->CreateFSub(exp, nexp), builder->CreateFAdd(exp, nexp));
#endif
        }
        default:
            return LogErrorV(std::string("Unknown unary operator: " + std::string(MNN::EnumNameUnaryOpOperation(Op))).c_str());
    }
}

Value *BinaryExprAST::codegen(LLVMTarget* target) {
    Value *L = LHS->codegen(target);
    Value *R = RHS->codegen(target);
    if (!L || !R || L->getType() != R->getType()) {
        return nullptr;
    }
    auto builder = target->getBuilder();
    auto& llvmContext = target->getContext();
    bool isInt = L->getType()->isIntegerTy();
    switch (Op) {
        case MNN::BinaryOpOperation_ADD:
            return isInt ? builder->CreateAdd(L, R) : builder->CreateFAdd(L, R);
        case MNN::BinaryOpOperation_SUB:
            return isInt ? builder->CreateSub(L, R) : builder->CreateFSub(L, R);
        case MNN::BinaryOpOperation_MUL:
            return isInt ? builder->CreateMul(L, R) : builder->CreateFMul(L, R);
        case MNN::BinaryOpOperation_DIV:
        case MNN::BinaryOpOperation_REALDIV:
        case MNN::BinaryOpOperation_FLOORDIV:
            return builder->CreateFDiv(L, R);
        case MNN::BinaryOpOperation_POW:
            return builder->CreateBinaryIntrinsic(Intrinsic::pow, L, R);
        case MNN::BinaryOpOperation_MINIMUM:
            // Minimun and Maximun Intrinsic will meet bug when LLVM backend select
            // so use MinNum and MaxNum instead
            return builder->CreateMinNum(L, R);
        case MNN::BinaryOpOperation_MAXIMUM:
            return builder->CreateMaxNum(L, R);
        case MNN::BinaryOpOperation_GREATER:
            L = builder->CreateFCmpUGT(L, R);
            return builder->CreateUIToFP(L, Type::getFloatTy(llvmContext));
        case MNN::BinaryOpOperation_GREATER_EQUAL:
            L = builder->CreateFCmpUGE(L, R);
            return builder->CreateUIToFP(L, Type::getFloatTy(llvmContext));
        case MNN::BinaryOpOperation_LESS:
            L = builder->CreateFCmpULT(L, R);
            // Convert bool 0/1 to double 0.0 or 1.0
            return builder->CreateUIToFP(L, Type::getFloatTy(llvmContext));
        case MNN::BinaryOpOperation_LESS_EQUAL:
            L = builder->CreateFCmpULE(L, R);
            // Convert bool 0/1 to double 0.0 or 1.0
            return builder->CreateUIToFP(L, Type::getFloatTy(llvmContext));
        case MNN::BinaryOpOperation_EQUAL:
            L = builder->CreateFCmpUEQ(L, R);
            // Convert bool 0/1 to double 0.0 or 1.0
            return builder->CreateUIToFP(L, Type::getFloatTy(llvmContext));
        default:
            return LogErrorV(std::string("Unknown Binary operator: " + std::string(MNN::EnumNameBinaryOpOperation(Op))).c_str());
    }
}

Value *AssignExprAST::codegen(LLVMTarget* target) {
    VariableExprAST *LHSV = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSV) {
        return LogErrorV("destination of '=' must be a variable");
    }
    Value *Variable = LHSV->getRef(target);
    Value *Val = RHS->codegen(target);
    // if (!Val || !Variable) {
    if (!Val) {
        return nullptr;
    }
    target->getBuilder()->CreateStore(Val, Variable);
    return Val;
}

Value *CallExprAST::codegen(LLVMTarget* target) {
    // Look up the name in the global module table.
    Function *CalleeF = getFunction(target, Callee);
    if (!CalleeF) {
        return LogErrorV("Unknown function referenced");
    }

    // If argument mismatch error.
    if (CalleeF->arg_size() != Args.size()) {
        return LogErrorV("Incorrect # arguments passed");
    }

    std::vector<Value *> ArgsV;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
        ArgsV.push_back(Args[i]->codegen(target));
        if (!ArgsV.back()) {
            return nullptr;
        }
    }
    return target->getBuilder()->CreateCall(CalleeF, ArgsV);
}

Value *IfExprAST::codegen(LLVMTarget* target) {
    Value *CondV = Cond->codegen(target);
    if (!CondV) {
        return nullptr;
    }
    auto builder = target->getBuilder();
    auto& llvmContext = target->getContext();

    // Convert condition to a bool by comparing non-equal to 0.0.
    CondV = builder->CreateFCmpONE(CondV, ConstantFP::get(llvmContext, APFloat(0.0)));

    Function *TheFunction = builder->GetInsertBlock()->getParent();

    // Create blocks for the then and else cases.  Insert the 'then' block at the
    // end of the function.
    BasicBlock *ThenBB = BasicBlock::Create(llvmContext, "then", TheFunction);
    BasicBlock *ElseBB = BasicBlock::Create(llvmContext, "else");
    BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "ifcont");

    builder->CreateCondBr(CondV, ThenBB, ElseBB);

    // Emit then value.
    builder->SetInsertPoint(ThenBB);

    Value *ThenV = Then->codegen(target);
    if (!ThenV) {
        return nullptr;
    }

    builder->CreateBr(MergeBB);
    // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
    ThenBB = builder->GetInsertBlock();

    // Emit else block.
    TheFunction->getBasicBlockList().push_back(ElseBB);
    builder->SetInsertPoint(ElseBB);

    Value *ElseV = Else->codegen(target);
    if (!ElseV) {
        return nullptr;
    }

    builder->CreateBr(MergeBB);
    // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
    ElseBB = builder->GetInsertBlock();

    // Emit merge block.
    TheFunction->getBasicBlockList().push_back(MergeBB);
    builder->SetInsertPoint(MergeBB);
    PHINode *PN = builder->CreatePHI(Type::getFloatTy(llvmContext), 2, "iftmp");

    PN->addIncoming(ThenV, ThenBB);
    PN->addIncoming(ElseV, ElseBB);
    return PN;
}

// Output for-loop as:
//   var = alloca double
//   ...
//   start = startexpr
//   store start -> var
//   goto loop
// loop:
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   endcond = endexpr
//
//   curvar = load var
//   nextvar = curvar + step
//   store nextvar -> var
//   br endcond, loop, endloop
// outloop:
Value *ForExprAST::codegen(LLVMTarget* target) {
    auto builder = target->getBuilder();
    auto& llvmContext = target->getContext();

    Function *TheFunction = builder->GetInsertBlock()->getParent();

    // Create an alloca for the variable in the entry block.
    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName, Type::getInt32Ty(llvmContext));

    // Emit the start code first, without 'variable' in scope.
    Value *StartVal = Start->codegen(target);
    if (!StartVal) {
        return nullptr;
    }

    // Store the value into the alloca.
    builder->CreateStore(StartVal, Alloca);

    // Make the new basic block for the loop header, inserting after current
    // block.
    BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "loop", TheFunction);

    // Insert an explicit fall through from the current block to the LoopBB.
    builder->CreateBr(LoopBB);

    // Start insertion in LoopBB.
    builder->SetInsertPoint(LoopBB);

    // Within the loop, the variable is defined equal to the PHI node.  If it
    // shadows an existing variable, we have to restore it, so save it now.
    AllocaInst *OldVal = NamedValues[VarName];
    NamedValues[VarName] = Alloca;

    // Emit the body of the loop.  This, like any other expr, can change the
    // current BB.  Note that we ignore the value computed by the body, but don't
    // allow an error.
    if (!Body->codegen(target)) {
        return nullptr;
    }

    // Emit the step value.
    Value *StepVal = nullptr;
    if (Step) {
        StepVal = Step->codegen(target);
        if (!StepVal) {
            return nullptr;
        }
    } else {
        // If not specified, use 1.0.
        StepVal = ConstantFP::get(llvmContext, APFloat(1.0));
    }

    // Compute the end condition.
    Value *EndVar = End->codegen(target);
    if (!EndVar) {
        return nullptr;
    }

    // Reload, increment, and restore the alloca.  This handles the case where
    // the body of the loop mutates the variable.
    Value *CurVar = builder->CreateLoad(Alloca, VarName.c_str());
    Value *NextVar = builder->CreateAdd(CurVar, StepVal, "nextvar");

    builder->CreateStore(NextVar, Alloca);

    // Convert condition to a bool by comparing non-equal to 0.0.
    Value *EndCond = builder->CreateICmpSLT(NextVar, EndVar, "loopcond");

    // Create the "after loop" block and insert it.
    BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "afterloop", TheFunction);

    // Insert the conditional branch into the end of LoopEndBB.
    builder->CreateCondBr(EndCond, LoopBB, AfterBB);

    // Any new code will be inserted in AfterBB.
    builder->SetInsertPoint(AfterBB);

    // Restore the unshadowed variable.
    if (OldVal) {
        NamedValues[VarName] = OldVal;
    } else {
        NamedValues.erase(VarName);
    }

    // for expr always returns 0.0.
    return Constant::getNullValue(Type::getFloatTy(llvmContext));
}
Value *ListExprAST::codegen(LLVMTarget* target) {
    for (auto& expr : exprs) {
        expr->codegen(target);
    }
    // list exprs always returns 0.0.
    return Constant::getNullValue(Type::getFloatTy(target->getContext()));
}

Value *VarExprAST::codegen(LLVMTarget* target) {
    std::vector<AllocaInst *> OldBindings;

    Function *TheFunction = target->getBuilder()->GetInsertBlock()->getParent();

    // Register all variables and emit their initializer.
    for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
        const std::string &VarName = VarNames[i].first;
        ExprAST *Init = VarNames[i].second.get();

        // Emit the initializer before adding the variable to scope, this prevents
        // the initializer from referencing the variable itself, and permits stuff
        // like this:
        //  var a = 1 in
        //    var a = a in ...   # refers to outer 'a'.
        Value *InitVal;
        if (Init) {
            InitVal = Init->codegen(target);
            if (!InitVal) {
                return nullptr;
            }
        } else { // If not specified, use 0.0.
            InitVal = ConstantFP::get(target->getContext(), APFloat(0.0));
        }

        AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName, Type::getFloatTy(target->getContext()));
        target->getBuilder()->CreateStore(InitVal, Alloca);

        // Remember the old variable binding so that we can restore the binding when
        // we unrecurse.
        OldBindings.push_back(NamedValues[VarName]);

        // Remember this binding.
        NamedValues[VarName] = Alloca;
    }

    // Codegen the body, now that all vars are in scope.
    Value *BodyVal = Body->codegen(target);
    if (!BodyVal) {
        return nullptr;
    }

    // Pop all our variables from scope.
    for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
        NamedValues[VarNames[i].first] = OldBindings[i];
    }

    // Return the body computation.
    return BodyVal;
}

Function *PrototypeAST::codegen(LLVMTarget* target) {
    // Make the function type:  double(double,double) etc.
    std::vector<std::string> Args {"inputs", "outputs"};
    std::vector<Type*> Types {PointerType::getUnqual(Type::getFloatPtrTy(target->getContext())), PointerType::getUnqual(Type::getFloatPtrTy(target->getContext()))};
    FunctionType *FT = FunctionType::get(Type::getVoidTy(target->getContext()), Types, false);

    Function *F = Function::Create(FT, Function::ExternalLinkage, Name, target->getModule());
    // Set names for all arguments.
    unsigned Idx = 0;
    for (auto &Arg : F->args()) {
        F->addParamAttr(Idx, Attribute::NoAlias);
        Arg.setName(Args[Idx++]);
    }

    return F;
}

Function *FunctionAST::codegen(LLVMTarget* target) {
    // Transfer ownership of the prototype to the FunctionProtos map, but keep a
    // reference to it for use below.
    auto &P = *Proto;
    FunctionProtos[Proto->getName()] = std::move(Proto);
    Function *TheFunction = getFunction(target, P.getName());
    if (!TheFunction) {
        return nullptr;
    }

    // Create a new basic block to start insertion into.
    BasicBlock *BB = BasicBlock::Create(target->getContext(), "entry", TheFunction);
    target->getBuilder()->SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map.
    NamedValues.clear();

    for (auto &Arg : TheFunction->args()) {
        // Create an alloca for this variable.
        AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName(), Arg.getType());

        // Store the initial value into the alloca.
        target->getBuilder()->CreateStore(&Arg, Alloca);

        // Add arguments to variable symbol table.
        NamedValues[std::string(Arg.getName())] = Alloca;
    }

    if (Value *RetVal = Body->codegen(target)) {
        // ret
        target->getBuilder()->CreateRetVoid();
        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        return TheFunction;
    }

    // Error reading body, remove function.
    TheFunction->eraseFromParent();

    return nullptr;
}
