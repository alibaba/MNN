//
//  LLVMJit.hpp
//  MNN
//
//  Created by MNN on 2021/2/2.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "llvm/IR/DataLayout.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::orc;

class LLVMJIT {
public:
    LLVMJIT(std::unique_ptr<TargetProcessControl> tpc, std::unique_ptr<ExecutionSession> es, JITTargetMachineBuilder jtmb, DataLayout dl);

    ~LLVMJIT();

    static LLVMJIT* createLLVMJIT();

    const DataLayout &getDataLayout() const { return dataLayout; }

    JITDylib &getMainJITDylib() { return mainJD; }

    void addModule(ThreadSafeModule tsm, ResourceTrackerSP rt = nullptr);

    Expected<JITEvaluatedSymbol> lookup(StringRef Name);

    void compileAllFunction(int num);

    uint64_t getFuncByName(std::string name);

    uint64_t getFuncByIdx(int idx);
private:
    static TargetMachine* GetTargetMachine(Triple TheTriple);
    static Expected<ThreadSafeModule> optimizeModule(ThreadSafeModule tsm, const MaterializationResponsibility &mr);
private:
    std::unique_ptr<TargetProcessControl> processControl;
    std::unique_ptr<ExecutionSession> executionSession;
    std::vector<uint64_t> functions;
    RTDyldObjectLinkingLayer objectLayer;
    IRCompileLayer compileLayer;
    IRTransformLayer optimizeLayer;
    DataLayout dataLayout;
    MangleAndInterner mangle;
    JITDylib &mainJD;
    ExitOnError ExitOnErr;
    Triple targetTriple;
    static LLVMJIT* llvmJit;
};
