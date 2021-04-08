//
//  LLVMJit.cpp
//  MNN
//
//  Created by MNN on 2021/2/2.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "jit/LLVMJit.hpp"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"

#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
class MCJITObjectCache : public ObjectCache {
public:
    MCJITObjectCache() {
        sys::fs::current_path(CacheDir);
        sys::path::append(CacheDir, "mnn_object_cache");
    }

    virtual ~MCJITObjectCache() {}

    bool isCached(std::string moduleId) {
        SmallString<128> IRCacheFile = CacheDir;
        sys::path::append(IRCacheFile, moduleId);
        return sys::fs::exists(IRCacheFile.str());
    }

    virtual void notifyObjectCompiled(const Module *M, MemoryBufferRef Obj) {
        const std::string ModuleID = M->getModuleIdentifier();

        if (0 == ModuleID.compare(0, 4, "jit-")) {
            std::string IRFileName = ModuleID;
            SmallString<128>IRCacheFile = CacheDir;
            sys::path::append(IRCacheFile, IRFileName);
            if (!sys::fs::exists(CacheDir.str()) && sys::fs::create_directory(CacheDir.str())) {
                fprintf(stderr, "Unable to create cache directory\n");
                return;
            }
            std::error_code ec;
            raw_fd_ostream IRObjectFile(IRCacheFile.c_str(), ec, sys::fs::F_None);
            IRObjectFile << Obj.getBuffer();
        }
    }

    virtual std::unique_ptr<MemoryBuffer> getObject(const Module* M) {
        if (!isCached(M->getModuleIdentifier())) {
            return nullptr;
        }
        SmallString<128> IRCacheFile = CacheDir;
        sys::path::append(IRCacheFile, M->getModuleIdentifier());
        ErrorOr<std::unique_ptr<MemoryBuffer>> IRObjectBuffer = MemoryBuffer::getFile(IRCacheFile.c_str(), -1, false);
        if (!IRObjectBuffer) {
            return nullptr;
        }
        return MemoryBuffer::getMemBufferCopy(IRObjectBuffer.get()->getBuffer());
  }

private:
    SmallString<128> CacheDir;
};

static MCJITObjectCache cacheObj;
LLVMJIT* LLVMJIT::llvmJit = nullptr;

LLVMJIT::LLVMJIT(std::unique_ptr<TargetProcessControl> tpc, std::unique_ptr<ExecutionSession> es, JITTargetMachineBuilder jtmb, DataLayout dl)
      : processControl(std::move(tpc)), executionSession(std::move(es)), dataLayout(std::move(dl)),
        mangle(*this->executionSession, this->dataLayout),
        objectLayer(*this->executionSession, []() { return std::make_unique<SectionMemoryManager>(); }),
        compileLayer(*this->executionSession, objectLayer, std::make_unique<ConcurrentIRCompiler>(std::move(jtmb))),
        optimizeLayer(*this->executionSession, compileLayer, optimizeModule),
        mainJD(this->executionSession->createBareJITDylib("<main>")) {
    mainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(dl.getGlobalPrefix())));
}

LLVMJIT::~LLVMJIT() {
    if (auto Err = executionSession->endSession()) {
        executionSession->reportError(std::move(Err));
    }
}

void LLVMJIT::addModule(ThreadSafeModule tsm, ResourceTrackerSP rt) {
    if (!rt) {
        rt = mainJD.getDefaultResourceTracker();
    }
    ExitOnErr(optimizeLayer.add(rt, std::move(tsm)));
}

Expected<JITEvaluatedSymbol> LLVMJIT::lookup(StringRef Name) {
    return executionSession->lookup({&mainJD}, mangle(Name.str()));
}

void LLVMJIT::compileAllFunction(int num) {
    auto comp = static_cast<ConcurrentIRCompiler*>(&compileLayer.getCompiler());
    comp->setObjectCache(&cacheObj);
    functions.resize(num);
    for (int i = 0; i < num; i++) {
        functions[i] = getFuncByName("kernel_" + std::to_string(i));
    }
}

uint64_t LLVMJIT::getFuncByName(std::string name) {
    return ExitOnErr(lookup(name)).getAddress();
}

uint64_t LLVMJIT::getFuncByIdx(int idx) {
    if (functions.size() <= idx) {
        return 0;
    }
    return functions[idx];
}

LLVMJIT* LLVMJIT::createLLVMJIT() {
    if (llvmJit != nullptr) {
        return llvmJit;
    }
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
    auto tpc = SelfTargetProcessControl::Create();
    if (!tpc) {
        return nullptr;
    }
    auto es = std::make_unique<ExecutionSession>();
    JITTargetMachineBuilder jtmb((*tpc)->getTargetTriple());
    auto dl = jtmb.getDefaultDataLayoutForTarget();
    if (!dl) {
        return nullptr;
    }
    llvmJit = new LLVMJIT(std::move(*tpc), std::move(es), std::move(jtmb), std::move(*dl));
    return llvmJit;
}

TargetMachine* LLVMJIT::GetTargetMachine(Triple TheTriple) {
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
    if (!TheTarget) {
        return nullptr;
    }
    return TheTarget->createTargetMachine(TheTriple.getTriple(), codegen::getCPUStr(), codegen::getFeaturesStr(), codegen::InitTargetOptionsFromCodeGenFlags(TheTriple),
                                          codegen::getExplicitRelocModel(), codegen::getExplicitCodeModel(), CodeGenOpt::Aggressive);
}

Expected<ThreadSafeModule> LLVMJIT::optimizeModule(ThreadSafeModule tsm, const MaterializationResponsibility &mr) {
    static codegen::RegisterCodeGenFlags CFG;
    tsm.withModuleDo([](Module &m) {
        if (cacheObj.isCached(m.getModuleIdentifier())) {
            return;
        }
        auto modulePassManager = std::make_unique<legacy::PassManager>();
        auto funcPassManager = std::make_unique<legacy::FunctionPassManager>(&m);
        {
            Triple moduleTriple(m.getTargetTriple());
            TargetMachine *Machine = nullptr;
            if (moduleTriple.getArch()) {
                Machine = GetTargetMachine(moduleTriple);
            }
            modulePassManager->add(createTargetTransformInfoWrapperPass(Machine->getTargetIRAnalysis()));
            funcPassManager->add(createTargetTransformInfoWrapperPass(Machine->getTargetIRAnalysis()));
            PassManagerBuilder builder;
            builder.OptLevel = 3;
            builder.SizeLevel = 0;
            // builder.Inliner = createFunctionInliningPass(3, 0, false);
            builder.DisableUnrollLoops = false;
            builder.LoopVectorize = true;
            builder.SLPVectorize = true;
            builder.populateFunctionPassManager(*funcPassManager.get());
            builder.populateModulePassManager(*modulePassManager.get());
            funcPassManager->doInitialization();
            for (auto &function : m) {
                funcPassManager->run(function);
            }
            funcPassManager->doFinalization();
            modulePassManager->run(m);
        }
    });
    return std::move(tsm);
}
