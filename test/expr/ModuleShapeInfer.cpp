#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN;
using namespace MNN::Express;

class ModuleShapeInfer : public MNNTestCase {
public:
    static float _reduceSum(const float* zPtr, int size) {
        float summer = 0.0f;
        for (int i=0; i<size; ++i) {
            summer+=zPtr[i];
        }
        return summer;
    }
    virtual bool run(int precision) {
        auto executor = cloneCurrentExecutor();
        ExecutorScope scope(executor);
        std::vector<VARP> empty;
        // Make Net
        auto x = _Input({1, 3, 2, 2}, NCHW, halide_type_of<float>());
        x->setName("x");
        auto y = x * x;
        VARP starts;
        VARP sizes;
        {
            std::vector<int> sta = {0, 0, 1, 1};
            std::vector<int> siz = {1, 1, 1, 1};
            starts = _Const(sta.data(), {4}, NCHW, halide_type_of<int>());
            sizes = _Const(siz.data(), {4}, NCHW, halide_type_of<int>());
        }
        auto z = _Slice(y, starts, sizes);
        z->setName("z");
        auto buffer = Variable::save({z});
        ScheduleConfig config;
        BackendConfig bnConfig;
        bnConfig.precision = MNN::BackendConfig::Precision_Low;
        config.backendConfig = &bnConfig;
        std::shared_ptr<Executor::RuntimeManager> rt(Executor::RuntimeManager::createRuntimeManager(config), Executor::RuntimeManager::destroy);
        std::shared_ptr<Module> net0(Module::load({"x"}, {"z"}, (const uint8_t*)buffer.data(), buffer.size(), rt), Module::destroy);
        std::shared_ptr<Module> net1(Module::load({"x"}, {"z"}, (const uint8_t*)buffer.data(), buffer.size(), rt), Module::destroy);
        x = _Input({1, 3, 2, 2}, NCHW, halide_type_of<float>());
        // Run Init Value
        auto inputPtr = x->writeMap<float>();
        for (int i=0; i<x->getInfo()->size; ++i) {
            inputPtr[i] = i;
        }
        y = net0->onForward({x})[0];
        auto yPtr = y->readMap<float>();
        auto ySize = y->getInfo()->size;
        auto valueFirst = _reduceSum(yPtr, ySize);
        for (int i=0; i<x->getInfo()->size; ++i) {
            inputPtr[i] = x->getInfo()->size - i;
        }
        y = net0->onForward({x})[0];
        yPtr = y->readMap<float>();
        auto valueSecond = _reduceSum(yPtr, ySize);
        
        // Shape Infer mode
        auto code = net1->traceOrOptimize(Interpreter::Module_Forward_Separate);
        if (0 != code) {
            FUNC_PRINT(1);
            return false;
        }
        for (int i=0; i<x->getInfo()->size; ++i) {
            inputPtr[i] = i;
        }
        y = net1->onForward({x})[0];
        yPtr = y->readMap<float>();
        auto tmp = net1->onForward(empty);
        if (tmp.size() > 0) {
            FUNC_PRINT(1);
            return false;
        }
        if (_reduceSum(yPtr, ySize) != valueFirst) {
            FUNC_PRINT(1);
            return false;
        }
        for (int i=0; i<x->getInfo()->size; ++i) {
            inputPtr[i] = x->getInfo()->size - i;
        }
        net1->onForward(empty);
        if (_reduceSum(yPtr, ySize) != valueSecond) {
            FUNC_PRINT(1);
            return false;
        }
        net1->traceOrOptimize(MNN::Interpreter::Module_Forward_Combine);
        for (int i=0; i<x->getInfo()->size; ++i) {
            inputPtr[i] = i;
        }
        y = net1->onForward({x})[0];
        yPtr = y->readMap<float>();
        if(_reduceSum(yPtr, ySize) != valueFirst) {
            FUNC_PRINT(1);
            return false;
        }
        for (int i=0; i<x->getInfo()->size; ++i) {
            inputPtr[i] = x->getInfo()->size - i;
        }
        y = net1->onForward({x})[0];
        yPtr = y->readMap<float>();
        if(_reduceSum(yPtr, ySize) != valueSecond) {
            FUNC_PRINT(1);
            return false;
        }
        return true;
    }
};

class VariableSaveLoad: public MNNTestCase { // Verify the order of load is the same as the order of save
public:
    virtual bool run(int precision) {
        std::vector<MNN::Express::VARP> vars;
        std::vector<int32_t> contents(4);
        std::string file = "file.txt";
        contents[0] = 0;
        contents[1] = 1;
        contents[2] = 2;
        contents[3] = 3;

        for (auto number: contents) {
            auto var = MNN::Express::_Const(&number, {1}, MNN::Express::NHWC, halide_type_of<int32_t>());
            if (var->getInfo() == nullptr) {
                MNN_PRINT("error\n");
                return false;
            }
            vars.emplace_back(var);
        }

        MNN::Express::Variable::save(vars, file.c_str());

        auto readVars = MNN::Express::Variable::load(file.c_str());
        std::vector<int32_t> readContents;
        for (auto var_: readVars) {
            auto var_ptr = var_->getInfo();
            if (var_ptr == nullptr) {
                MNN_PRINT("error\n");
                return false;
            }
            readContents.push_back(var_->readMap<int32_t>()[0]);
        }
        for (int i = 0; i < 4; ++i) {
            if (readContents[i] != contents[i]) {
                MNN_PRINT("error %d: read %d, expect %d\n", i, readContents[i], contents[i]);
                return false;
            }
        }

        int result = std::remove(file.c_str());
        if (result == 0) {
            MNN_PRINT("delete file success\n");
            return true;
        } else {
            MNN_PRINT("delete file failed\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ModuleShapeInfer, "expr/ModuleShapeInfer");
MNNTestSuiteRegister(VariableSaveLoad, "variable/saveLoad");
