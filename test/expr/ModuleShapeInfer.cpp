#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
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
MNNTestSuiteRegister(ModuleShapeInfer, "expr/ModuleShapeInfer");
