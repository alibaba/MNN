#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/Interpreter.hpp>
#include "MNNTestSuite.h"

using namespace MNN;
using namespace MNN::Express;

class RuntimeLifetimeTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto type = static_cast<MNNForwardType>(MNNTestSuite::get()->pStaus.forwardType);
        BackendConfig config;
        config.precision = static_cast<BackendConfig::PrecisionMode>(precision);
        VARP output;
        std::weak_ptr<Runtime> runtime;
        {
            auto executor = Executor::newExecutor(type, config, type == MNN_FORWARD_VULKAN ? 516 : 1);
            if (!executor) {
                return false;
            }
            ExecutorScope scope(executor);
            runtime = executor->getRuntime().first.at(type);
            auto input = _Input({1, 4, 1, 1}, NCHW);
            auto values = input->writeMap<float>();
            for (int i = 0; i < 4; ++i) {
                values[i] = static_cast<float>(i + 1);
            }
            output = _Square(input);
            if (!output.fix(VARP::CONSTANT)) {
                return false;
            }
        }
        if (runtime.expired()) {
            return false;
        }
        auto values = output->readMap<float>();
        if (!values) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (values[i] != static_cast<float>((i + 1) * (i + 1))) {
                return false;
            }
        }
        output = nullptr;
        return runtime.expired();
    }
};
MNNTestSuiteRegister(RuntimeLifetimeTest, "expr/runtime_lifetime");

class ModuleOutputLifetimeTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto type = static_cast<MNNForwardType>(MNNTestSuite::get()->pStaus.forwardType);
        BackendConfig backend;
        backend.precision = static_cast<BackendConfig::PrecisionMode>(precision);
        VARP retained;
        std::weak_ptr<Runtime> weak;
        {
            auto executor = Executor::newExecutor(type, backend, 1);
            ExecutorScope scope(executor);
            auto input = _Input({1, 4}, NCHW);
            input->setName("input");
            auto output = _Square(input);
            output->setName("output");
            auto data = Variable::save({output});
            ScheduleConfig schedule;
            schedule.type = type;
            schedule.backendConfig = &backend;
            std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(schedule));
            weak = executor->getRuntime().first.at(type);
            Module::Config options;
            options.shapeMutable = true;
            std::shared_ptr<Module> module(Module::load(
                {"input"}, {"output"}, reinterpret_cast<const uint8_t*>(data.data()), data.size(), runtime, &options));
            if (!module)
                return false;
            auto values = input->writeMap<float>();
            for (int i = 0; i < 4; ++i)
                values[i] = i + 1;
            auto results = module->onForward({input});
            if (results.size() != 1 || !results[0]->readMap<float>())
                return false;
            retained = results[0];
        }
        if (weak.expired())
            return false;
        auto values = retained->readMap<float>();
        if (!values)
            return false;
        for (int i = 0; i < 4; ++i)
            if (values[i] != (i + 1) * (i + 1))
                return false;
        retained = nullptr;
        return weak.expired();
    }
};
MNNTestSuiteRegister(ModuleOutputLifetimeTest, "expr/module_output_lifetime");
