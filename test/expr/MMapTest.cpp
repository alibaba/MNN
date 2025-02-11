//
//  MMapTest.cpp
//  MNNTests
//
//  Created by MNN on 2025/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "core/MNNFileUtils.h"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "MNN_generated.h"

using namespace MNN::Express;

class MMapTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        if (MNN_FORWARD_CPU != getCurrentType()) {
            MNN_ERROR("Currently don't test not cpu mmap\n");
            return true;
        }
        auto x = _Input({1, 3, 224, 224}, NC4HW4, halide_type_of<float>());
        x->setName("x");
        auto y = _Conv(1.0f, 0.01f, x, {3, 16}, {5, 5});
        y->setName("y");
        auto buffer = MNN::Express::Variable::save({y});
        ScheduleConfig config;
        config.type = getCurrentType();
        std::shared_ptr<Executor::RuntimeManager> rtm( Executor::RuntimeManager::createRuntimeManager(config));
        rtm->setExternalPath("tmp", MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
        Module::Config mconfig;
        mconfig.rearrange = true;
        std::shared_ptr<MNN::Express::Module> m(Module::load({"x"}, {"y"}, (const unsigned char*)buffer.data(), buffer.size(), rtm, &mconfig), Module::destroy);

        // forward_type, precision_type, memory_type, power_type
        auto pStatus = MNNTestSuite::get()->pStaus;
        // The same as CPUBackend
        std::string prefix = "0_0_0_0_";
        prefix[2] += pStatus.precision;
        prefix[4] += pStatus.memory;
        prefix[6] += pStatus.power;
        std::string fileName = MNNFilePathConcat("tmp", prefix + "0.static");
        if (!MNNFileExist(fileName.c_str())) {
            FUNC_PRINT(1);
            return false;
        }
        m.reset();
        if (MNNFileExist(fileName.c_str())) {
            FUNC_PRINT(1);
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(MMapTest, "expr/mmaptest");
