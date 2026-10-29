// Copyright (c) Alibaba Group Holding Limited. All rights reserved.
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <cmath>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/SharedAttentionModule.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace {
std::vector<int8_t> attentionGraph() {
    auto q = _Input({1, 1, 1, 8}, NCHW);
    auto k = _Input({1, 1, 1, 8}, NCHW);
    auto v = _Input({1, 1, 1, 8}, NCHW);
    auto mask = _Input({}, NCHW);
    q->setName("Q"); k->setName("K"); v->setName("V"); mask->setName("mask");
    OpT op;
    op.type = OpType_Attention;
    op.name = "attention";
    op.main.type = OpParameter_AttentionParam;
    op.main.value = new AttentionParamT;
    op.main.AsAttentionParam()->kv_cache = true;
    auto hidden = Variable::create(Expr::create(&op, {q, k, v, mask}));
    hidden->setName("hidden_states");
    auto output = _Add(hidden, _Scalar<float>(0.25f));
    output->setName("logits");
    return Variable::save({output});
}
std::shared_ptr<Executor::RuntimeManager> makeRuntime(KVMeta* meta) {
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    auto runtime = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(config));
    runtime->setHintPtr(Interpreter::KVCACHE_INFO, meta);
    runtime->setHint(Interpreter::ATTENTION_OPTION, 0);
    return runtime;
}
std::vector<float> runAttention(Module* module, KVMeta& meta, int length) {
    auto q = _Input({1, length, 1, 8}, NCHW);
    auto k = _Input({1, length, 1, 8}, NCHW);
    auto v = _Input({1, length, 1, 8}, NCHW);
    auto mask = _Input({}, NCHW);
    mask->writeMap<float>()[0] = 0;
    for (int i = 0; i < length * 8; ++i) {
        q->writeMap<float>()[i] = std::sin(float(i) * 0.19f);
        k->writeMap<float>()[i] = std::cos(float(i) * 0.17f);
        v->writeMap<float>()[i] = float((i * 37) % 101) / 13.0f;
    }
    meta.add = length;
    auto outputs = module->onForward({q, k, v, mask});
    if (outputs.size() != 1 || !outputs[0]->getInfo() || outputs[0]->getInfo()->size != length * 8) {
        return {};
    }
    const auto data = outputs[0]->readMap<float>();
    if (!data) {
        return {};
    }
    std::vector<float> result(data, data + length * 8);
    meta.sync();
    return result;
}
}

class SharedAttentionModuleTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MNNTEST_CPU_ONLY();
        auto executor = Executor::newExecutor(MNN_FORWARD_CPU, BackendConfig(), 1);
        std::weak_ptr<Executor> weakExecutor = executor;
        std::shared_ptr<Module> derived;
        std::weak_ptr<Module> weakBase;
        std::weak_ptr<KVMeta> weakMeta;
        std::weak_ptr<Executor::RuntimeManager> weakRuntime;
        KVMeta* retainedMeta = nullptr;
        auto graph = attentionGraph();
        auto bytes = reinterpret_cast<const uint8_t*>(graph.data());
        const std::vector<std::string> inputs = {"Q", "K", "V", "mask"};
        Module::Config config;
        config.rearrange = true;
        {
            ExecutorScope scope(executor);
            auto meta = std::make_shared<KVMeta>();
            meta->layer_nums = 1;
            auto runtime = makeRuntime(meta.get());
            auto base = std::shared_ptr<Module>(
                Module::load(inputs, {"logits"}, bytes, graph.size(), runtime, &config));
            MNNTEST_ASSERT(base != nullptr);
            config.base = base.get();
            SharedAttentionOwners owners{base, runtime, meta, executor};
            auto load = [&](const SharedAttentionOwners& value) {
                return std::shared_ptr<Module>(loadSharedAttentionModule(
                    inputs, {"hidden_states"}, bytes, graph.size(), value, &config));
            };
            for (int field = 0; field < 4; ++field) {
                auto missing = owners;
                if (field == 0) missing.base.reset();
                if (field == 1) missing.runtime.reset();
                if (field == 2) missing.meta.reset();
                if (field == 3) missing.executor.reset();
                MNNTEST_ASSERT(!load(missing));
            }
            auto mismatched = owners;
            mismatched.meta = std::make_shared<KVMeta>();
            MNNTEST_ASSERT(!load(mismatched));
            derived = load(owners);
            MNNTEST_ASSERT(derived && !Module::clone(derived.get()));
            retainedMeta = meta.get();
            weakBase = base; weakMeta = meta; weakRuntime = runtime;
        }
        executor.reset();
        MNNTEST_ASSERT(!weakBase.expired() && !weakMeta.expired() && !weakRuntime.expired());
        MNNTEST_ASSERT(!weakExecutor.expired());
        {
            ExecutorScope scope(weakExecutor.lock());
            KVMeta referenceMeta;
            referenceMeta.layer_nums = 1;
            auto runtime = makeRuntime(&referenceMeta);
            config.base = nullptr;
            auto reference = std::shared_ptr<Module>(
                Module::load(inputs, {"logits"}, bytes, graph.size(), runtime, &config));
            MNNTEST_ASSERT(reference != nullptr);
            for (int length : {3, 1, 80, 2}) {
                if (length == 2) {
                    retainedMeta->remove = retainedMeta->previous;
                    referenceMeta.remove = referenceMeta.previous;
                }
                auto sharedBase = weakBase.lock();
                auto actual = runAttention(length == 1 ? sharedBase.get() : derived.get(), *retainedMeta, length);
                auto expected = runAttention(reference.get(), referenceMeta, length);
                MNNTEST_ASSERT(actual.size() == size_t(length * 8) && actual.size() == expected.size());
                for (size_t i = 0; i < actual.size(); ++i) {
                    MNNTEST_ASSERT(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                    MNNTEST_ASSERT(std::fabs(actual[i] - (expected[i] - (length == 1 ? 0.0f : 0.25f))) <= 1e-5f);
                }
                MNNTEST_ASSERT(retainedMeta->previous == referenceMeta.previous);
            }
        }
        derived.reset();
        MNNTEST_ASSERT(weakBase.expired() && weakMeta.expired() && weakRuntime.expired() && weakExecutor.expired());
        return true;
    }
};
MNNTestSuiteRegister(SharedAttentionModuleTest, "expr/SharedAttentionModule");
#endif
