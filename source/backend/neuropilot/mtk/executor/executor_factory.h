#pragma once
#include "common/logging.h"
#include "common/cpp11_compat.h"
#include "executor/llm_executor.h"
#include "executor/llm_medusa_executor.h"
#include "executor/llm_ringbuffer_executor.h"
#include "executor/neuron_executor.h"
#include "executor/neuron_usdk_executor.h"
#include "executor/tflite_executor.h"

#include <type_traits>

namespace mtk {

#ifdef USE_USDK_BACKEND
using NeuronModelExecutor = NeuronUsdkExecutor;
#else
using NeuronModelExecutor = NeuronExecutor;
#endif

#ifdef DISABLE_RING_BUFFER
using LlmModelExecutor = LlmExecutor;
#else
using LlmModelExecutor = LlmRingBufferExecutor;
#endif

using LlmMedusaModelExecutor = LlmMedusaExecutor;

using TFLiteModelExecutor = TfliteExecutor;

#define GetExecutorClass(ExecType) mtk::ExecType##ModelExecutor

enum class ExecutorType {
    Neuron,
    TFLite,
    Llm,
    LlmMedusa
};

class ExecutorFactory {
public:
    explicit ExecutorFactory(const ExecutorType executorType = ExecutorType::Llm)
        : mExecutorType(executorType) {}

    ExecutorFactory& setType(const ExecutorType executorType) {
        mExecutorType = executorType;
        return *this;
    }

    // The created executor object will be cast to `std::unique_ptr<TargetType>`, which defaults to
    // `std::unique_ptr<Executor>`.
    template <typename TargetType = Executor, typename... Args>
    std::unique_ptr<TargetType> create(Args&&... args) const {
#define __DECL__(ExecType)                                                                        \
    case ExecutorType::ExecType: {                                                                \
        using ExecutorClass = ExecType##ModelExecutor;                                            \
        return createExecutorIfCompatible<TargetType, ExecutorClass>(std::forward<Args>(args)...); \
    }

        switch (mExecutorType) {
            __DECL__(Neuron)
            __DECL__(TFLite)
            __DECL__(Llm)
            __DECL__(LlmMedusa)
        }

#undef __DECL__
    }

private:
    // Helper to create executor if compatible with target type
    template <typename TargetType, typename ExecutorClass, typename... Args>
    static typename cpp11_compat::enable_if_t<cpp11_compat::is_base_of<TargetType, ExecutorClass>::value,
                                               std::unique_ptr<TargetType>>
    createExecutorIfCompatible(Args&&... args) {
        auto executor = createImpl<ExecutorClass>(std::forward<Args>(args)...);
        DCHECK(executor != nullptr)
            << "Unable to create executor with the given " << sizeof...(Args) << " arguments.";
        return executor;
    }
    
    template <typename TargetType, typename ExecutorClass, typename... Args>
    static typename cpp11_compat::enable_if_t<!cpp11_compat::is_base_of<TargetType, ExecutorClass>::value,
                                               std::unique_ptr<TargetType>>
    createExecutorIfCompatible(Args&&... args) {
        return nullptr;
    }

    // Can be constructed with the provided arguments
    template <typename ExecutorClass, typename... Args>
    static typename cpp11_compat::enable_if_t<cpp11_compat::is_constructible<ExecutorClass, Args...>::value,
                                               std::unique_ptr<ExecutorClass>>
    createImpl(Args&&... args) {
        return cpp11_compat::make_unique<ExecutorClass>(std::forward<Args>(args)...);
    }

    // Cannot be constructed with the provided arguments
    template <typename ExecutorClass, typename... Args>
    static typename cpp11_compat::enable_if_t<!cpp11_compat::is_constructible<ExecutorClass, Args...>::value,
                                               std::unique_ptr<ExecutorClass>>
    createImpl(Args&&... args) {
        return nullptr;
    }

private:
    ExecutorType mExecutorType;
};

} // namespace mtk