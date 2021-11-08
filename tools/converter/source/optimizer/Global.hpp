//
//  Global.hpp
//  MNNConverter
//
//  Created by MNN on 2020/06/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_OPTIMIZER_GLOBAL_HPP_
#define MNN_CONVERTER_OPTIMIZER_GLOBAL_HPP_

#include <mutex>
#include <unordered_map>

template <typename T>
class Global {
public:
    static void Reset(T* pvalue) {
        auto* instance = Instance();
        std::unique_lock<std::mutex> lock(instance->mutex_);
        *(instance->value_) = pvalue;
    }

    static T* Get() {
        auto* instance = Instance();
        std::unique_lock<std::mutex> lock(instance->mutex_);
        return *(instance->value_);
    }

private:
    Global() {
        value_.reset(new T*);
        *value_ = nullptr;
    }

    static Global<T>* Instance() {
        static auto* g_instance = new Global<T>();
        return g_instance;
    }

private:
    std::mutex mutex_;
    std::unique_ptr<T*> value_ = nullptr;
};

#endif  // MNN_CONVERTER_OPTIMIZER_GLOBAL_HPP_
