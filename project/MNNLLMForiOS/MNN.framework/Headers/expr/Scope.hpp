//
//  RuntimeScope.hpp
//  MNN
//
//  Created by MNN on 2020/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_EXPR_SCOPE_HPP_
#define MNN_EXPR_SCOPE_HPP_

#include <cstdio>
#include <vector>
#include <string>
#include <mutex>

#include <MNN/Interpreter.hpp>

namespace MNN {
namespace Express {

template <typename T>
class Scope {
public:
    Scope();
    virtual ~Scope() = default;

    struct ScopedContent {
        std::string scope_name;
        T content;
    };
    void EnterScope(const ScopedContent& current);
    void EnterScope(const T& current);
    void EnterScope(const std::string& scope_name, const T& current);

    void ExitScope();

    const ScopedContent& Current() const;
    const T Content() const;

    int ScopedLevel() const { return scoped_level_; }

private:
    std::string MakeScopeName(const std::string& prefix, int level) const;

    mutable std::mutex mutex_;
    int scoped_level_ = 0;
    std::vector<ScopedContent> scoped_contents_;
};

template <typename T>
Scope<T>::Scope() : scoped_level_(0) {
}

template <typename T>
void Scope<T>::EnterScope(const ScopedContent& current) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++scoped_level_;
    scoped_contents_.push_back(current);
}

template <typename T>
void Scope<T>::EnterScope(const T& current) {
    EnterScope("scope", current);
}

template <typename T>
void Scope<T>::EnterScope(const std::string& scope_name,
                          const T& current) {
    std::lock_guard<std::mutex> lock(mutex_);
    int scoped_level = ScopedLevel();
    std::string name = MakeScopeName(scope_name, scoped_level++);
    ScopedContent content{name, current};
    ++scoped_level_;
    scoped_contents_.push_back(content);
}

template <typename T>
void Scope<T>::ExitScope() {
    std::lock_guard<std::mutex> lock(mutex_);
    --scoped_level_;
    scoped_contents_.resize(scoped_level_);
}

template <typename T>
const typename Scope<T>::ScopedContent& Scope<T>::Current() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MNN_CHECK(scoped_contents_.size() > 0, "Scope level should not be 0.");
    return scoped_contents_.back();
}

template <typename T>
const T Scope<T>::Content() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (scoped_contents_.empty()) {
        return nullptr;
    }
    return scoped_contents_.back().content;
}

template <typename T>
std::string Scope<T>::MakeScopeName(const std::string& prefix,
                                    int level) const {
    char s[16];
    snprintf(s, 16, "%d", level);
    return prefix + "/" + std::string(s);
}

}  // namespace Express
}  // namespace MNN

#endif  // MNN_EXPR_SCOPE_HPP_
