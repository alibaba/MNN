//
//  Expr.hpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Expr_hpp
#define Expr_hpp

#include <functional>
#include <list>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "HalideRuntime.h"
#include "MNNDefine.h"

#if defined(_MSC_VER)
#if defined(BUILDING_MNN_EXPRESS_DLL)
#define MNN_EXPRESS_PUBLIC __declspec(dllexport)
#elif defined(USING_MNN_EXPRESS_DLL)
#define MNN_EXPRESS_PUBLIC __declspec(dllimport)
#else
#define MNN_EXPRESS_PUBLIC
#endif
#else
#define MNN_EXPRESS_PUBLIC __attribute__((visibility("default")))
#endif

namespace MNN {
struct OpT;
struct Op;
struct NetT;
namespace Express {
class Solution;
class Variable;
class Expr;
class Executor;
typedef std::shared_ptr<Expr> EXPRP;
typedef std::weak_ptr<Expr> WeakEXPRP;
typedef std::shared_ptr<Variable> VARP;
typedef std::weak_ptr<Variable> WeakVARP;
typedef std::vector<int> INTS;
typedef std::vector<VARP> VARPS;
enum Dimensionformat { NHWC, NC4HW4, NCHW };
class MNN_EXPRESS_PUBLIC Variable {
public:
    struct Info {
        Dimensionformat order = NHWC;
        INTS dim;
        halide_type_t type;
        int size;
        void* ptr = nullptr;
    };
    const std::string& name() const {
        return mName;
    }
    void setName(const std::string& name);
    std::pair<EXPRP, int> expr() const {
        return std::make_pair(mFrom, mFromIndex);
    }
    static void setExpr(VARP dst, EXPRP from, int index);

    // If compute info error, return nullptr
    const Info* getInfo();
    bool resize(INTS dims);
    template <typename T>
    const T* readMap() {
        return (const T*)readInternal();
    }

    template <typename T>
    T* writeMap() {
        return (T*)writeInternal();
    }
    
    //Depecerate
    void unMap();
    
    bool input(VARP src);
    static void replace(VARP dst, VARP src);

    static VARP create(EXPRP expr, int index = 0);

    void visitOutputs(const std::function<bool(VARP, int)>& visit);

    static void visit(VARP var, const std::function<bool(VARP)>& before, const std::function<bool(VARP)>& after);

    static std::vector<VARP> load(const char* fileName);
    static std::map<std::string, VARP> loadMap(const char* fileName);
    static std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> getInputAndOutput(const std::map<std::string, VARP>& allVariable);
    static std::vector<VARP> mapToSequence(const std::map<std::string, VARP>& source);
    static std::vector<VARP> getExecuteOrder(const std::vector<VARP>& output);
    static void save(const std::vector<VARP>& vars, const char* fileName);
    static void save(const std::vector<VARP>& vars, NetT* dest);

    size_t linkNumber() const {
        return mTo.size();
    }
private:
    Variable(EXPRP expr, int index) {
        mFrom      = expr;
        mFromIndex = index;
    }

    void* readInternal();
    void* writeInternal(bool inform=true);
    void informDirty();

    friend class Expr;
    EXPRP mFrom;
    int mFromIndex;
    std::string mName;
    std::list<std::pair<int, WeakEXPRP>> mTo;
};

class MNN_EXPRESS_PUBLIC Expr {
public:
    struct Inside;
    static EXPRP create(const OpT* op, std::vector<VARP> inputs, int outputSize = 1,
                        std::shared_ptr<Executor> executor = nullptr);
    static EXPRP create(std::unique_ptr<OpT>&& op, std::vector<VARP> inputs, int outputSize = 1,
                        std::shared_ptr<Executor> executor = nullptr) {
        return create(op.get(), inputs, outputSize, executor);
    }
    void setName(const std::string& name);
    void setExecutor(std::shared_ptr<Executor> exe);

    const Op* get() const {
        return mOp;
    }
    void set(const OpT* op);
    const std::vector<VARP>& inputs() const {
        return mInputs;
    }
    const Variable::Info* outputInfo(int index) const;
    int outputSize() const {
        return mOutputSize;
    }
    bool requireInfo();
    bool requireAlloc();
    bool requireCompute();

    Solution* inside();

    const std::list<WeakVARP>& outputs() const {
        return mOutputs;
    }
    static void setInput(EXPRP dst, VARP src, int index);
    ~Expr();

    bool visited() const {
        return mVisited;
    }
    void setVisited(bool visited) {
        mVisited = visited;
    }
    const std::string& name() const {
        return mName;
    }

private:
    bool setContentDirty(int inputIndex);
    bool setInfoDirty();

    Expr(int outputSize);

    friend class Variable;
    const Op* mOp;
    std::vector<VARP> mInputs;
    std::list<WeakVARP> mOutputs;
    const int mOutputSize;

    bool mValid = true;
    bool mInfoDirty    = true;
    bool mAllocated    = false;
    bool mContentDirty = true;
    char* mExtraBuffer = nullptr;
    std::string mName;
    std::shared_ptr<Inside> mInside = nullptr;
    bool mVisited                   = false;
    std::shared_ptr<Executor> mExecutor;
};
} // namespace Express
} // namespace MNN

#endif /* Expr_hpp */
