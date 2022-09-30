//
//  Expr.hpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Expr_hpp
#define MNN_Expr_hpp

#include <functional>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <MNN/HalideRuntime.h>
#include <MNN/MNNDefine.h>

namespace MNN {
struct BufferStorage;
struct OpT;
struct Op;
struct NetT;
class Tensor;
namespace Express {
class Variable;
class Expr;
class Executor;
typedef std::shared_ptr<Expr> EXPRP;
typedef std::weak_ptr<Expr> WeakEXPRP;
typedef std::vector<int> INTS;
enum Dimensionformat { NHWC, NC4HW4, NCHW };
class MNN_PUBLIC VARP {
public:
    VARP() {
        // Do nothing
    }
    VARP(std::shared_ptr<Variable> c) {
        mContent = std::move(c);
    }
    VARP(Variable* c) {
        mContent.reset(c);
    }
    Variable* get() const  {
        return mContent.get();
    }
    ~ VARP() {
        // Do nothing
    }
    VARP(const VARP& var) {
        mContent = var.mContent;
    }
    VARP(VARP&& var) {
        mContent = std::move(var.mContent);
    }
    VARP operator+(VARP var) const;
    VARP operator-(VARP var) const;
    VARP operator*(VARP var) const;
    VARP operator/(VARP var) const;
    VARP mean(INTS dims) const;
    VARP sum(INTS dims) const;

    bool operator==(const VARP& var) const {
        return var.mContent == mContent;
    }
    bool operator<(const VARP& var) const {
        return mContent < var.mContent;
    }
    bool operator<=(const VARP& var) const {
        return mContent <= var.mContent;
    }
    VARP& operator=(const VARP& var) {
        mContent = var.mContent;
        return *this;
    }
    VARP& operator=(Variable* var) {
        mContent.reset(var);
        return *this;
    }
    Variable* operator->() const  {
        return mContent.get();
    }
    enum InputType {
        INPUT = 0,
        CONSTANT = 1,
        TRAINABLE = 2,
    };
    bool fix(InputType type) const;
private:
    friend class Variable;
    std::shared_ptr<Variable> mContent;
};
inline bool operator==(Variable* src, VARP dst) {
    return src == dst.get();
}
inline bool operator!=(Variable* src, VARP dst) {
    return src != dst.get();
}
// inline bool operator<(VARP src, VARP dst) {
//     return src.get() < dst.get();
// }
typedef std::vector<VARP> VARPS;

class MNN_PUBLIC Variable {
public:
    struct Info {
        Dimensionformat order = NHWC;
        INTS dim;
        halide_type_t type;
        int size;
        void syncSize();
    };
    const std::string& name() const;
    void setName(const std::string& name);
    std::pair<EXPRP, int> expr() const {
        return std::make_pair(mFrom, mFromIndex);
    }
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

    static std::vector<VARP> load(const char* fileName);
    static std::map<std::string, VARP> loadMap(const char* fileName);
    static std::vector<VARP> load(const uint8_t* buffer, size_t length);
    static std::map<std::string, VARP> loadMap(const uint8_t* buffer, size_t length);
    static std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> getInputAndOutput(const std::map<std::string, VARP>& allVariable);
    static std::vector<VARP> mapToSequence(const std::map<std::string, VARP>& source);
    static std::vector<EXPRP> getExecuteOrder(const std::vector<VARP>& output);
    static void save(const std::vector<VARP>& vars, const char* fileName);
    static std::vector<int8_t> save(const std::vector<VARP>& vars);
    static void save(const std::vector<VARP>& vars, NetT* dest);

    // Pack a few Variable to compute in one pipeline
    static void prepareCompute(const std::vector<VARP>& vars, bool forceCPU = false);
    static void compute(const std::vector<VARP>& vars, bool forceCPU = false);

    size_t linkNumber() const;
    const std::vector<WeakEXPRP>& toExprs() const;
    void setExpr(EXPRP expr, int index) {
        mFrom = expr;
        mFromIndex = index;
    }

private:
    Variable(EXPRP expr, int index) {
        mFrom      = expr;
        mFromIndex = index;
    }

    void* readInternal(bool forShape = false);
    void* writeInternal(bool inform=true);
    void informDirty();

    friend class Expr;
    EXPRP mFrom;
    int mFromIndex;
};
class MNN_PUBLIC Expr {
public:
    struct Inside;
    enum MemoryType {
        COPY,
        MOVE,
        REF
    };
    static EXPRP create(Tensor* tensor, bool own = false);

    static EXPRP create(Variable::Info&& info, const void* ptr, VARP::InputType type, MemoryType copy = COPY);
    static EXPRP create(const OpT* op, std::vector<VARP> inputs, int outputSize = 1);
    static EXPRP create(std::shared_ptr<BufferStorage> extra, std::vector<VARP>&& inputs, int outputSize = 1);
    static EXPRP create(std::unique_ptr<OpT>&& op, std::vector<VARP> inputs, int outputSize = 1) {
        return create(op.get(), inputs, outputSize);
    }
    void setName(const std::string& name);

    const Op* get() const {
        return mOp;
    }
    const std::vector<VARP>& inputs() const {
        return mInputs;
    }
    int outputSize() const {
        return (int)mOutputNames.size();
    }
    static void replace(EXPRP oldExpr, EXPRP newExpr);
    bool requireInfo();
    void visitOutputs(const std::function<bool(EXPRP, int)>& visit);
    static void visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after);

    const std::vector<WeakEXPRP>& outputs() const {
        return mTo;
    }
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
    const std::string& outputName(int index) {
        return mOutputNames[index];
    }

    VARP::InputType inputType() const {return mType;}
    Variable::Info* outputInfo(int index) const;
    std::shared_ptr<BufferStorage> extra() const {
        return mStorage;
    }
    bool setInfoDirty();
    std::shared_ptr<Inside> inside() const {
        return mInside;
    }
    bool valid() const {
        return mValid;
    }

private:
    static void _addLinkForInputs(EXPRP expr);

    Expr(int outputSize);
    Expr(Tensor* tensor, bool own = false);

    friend class Variable;
    friend class VARP;
    VARP::InputType mType;
    const Op* mOp;
    std::vector<VARP> mInputs;
    std::vector<std::string> mOutputNames;

    bool mValid = true;
    std::shared_ptr<BufferStorage> mStorage;
    std::string mName;
    std::shared_ptr<Inside> mInside = nullptr;
    bool mVisited                   = false;
    std::vector<WeakEXPRP> mTo;

};
} // namespace Express
} // namespace MNN

#endif /* Expr_hpp */
