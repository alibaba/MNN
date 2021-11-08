//
//  DemoUnit.hpp
//  MNN
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DemoUnit_hpp
#define DemoUnit_hpp

#include <assert.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <vector>
#include <MNN/expr/Expr.hpp>
using namespace MNN::Express;

/** test case */
class DemoUnit {
    friend class DemoUnitSet;

public:
    /**
     * @brief deinitializer
     */
    virtual ~DemoUnit() = default;
    /**
     * @brief run test case
     */
    virtual int run(int argc, const char* argv[]) = 0;

private:
    /** case name */
    std::string name;
};

/** test suite */
class DemoUnitSet {
public:
    /**
     * @brief deinitializer
     */
    ~DemoUnitSet();
    /**
     * @brief get shared instance
     * @return shared instance
     */
    static DemoUnitSet* get();

public:
    /**
     * @brief register runable test case
     * @param test test case
     * @param name case name
     */
    void add(DemoUnit* test, const char* name);

    /**
     * @brief run registered test case that matches in name
     * @param name case name
     */
    DemoUnit* search(const char* name);

    const std::map<std::string, DemoUnit*>& list() const {
        return mUnit;
    }

private:
    DemoUnitSet(){};
    /** get shared instance */
    static DemoUnitSet* gInstance;
    /** registered test cases */
    std::map<std::string, DemoUnit*> mUnit;
};

/**
 static register for test case
 */
template <class Case>
class DemoUnitRegister {
public:
    /**
     * @brief initializer. register test case to suite.
     * @param name test case name
     */
    DemoUnitRegister(const char* name) {
        DemoUnitSet::get()->add(new Case, name);
    }
    /**
     * @brief deinitializer
     */
    ~DemoUnitRegister() {
    }
};

#define DemoUnitSetRegister(Case, name) static DemoUnitRegister<Case> __r##Case(name)
#define MNNTEST_ASSERT(x)                                        \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            MNN_ERROR("Error for %s, %d\n", __func__, __LINE__); \
            return false;                                        \
        }                                                        \
    }

#endif
