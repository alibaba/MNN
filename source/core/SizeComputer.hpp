//
//  SizeComputer.hpp
//  MNN
//
//  Created by MNN on 2019/01/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SizeComputer_hpp
#define SizeComputer_hpp

#include <map>
#include <string>
#include <vector>
#include "Execution.hpp"
#include "MNN_generated.h"
#include "Tensor.hpp"
#define FLOPS_M 1000000.0f

namespace MNN {

/** computer for op. calculate input and output tensors' shape. when analyzing model, calculate flops too. */
class SizeComputer {
    friend class SizeComputerSuite;

public:
    /**
     * @brief deinitializer.
     */
    virtual ~SizeComputer() = default;

public:
    /**
     * @brief calculate input and output tensors' shape for given op.
     * @param op        given op.
     * @param inputs    given input tensors.
     * @param outputs   given output tensors.
     * @return true if success, false otherwise.
     */
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const = 0;

    /**
     * @brief calculate the flops of this op with the info of inputs size.
     * @param op        given op.
     * @param inputs    given input tensors.
     * @param outputs   given output tensors.
     * @return the flops in M.
     */
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const;

    /**
     * @brief calculate input and output tensors' shape for any registed op.
     * @param op        given registed op.
     * @param inputs    given input tensors.
     * @param outputs   given output tensors.
     * @return true if success, false otherwise.
     */
    static bool computeOutputSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                  const std::vector<Tensor*>& outputs);
};

/** size computer suite */
class SizeComputerSuite {
public:
    /**
     * @brief deinitializer.
     */
    ~SizeComputerSuite();
    /**
     * @brief get shared instance.
     * @return shared instance.
     */
    static SizeComputerSuite* get();

public:
    /**
     * @brief register size computer for designated type
     * @param computer  size computer
     * @param type      designated type
     */
    void insert(SizeComputer* computer, OpType type);

    /**
     * @brief query size computer for designated type
     * @param type      designated type
     * @return size computer if found, nullptr otherwise.
     */
    SizeComputer* search(OpType type);

private:
    /** shared instance */
    static SizeComputerSuite* gInstance;
    /** registered size computer */
    std::map<OpType, SizeComputer*> mRegistry;
};

/** register for size computer */
template <class T>
class SizeComputerRegister {
public:
    /**
     * @brief initializer. register size computer to suite.
     * @param type  designated type
     */
    SizeComputerRegister(OpType type) {
        T* test               = new T;
        SizeComputerSuite* ts = SizeComputerSuite::get();
        ts->insert(test, type);
    }
};
} // namespace MNN

#ifdef MNN_CODEGEN_REGISTER
#define REGISTER_SHAPE(name, op)                          \
    void ___##name##__##op##__() {                        \
        SizeComputerSuite* ts = SizeComputerSuite::get(); \
        ts->insert(new name, op);                         \
    }
#else
#define REGISTER_SHAPE(name, op) static SizeComputerRegister<name> _Shape##op(op)
#endif

#endif
