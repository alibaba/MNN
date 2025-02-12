//
//  SizeComputer.hpp
//  MNN
//
//  Created by MNN on 2019/01/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SizeComputer_hpp
#define SizeComputer_hpp

#include <MNN/Tensor.hpp>
#include <map>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#define FLOPS_M 1000000.0f

namespace MNN {

/** computer for op. calculate input and output tensors' shape. when analyzing model, calculate flops too. */
class MNN_PUBLIC SizeComputer {
    friend class SizeComputerSuite;

public:
    void setInputIndex(std::vector<int>&& index) {
        mNeedContentInputIndex = std::move(index);
    }
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
    static float computeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs);

    static bool computeBroadCastDims(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs);

    static std::vector<int> needInputContent(const MNN::Op* op, int inputSize);
private:
    std::vector<int> mNeedContentInputIndex;
};

/** size computer suite */
class MNN_PUBLIC SizeComputerSuite {
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
    static void init();

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
    std::vector<SizeComputer*> mRegistry;
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
    SizeComputerRegister(OpType type, std::vector<int>&& index) {
        T* test = new T;
        test->setInputIndex(std::move(index));
        SizeComputerSuite* ts = SizeComputerSuite::get();
        ts->insert(test, type);
    }
};
} // namespace MNN
#define REGISTER_SHAPE(name, op)                          \
    void ___##name##__##op##__() {                        \
        name* _temp = new name;                            \
        SizeComputerSuite* ts = SizeComputerSuite::get(); \
        ts->insert(_temp, op);                           \
    }
// Output shape is depent on index-input content data
#define REGISTER_SHAPE_INPUTS(name, op, index)            \
    void ___##name##__##op##__() {                        \
        SizeComputerSuite* ts = SizeComputerSuite::get(); \
        name* computer = new name;                             \
        computer->setInputIndex(index);                    \
        ts->insert(computer, op);                        \
    }

#ifdef MNN_SUPPORT_DEPRECATED_OP
#define REGISTER_SHAPE_OLD(name, op)                          \
    void ___##name##__##op##__() {                        \
        name* _temp = new name;                            \
        SizeComputerSuite* ts = SizeComputerSuite::get(); \
        ts->insert(_temp, op);                           \
    }
#else
#define REGISTER_SHAPE_OLD(name, op) void ___##name##__##op##__() {}

#endif


#ifdef MNN_SUPPORT_RENDER
#define REGISTER_SHAPE_INPUTS_RENDER(name, op, index)            \
    void ___##name##__##op##__() {                        \
        SizeComputerSuite* ts = SizeComputerSuite::get(); \
        name* computer = new name;                             \
        computer->setInputIndex(index);                    \
        ts->insert(computer, op);                        \
    }

#else
#define REGISTER_SHAPE_INPUTS_RENDER(name, op, index) void ___##name##__##op##__() {}

#endif

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#define REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(name, op)            \
    void ___##name##__##op##__() {                        \
        name* _temp = new name;                            \
        SizeComputerSuite* ts = SizeComputerSuite::get(); \
        ts->insert(_temp, op);                           \
    }

#else
#define REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(name, op) void ___##name##__##op##__() {}

#endif

#endif
