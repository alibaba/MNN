//
//  OpGrad.hpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpGrad_hpp
#define OpGrad_hpp
#include <map>
#include <vector>
#include "OpConverter.hpp"
#include "Tensor.hpp"

class MNN_PUBLIC OpGrad {
public:
    enum Type { LINEAR, SEMI_LINEAR, NO_LINEAR };

    OpGrad()          = default;
    virtual ~OpGrad() = default;

    Type type() const {
        return mType;
    }

    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* op,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) = 0;

    virtual bool onGradCommon(MNN::NetT* net, const MNN::OpT* op, std::map<int, std::vector<int>>& backwardTensors);

    class Creator {
    public:
        Creator() {
        }
        virtual ~Creator()                                                       = default;
        virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                                 const std::vector<MNN::Tensor*>& outputs) const = 0;
    };
    static Creator* get(MNN::OpType type);
    static void insert(MNN::OpType type, Creator* creator);

protected:
    Type mType = LINEAR;
};
MNN_PUBLIC std::string numberToString(int index);
#endif
