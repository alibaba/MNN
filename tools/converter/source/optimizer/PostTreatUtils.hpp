//
//  PostTreatUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef POSTTREATUTILS_HPP
#define POSTTREATUTILS_HPP

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include "MNN_generated.h"
#include "logkit.h"
class PostConverter {
public:
    PostConverter()                                               = default;
    virtual ~PostConverter()                                      = default;
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const = 0;
    static PostConverter* get(std::string key);
    static void add(std::shared_ptr<PostConverter> converter, std::string key);

private:
    static std::map<std::string, std::shared_ptr<PostConverter>>* getConvertMap();
};

template <class T>
class PostConverterRegister {
public:
    PostConverterRegister(const char* claim) {
        T* instance = new T;
        PostConverter::add(std::shared_ptr<PostConverter>(instance), claim);
    }
};

class PostTreatUtils {
public:
    static MNN::OpT* _findOpByOutputIndex(int outputIndex, const MNN::NetT* net);
    static std::vector<MNN::OpT*> _findOpByInputIndex(int inputIndex, const MNN::NetT* net);
    static void _removeOpInNet(MNN::OpT* op, MNN::NetT* net);
    static bool _isSingleInputOutput(const MNN::OpT* op);

    static int _getOpDecestorCount(MNN::OpT* op, const MNN::NetT* net);
    static bool _replace(std::vector<int>& indexes, int freshIndex, int oldIndex);
    
private:
    PostTreatUtils();
};

#endif // POSTTREATUTILS_HPP
