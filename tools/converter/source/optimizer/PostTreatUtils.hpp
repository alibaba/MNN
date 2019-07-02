//
//  PostTreatUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef POSTTREATUTILS_HPP
#define POSTTREATUTILS_HPP

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include "MNN_generated.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "logkit.h"

class PostTreatUtils {
public:
    PostTreatUtils(std::unique_ptr<MNN::NetT>& net);

    void turnGroupConvolution();

    void turnInnerProduct2Convolution();

    void removeInplaceOp();

    void treatIm2Seq();

    void deleteUnusefulOp();

    void merge2Convolution();

    void reIndexTensor();

    void addTensorType();

    void addConverterForTensorFlowModel();

    void removeDeconvolutionShapeInput();

    void changeBatchnNorm2Scale();

    void turnOnnxPadToTensorflow();

    void pluginConvert();
public:
    std::unique_ptr<MNN::NetT> mNet;
    static const std::set<MNN::OpType> NC4HW4_OPs;

    static const std::set<MNN::OpType> COMPABILITY_OPs;
    static const std::vector<MNN::OpType> DELETE_Ops;

private:
    MNN::OpT* _findOpByOutputIndex(int outputIndex);
    std::vector<MNN::OpT*> _findOpByInputIndex(int inputIndex);
    bool _merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp);
    void _removeOpInNet(MNN::OpT* op);
    bool _isSingleInputOutput(const MNN::OpT* op);

    void _removeOnlyOneDecestorOps(MNN::OpT* op);

    int _getOpDecestorCount(MNN::OpT* op);

private:
    PostTreatUtils();
};

template <typename T>
bool inVector(const std::vector<T>& vec, const T& val) {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}

#endif // POSTTREATUTILS_HPP
