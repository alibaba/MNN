//
//  Initializer.cpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Initializer.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <cmath>
#include <vector>
#include "Distributions.hpp"
#include "RandomGenerator.hpp"

namespace MNN {
namespace Express {

Express::VARP Initializer::createConstVar(Express::INTS dim, Express::Dimensionformat format) {
    auto res = Express::_Input(dim, format, halide_type_of<float>());
    this->onExecute(res);
    res.fix(Express::VARP::CONSTANT);
    return res;
}

class ConstantInitializer : public Initializer {
public:
    ConstantInitializer(float value) : mConstant(value) {
    }

    virtual void onExecute(Express::VARP p) override {
        const int count = p->getInfo()->size;
        MNN_ASSERT(count > 0);
        auto ptr = p->writeMap<float>();
        for (int i = 0; i < count; i++) {
            ptr[i] = mConstant;
        }
    }

private:
    float mConstant;
};
Initializer* Initializer::constValue(float value) {
    return new ConstantInitializer(value);
}

class UniformInitializer : public Initializer {
public:
    UniformInitializer(float min = 0, float max = 1) {
        mMin = min;
        mMax = max;
    }

    virtual void onExecute(Express::VARP p) override {
        const int count = p->getInfo()->size;
        MNN_ASSERT(count > 0);
        Distributions::uniform(count, mMin, mMax, p->writeMap<float>(), RandomGenerator::generator());
    }

private:
    float mMin;
    float mMax;
};

Initializer* Initializer::uniform(float minValue, float maxValue) {
    return new UniformInitializer(minValue, maxValue);
}

class XavierInitializer : public Initializer {
public:
    XavierInitializer(VarianceNorm norm = FANIN) {
        mNorm = norm;
    }

    virtual void onExecute(Express::VARP p) override {
        const int count = p->getInfo()->size;
        MNN_ASSERT(count > 0);
        const std::vector<int> dims = p->getInfo()->dim;
        // referenced from Caffe
        // https://github.com/BVLC/caffe/blob/master/include/caffe/filler.hpp
        int fanIn  = count / dims[0];
        int fanOut = dims.size() > 1 ? count / dims[1] : count;
        float n    = fanIn; // default: FANIN
        if (mNorm == VarianceNorm::AVERAGE) {
            n = (fanIn + fanOut) / 2.0f;
        } else if (mNorm == VarianceNorm::FANOUT) {
            n = fanOut;
        }
        float scale = sqrtf(3.0f / n);

        Distributions::uniform(count, -scale, scale, p->writeMap<float>(), RandomGenerator::generator());
    }

private:
    VarianceNorm mNorm;
};
Initializer* Initializer::xavier(VarianceNorm norm) {
    return new XavierInitializer(norm);
}

class GaussianInitializer : public Initializer {
public:
    GaussianInitializer(float mean = 0, float std = 1) {
        mMean = mean;
        mStd  = std;
    }

    virtual void onExecute(Express::VARP p) override {
        const int count = p->getInfo()->size;
        MNN_ASSERT(count > 0);
        Distributions::gaussian(count, mMean, mStd, p->writeMap<float>(), RandomGenerator::generator());
    }

private:
    float mMean;
    float mStd;
};
Initializer* Initializer::gauss(float mean, float std) {
    return new GaussianInitializer(mean, std);
}

class MSRAInitializer : public Initializer {
public:
    MSRAInitializer(VarianceNorm norm = FANIN) {
        mNorm = norm;
    }

    virtual void onExecute(Express::VARP p) override {
        const int count = p->getInfo()->size;
        MNN_ASSERT(count > 0);
        const std::vector<int> dims = p->getInfo()->dim;
        // referenced from Caffe
        // https://github.com/BVLC/caffe/blob/master/include/caffe/filler.hpp
        int fanIn  = count / dims[0];
        int fanOut = dims.size() > 1 ? count / dims[1] : count;
        float n    = fanIn; // default: FANIN
        if (mNorm == VarianceNorm::AVERAGE) {
            n = (fanIn + fanOut) / 2.0f;
        } else if (mNorm == VarianceNorm::FANOUT) {
            n = fanOut;
        }
        float std = sqrtf(2.0f / n);

        Distributions::gaussian(count, 0.0f, std, p->writeMap<float>(), RandomGenerator::generator());
    }

private:
    VarianceNorm mNorm;
};
Initializer* Initializer::MSRA(VarianceNorm norm) {
    return new MSRAInitializer(norm);
}

class BilinearInitializer : public Initializer {
public:
    BilinearInitializer() = default;

    virtual void onExecute(Express::VARP p) override {
        const int count = p->getInfo()->size;
        MNN_ASSERT(count > 0);
        const std::vector<int> dims = p->getInfo()->dim;
        MNN_ASSERT(dims.size() == 4);
        MNN_ASSERT(dims[2] == dims[3]); // NCHW, H == W
        // referenced from Caffe
        // https://github.com/BVLC/caffe/blob/master/include/caffe/filler.hpp
        int f   = ceilf(dims[3] / 2.0f);
        float c = (dims[3] - 1) / (2.0f * f);
        auto ptr = p->writeMap<float>();

        for (int i = 0; i < count; i++) {
            float x                 = i % dims[3];
            float y                 = (i / dims[3]) % dims[2];
            ptr[i] = (1 - std::fabs(x / f - c)) * (1 - std::fabs(y / f - c));
        }
    }
};
Initializer* Initializer::bilinear() {
    return new BilinearInitializer();
}

class PositiveUnitball : public Initializer {
public:
    PositiveUnitball() = default;

    virtual void onExecute(Express::VARP p) override {
        const int count = p->getInfo()->size;
        MNN_ASSERT(count > 0);
        const std::vector<int> dims = p->getInfo()->dim;
        auto ptr = p->writeMap<float>();

        Distributions::uniform(count, 0, 1, ptr, RandomGenerator::generator());

        int dim = count / dims[0];
        for (int i = 0; i < dims[0]; i++) {
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += ptr[i * dim + j];
            }
            for (int j = 0; j < dim; j++) {
                ptr[i * dim + j] = ptr[i * dim + j] / sum;
            }
        }
    }
};
Initializer* Initializer::positiveUnitball() {
    return new PositiveUnitball();
}

} // namespace Express
} // namespace MNN
