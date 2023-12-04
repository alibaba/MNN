#ifndef UnaryUtils_hpp
#define UnaryUtils_hpp
#include <cmath>
#include <vector>
#include <limits>


template <typename Func, typename T>
static void _unaryOp(void* outputPtr, const void* inputPtr, int elementSize) {
    Func f;
    const T *inputData = (T*)inputPtr;
    T *outputData      = (T *)outputPtr;
    for (int i=0; i<elementSize; ++i) {
        outputData[i] = f(inputData[i]);
    }
}

template <typename T>
struct UnarySquare {
    T operator()(const T &x) const {
        return x * x;
    }
};

template <typename T>
struct UnaryRsqrt {
    T operator()(const T &x) const {
        return 1.f / sqrtf(x);
    }
};

template <typename T>
struct UnarySqrt {
    T operator()(const T &x) const {
        return sqrtf(x);
    }
};

template <typename T>
struct UnaryNeg {
    T operator()(const T &x) const {
        return -x;
    }
};

template <typename T>
struct UnaryExp {
    T operator()(const T &x) const {
        return expf(x);
    }
};

template <typename T>
struct UnaryAbs {
    T operator()(const T &x) const {
        return fabsf((float)x);
    }
};

template <typename T>
struct UnaryCeil {
    T operator()(const T &x) const {
        return ceilf(x);
    }
};
template <typename T>
struct UnaryRecipocal {
    T operator()(const T &x) const {
        return (T)1 / (x);
    }
};
template <typename T>
struct UnaryLog1p {
    T operator()(const T &x) const {
        return (T)logf((T)1 + (x));
    }
};
template <typename T>
struct UnaryLog {
    T operator()(const T &x) const {
        return (T)logf((T)(x));
    }
};
template <typename T>
struct UnaryCos {
    T operator()(const T &x) const {
        return (T)cosf((T)(x));
    }
};
template <typename T>
struct UnarySin {
    T operator()(const T &x) const {
        return (T)sinf((T)(x));
    }
};
template <typename T>
struct UnaryTan {
    T operator()(const T &x) const {
        return (T)tanf((T)(x));
    }
};
template <typename T>
struct UnaryATan {
    T operator()(const T &x) const {
        return (T)atanf((T)(x));
    }
};

template <typename T>
struct UnaryFloor {
    T operator()(const T &x) const {
        return (T)floor((T)(x));
    }
};

template <typename T>
struct UnarySign {
    T operator()(const T &x) const {
        if (x > 0) {
            return 1;
        }
        if (x < 0) {
            return -1;
        }
        return 0;
    }
};

template <typename T>
struct UnaryBNLL {
    T operator()(const T &x) const {
        float r = x > 0 ? (x + log(1. + exp(-x))) : log(1. + exp(x));
        return (T)r;
    }
};

template <typename T>
struct UnaryAcosh {
    T operator()(const T &x) const {
        return (T)acoshf((T)(x));
    }
};

template <typename T>
struct UnarySinh {
    T operator()(const T &x) const {
        return (T)sinhf((T)(x));
    }
};

template <typename T>
struct UnaryAsinh {
    T operator()(const T &x) const {
        return (T)asinhf((T)(x));
    }
};

template <typename T>
struct UnaryAtanh {
    T operator()(const T &x) const {
        return (T)atanhf((T)(x));
    }
};
template <typename T>
struct UnaryRound {
    T operator()(const T &x) const {
        return (T)roundf((T)(x));
    }
};

template <typename T>
struct UnaryCosh {
    T operator()(const T &x) const {
        return (T)coshf((T)(x));
    }
};


template <typename T>
struct UnaryErf {
    T operator()(const T &x) const {
        return erff(x);
    }
};

template <typename T>
struct UnaryErfc {
    T operator()(const T &x) const {
        return erfc(x);
    }
};

template <typename T>
struct UnaryErfinv {
    // referenced from tensorflow
    const int kDegree = 9;
    const std::vector<float> w_less_than_5_constants = {
            2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
            -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
            -0.00417768164f,  0.246640727f,    1.50140941f};
    const std::vector<float> w_greater_than_5_constants = {
            -0.000200214257f, 0.000100950558f, 0.00134934322f,
            -0.00367342844f,  0.00573950773f,  -0.0076224613f,
            0.00943887047f,   1.00167406f,     2.83297682f};

    T operator()(const T &x) const {
        // Compute logarithm of (1+arg) using log1p(arg) which is more precise than
        // log(1+arg) when arg is close to zero. For more details, see
        // https://en.cppreference.com/w/cpp/numeric/math/log1p
        auto w = -log1p(-x * x);
        bool lt = (w < 5.0);
        auto coefficient = [&](int i) {
            if (lt) {
                return w_less_than_5_constants[i];
            } else {
                return w_greater_than_5_constants[i];
            }
        };
        if (lt) {
            w = w - 2.5;
        } else {
            w = sqrt(w) - 3.0;
        }
        auto p = coefficient(0);
        for (int i = 1; i < kDegree; i++) {
            p = coefficient(i) + p * w;
        }
        auto result = p * x;
        if (fabsf(fabsf(x) - 1) < 1e-8) {
            return std::numeric_limits<float>::infinity();
        } else {
            return result;
        }
    }
};

template <typename T>
struct UnaryExpm1 {
    T operator()(const T &x) const {
        return (T)expm1((T)(x));
    }
};

template <typename T>
struct UnaryAsin {
    T operator()(const T &x) const {
        return (T)asin((T)(x));
    }
};

template <typename T>
struct UnaryAcos {
    T operator()(const T &x) const {
        return (T)acos((T)(x));
    }
};
#endif
