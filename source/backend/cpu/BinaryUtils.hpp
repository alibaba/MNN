#include <math.h>
#include <algorithm>

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMax : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return std::max(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMin : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return std::min(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMul : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x * y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryAdd : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x + y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinarySub : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryRealDiv : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x / y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMod : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - x / y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryGreater : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x > y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLess : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x < y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryGreaterEqual : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x >= y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLessEqual : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x <= y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryEqual : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x == y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryFloorDiv : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return floor(static_cast<float>(x) / y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryFloorMod : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - floor(x / y) * y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinarySquaredDifference : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (x - y) * (x - y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryPow : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return pow(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryAtan2 : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return atan(x / y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLogicalOr : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x || y) ? 1 : 0);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryNotEqual : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x != y) ? 1 : 0);
    }
};
