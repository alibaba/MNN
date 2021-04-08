#include <math.h>
#include <algorithm>

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMax {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return std::max(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMin {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return std::min(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMul {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x * y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryAdd {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x + y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinarySub {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryRealDiv {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x / y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMod {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - x / y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryGreater {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x > y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLess {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x < y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryGreaterEqual {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x >= y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLessEqual {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x <= y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryEqual {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x == y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryFloorDiv {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return floor(static_cast<float>(x) / y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryFloorMod {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - floor(x / y) * y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinarySquaredDifference {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (x - y) * (x - y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryPow {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return pow(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryAtan2 {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return atan(x / y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLogicalOr {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x || y) ? 1 : 0);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryNotEqual {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x != y) ? 1 : 0);
    }
};
