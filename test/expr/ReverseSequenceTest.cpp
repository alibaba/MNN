//
//  ReverseSequenceTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"

using namespace MNN::Express;

class ReverseSequenceTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        // high dimension, batch_dim ahead
        {
            auto y               = _Input({4}, NHWC, halide_type_of<int32_t>());
            std::vector<int> seq = {7, 2, 3, 5};
            auto yPtr            = y->writeMap<int32_t>();
            ::memcpy(yPtr, seq.data(), seq.size() * sizeof(int32_t));
            auto x    = _Input({6, 4, 7, 10, 8}, NHWC, halide_type_of<float>());
            auto xPtr = x->writeMap<float>();
            for (int o = 0; o < 6; ++o) {
                for (int i = 0; i < 4; ++i) {
                    for (int m = 0; m < 7; ++m) {
                        for (int j = 0; j < 10; ++j) {
                            for (int k = 0; k < 8; ++k) {
                                xPtr[2240 * o + 560 * i + 80 * m + 8 * j + k] = 10000 * o + 1000 * i + 100 * m + 10 * j + k;
                            }
                        }
                    }
                }
            }

            auto ry    = _ReverseSequence(x, y, 1, 3);
            auto ryPtr = ry->readMap<float>();

            auto func_equal = [](float a, float b) -> bool {
                if (a - b > 0.0001 || a - b < -0.0001) {
                    return false;
                } else {
                    return true;
                }
            };

            int count = 0;
            for (int o = 0; o < 6; ++o) {
                for (int i = 0; i < 4; ++i) {
                    auto req = seq[i];
                    for (int m = 0; m < 7; ++m) {
                        for (int j = 0; j < 10; ++j) {
                            for (int k = 0; k < 8; ++k) {
                                float compute = ryPtr[2240 * o + 560 * i + 80 * m + 8 * j + k];
                                float need    = 10000 * o + 1000 * i + 100 * m + 10 * j + k;
                                if (j < req) {
                                    need = 10000 * o + 1000 * i + 100 * m + 10 * (req - j - 1) + k;
                                }

                                if (!func_equal(need, compute)) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
            return true;
        }

        // high dimension, seq_dim ahead
        {
            auto y               = _Input({4}, NHWC, halide_type_of<int32_t>());
            std::vector<int> seq = {7, 2, 3, 5};
            auto yPtr            = y->writeMap<int32_t>();
            ::memcpy(yPtr, seq.data(), seq.size() * sizeof(int32_t));
            auto x    = _Input({6, 10, 7, 4, 8}, NHWC, halide_type_of<float>());
            auto xPtr = x->writeMap<float>();
            for (int o = 0; o < 6; ++o) {
                for (int i = 0; i < 10; ++i) {
                    for (int m = 0; m < 7; ++m) {
                        for (int j = 0; j < 4; ++j) {
                            for (int k = 0; k < 8; ++k) {
                                xPtr[2240 * o + 224 * i + 32 * m + 8 * j + k] = 10000 * o + 1000 * i + 100 * m + 10 * j + k;
                            }
                        }
                    }
                }
            }

            auto ry    = _ReverseSequence(x, y, 3, 1);
            auto ryPtr = ry->readMap<float>();

            auto func_equal = [](float a, float b) -> bool {
                if (a - b > 0.0001 || a - b < -0.0001) {
                    return false;
                } else {
                    return true;
                }
            };

            int count = 0;
            for (int o = 0; o < 6; ++o) {
                for (int i = 0; i < 10; ++i) {
                    for (int m = 0; m < 7; ++m) {
                        for (int j = 0; j < 4; ++j) {
                            auto req = seq[j];
                            for (int k = 0; k < 8; ++k) {
                                auto compute = ryPtr[2240 * o + 224 * i + 32 * m + 8 * j + k];
                                auto need    = 10000 * o + 1000 * i + 100 * m + 10 * j + k;
                                if (i < req) {
                                    need = 10000 * o + 1000 * (req - i - 1) + 100 * m + 10 * j + k;
                                }
                                if (!func_equal(need, compute)) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
            return true;
        }

        // 3 dimension
        {
            auto y               = _Input({4}, NHWC, halide_type_of<int32_t>());
            std::vector<int> seq = {7, 2, 3, 5};
            auto yPtr            = y->writeMap<int32_t>();
            ::memcpy(yPtr, seq.data(), seq.size() * sizeof(int32_t));
            auto x    = _Input({10, 4, 8}, NHWC, halide_type_of<float>());
            auto xPtr = x->writeMap<float>();
            for (int i = 0; i < 10; ++i) {
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        xPtr[32 * i + 8 * j + k] = 100 * i + 10 * j + k;
                    }
                }
            }

            auto ry    = _ReverseSequence(x, y, 1, 0);
            auto ryPtr = ry->readMap<float>();

            auto func_equal = [](float a, float b) -> bool {
                if (a - b > 0.0001 || a - b < -0.0001) {
                    return false;
                } else {
                    return true;
                }
            };

            for (int i = 0; i < 10; ++i) {
                for (int j = 0; j < 4; ++j) {
                    auto req = seq[j];
                    for (int k = 0; k < 8; ++k) {
                        auto compute = ryPtr[32 * i + 8 * j + k];
                        auto need    = 100 * i + 10 * j + k;
                        if (i < req) {
                            need = 100 * (req - i - 1) + 10 * j + k;
                        }
                        if (!func_equal(need, compute)) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
    }
};
MNNTestSuiteRegister(ReverseSequenceTest, "expr/ReverseSequence");
