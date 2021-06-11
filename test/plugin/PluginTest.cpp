//
//  PluginTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_WITH_PLUGIN

#include <vector>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

#include "./PluginMatMulCommon.hpp"

// Import _intPluginMatMul in order to link the shared library successfully.
extern int _intPluginMatMul;

namespace MNN {

using Express::Expr;
using Express::Variable;
using Express::VARP;

namespace plugin {

static void VecRandInit(std::vector<float>* vec) {
    for (int i = 0; i < vec->size(); ++i) {
        (*vec)[i] = rand() % 255 / 255.f;
    }
}

static VARP _PluginMatMul(VARP x, VARP y, bool transpose_x = false, // NOLINT
                          bool transpose_y = false) {
    std::unique_ptr<OpT> pluginOp(new OpT);
    pluginOp->type = OpType_Plugin;

    auto x_shape = x->getInfo();
    auto y_shape = y->getInfo();
    MNN_CHECK(x_shape->dim.size() == 2, "2-D shape is required.");
    MNN_CHECK(y_shape->dim.size() == 2, "2-D shape is required.");

    int M = x_shape->dim[0];
    int K = x_shape->dim[1];
    int N = y_shape->dim[1];
    if (transpose_x) {
        M = x_shape->dim[1];
        K = x_shape->dim[0];
    }
    if (transpose_y) {
        N = y_shape->dim[0];
        MNN_CHECK(K == y_shape->dim[1], "K dim is not match.");
    } else {
        MNN_CHECK(K == y_shape->dim[0], "K dim is not match.");
    }

    PluginT* plugin_param = new PluginT;
    plugin_param->type    = "PluginMatMul";
    plugin_param->attr.resize(2);
    plugin_param->attr[0].reset(new AttributeT);
    plugin_param->attr[0]->key = "transpose_x";
    plugin_param->attr[0]->b   = transpose_x;
    plugin_param->attr[1].reset(new AttributeT);
    plugin_param->attr[1]->key = "transpose_y";
    plugin_param->attr[1]->b   = transpose_y;

    pluginOp->main.type  = OpParameter_Plugin;
    pluginOp->main.value = plugin_param;
    return Variable::create(Expr::create(pluginOp.get(), {x, y}));
}

struct PluginTestHelper {
    bool operator()();
};

bool PluginTestHelper::operator()() {
    VARP x = _Input({3, 10}, Express::NCHW);
    VARP y = _Input({10, 3}, Express::NCHW);
    VARP z = _PluginMatMul(x, y, false, false);

    std::vector<float> x_data(30);
    std::vector<float> y_data(30);
    VecRandInit(&x_data);
    VecRandInit(&y_data);
    memcpy(x->writeMap<float>(), x_data.data(), x_data.size() * sizeof(float));
    memcpy(y->writeMap<float>(), y_data.data(), y_data.size() * sizeof(float));
    const float* z_data = z->readMap<float>();

    std::vector<float> out(9);
    doGemm(3, 3, 10, false, false, x_data.data(), y_data.data(), out.data());
    for (int i = 0; i < 9; ++i) {
        if ((abs(out[i] - z_data[i]) > 1e-5)) {
            MNN_ERROR("z[%i] = %f\n, but %f is right.", i, z_data[i], out[i]);
            return false;
        }
    }
    return true;
}

} // namespace plugin
} // namespace MNN

class PluginTest : public MNNTestCase {
public:
    bool run(int precision) override {
        // The statment in `MNN_ASSERT` will be ignored for release version, so
        // the plugin dynamic library will be linked failed.
        // MNN_ASSERT(_intPluginMatMul == 10);
        if (_intPluginMatMul != 10) {
            MNN_ERROR("intPluginMatMul should be 10 other than %d.\n",  // NOLINT
                      _intPluginMatMul);
            return false;
        }
        // Run plugin unittest.
        return MNN::plugin::PluginTestHelper()();
    }
};

MNNTestSuiteRegister(PluginTest, "plugin");

#endif // MNN_WITH_PLUGIN
