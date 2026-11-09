#include "dsp_op_name.h"
#include "dsp/int8_cnn.h"
#include "htp_command.h"

#include <cstdio>
#include <cstring>

namespace {

#define CHECK(condition)                                                                            \
    do {                                                                                            \
        if (!(condition)) {                                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                        \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

bool testStableAbiValues() {
    CHECK(DSP_OP_QUANTIZE_I8 == 50);
    CHECK(DSP_OP_DEQUANTIZE_I8 == 51);
    CHECK(DSP_OP_REQUANTIZE_I8 == 52);
    CHECK(DSP_OP_CONV2D_W8A8 == 53);
    CHECK(DSP_CAP_CONV1X1_W8A8 == (1u << 6));
    CHECK(sizeof(DSPQuantizeParam) == 32);
    CHECK(sizeof(DSPRequantizeCommandParam) == 24);
    CHECK(sizeof(DSPRequantParam) == 20);
    CHECK(sizeof(DSPConv2DInt8Param) == 84);
    return true;
}

bool testDiagnosticNames() {
    CHECK(std::strcmp(getDSPOpName(DSP_OP_QUANTIZE_I8), "QUANTIZE_I8") == 0);
    CHECK(std::strcmp(getDSPOpName(DSP_OP_DEQUANTIZE_I8), "DEQUANTIZE_I8") == 0);
    CHECK(std::strcmp(getDSPOpName(DSP_OP_REQUANTIZE_I8), "REQUANTIZE_I8") == 0);
    CHECK(std::strcmp(getDSPOpName(DSP_OP_CONV2D_W8A8), "CONV2D_W8A8") == 0);
    return true;
}

} // namespace

int main() {
    if (!testStableAbiValues() || !testDiagnosticNames()) {
        return 1;
    }
    std::printf("PASS: Hexagon W8A8 DSP ABI contract\n");
    return 0;
}
