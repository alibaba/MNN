#include "HexagonCommandStatus.hpp"

#include <cstdio>

namespace {

#define CHECK(condition)                                                                            \
    do {                                                                                            \
        if (!(condition)) {                                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                        \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

bool testFirstFailureIsStickyUntilConsumed() {
    MNN::HexagonCommandStatus status;
    CHECK(status.peek() == 0);
    status.latch(0);
    CHECK(status.peek() == 0);
    status.latch(73);
    status.latch(91);
    CHECK(status.peek() == 73);
    CHECK(status.consume() == 73);
    CHECK(status.peek() == 0);
    CHECK(status.consume() == 0);
    return true;
}

} // namespace

int main() {
    if (!testFirstFailureIsStickyUntilConsumed()) {
        return 1;
    }
    std::printf("PASS: Hexagon command failure status contract\n");
    return 0;
}
