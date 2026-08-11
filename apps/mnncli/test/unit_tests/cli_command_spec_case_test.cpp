#include <catch2/catch_test_macros.hpp>

#include "cli_command_spec.hpp"

namespace {
bool HasValueOption(const mnncli::CommandSpec& spec, const std::string& long_name) {
    for (const auto& opt : spec.options) {
        if (opt.long_name == long_name && opt.kind == mnncli::ArgKind::Value) {
            return true;
        }
    }
    return false;
}
} // namespace

TEST_CASE("run command exposes thinking switch option", "[cli_command_spec]") {
    const auto& spec = mnncli::GetSpec("run");
    REQUIRE(HasValueOption(spec, "thinking"));
}

TEST_CASE("serve command exposes thinking switch option", "[cli_command_spec]") {
    const auto& spec = mnncli::GetSpec("serve");
    REQUIRE(HasValueOption(spec, "thinking"));
}
