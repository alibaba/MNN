#include "MNNTestSuite.h"
#include "tokenizer/jinja.hpp"

class JinjaDefaultTest : public MNNTestCase {
    bool run(int precision) override {
        // Expected values follow Python Jinja2's default Undefined environment.
        const struct {
            const char* context;
            const char* expected;
            const char* booleanExpected;
        } cases[] = {
            {"{}", "fallback", "fallback"},
            {"{\"value\":null}", "None", "fallback"},
            {"{\"value\":false}", "False", "fallback"},
            {"{\"value\":0}", "0", "fallback"},
            {"{\"value\":\"\"}", "", "fallback"},
            {"{\"value\":[]}", "[]", "fallback"},
            {"{\"value\":{}}", "{}", "fallback"},
            {"{\"value\":\"text\"}", "text", "text"},
        };
        for (const auto& item : cases) {
            auto context = jinja::json::parse(item.context);
            MNNTEST_ASSERT(jinja::Template("{{ value | default('fallback') | string }}").render(context) == item.expected);
            MNNTEST_ASSERT(jinja::Template("{{ value | d('fallback') | string }}").render(context) == item.expected);
            MNNTEST_ASSERT(jinja::Template("{{ value | default('fallback', true) | string }}").render(context)
                           == item.booleanExpected);
            MNNTEST_ASSERT(jinja::Template("{{ value | default(boolean=true, default_value='fallback') | string }}")
                           .render(context) == item.booleanExpected);
        }
        auto context = jinja::json::object();
        context["obj"] = jinja::json::object();
        MNNTEST_ASSERT(jinja::Template("{{ missing | default }}").render(context).empty());
        MNNTEST_ASSERT(jinja::Template("{{ missing | string }}").render(context).empty());
        MNNTEST_ASSERT(jinja::Template("{{ 'a' ~ missing ~ 'b' }}").render(context) == "ab");
        MNNTEST_ASSERT(jinja::Template("{{ [missing] | string }}").render(context) == "[Undefined]");
        MNNTEST_ASSERT(jinja::Template("{{ obj.missing | default('fallback') }}").render(context) == "fallback");
        MNNTEST_ASSERT(jinja::Template("{{ obj.get('missing') | default('fallback') | string }}").render(context) == "None");
        const char* thinking = "{% set enable_thinking = enable_thinking | default(true) %}"
                               "{% if enable_thinking %}<think>{% endif %}";
        MNNTEST_ASSERT(jinja::Template(thinking).render(context) == "<think>");
        context["enable_thinking"] = false;
        MNNTEST_ASSERT(jinja::Template(thinking).render(context).empty());
        return true;
    }
};

MNNTestSuiteRegister(JinjaDefaultTest, "llm/jinja_default");
