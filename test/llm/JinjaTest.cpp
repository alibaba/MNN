#include "MNNTestSuite.h"
#include "../../transformers/llm/engine/src/tokenizer/jinja.hpp"

class JinjaDictionaryGetTest : public MNNTestCase {
public:
    bool run(int precision) override {
        const char* source =
            R"({% macro content(m) %}{% if m.get('content') %}{{ m.get('content') }}{% endif %}{% endmacro %}{% for m in messages %}{{ content(m) }}|{{ m.get('missing', 'fallback') }}|{{ m.get('empty', 'fallback') is none }}{% endfor %})";
        jinja::Template tmpl(source);
        auto messages = jinja::json::parse(R"([{"role":"user","content":"hello","empty":null}])");
        auto actual = tmpl.apply_chat_template(messages, true);
        if (actual != "hello|fallback|true") {
            MNN_ERROR("Dictionary get rendered: %s\n", actual.c_str());
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(JinjaDictionaryGetTest, "llm/jinja_dictionary_get");
