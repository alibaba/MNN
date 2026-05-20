/*
 jinja.hpp - A lightweight, single-header C++11 Jinja2 template engine for LLM chat templates.
 https://github.com/wangzhaode/jinja.cpp

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <regex>
#include <initializer_list>
#include <ctime>
#include <iomanip>
#include <chrono>

// External dependency: nlohmann/json
#include "../ujson.hpp"

#define JINJA_VERSION_MAJOR 0
#define JINJA_VERSION_MINOR 0
#define JINJA_VERSION_PATCH 1
#define JINJA_VERSION_STRING STR(JINJA_VERSION_MAJOR) "." STR(JINJA_VERSION_MINOR) "." STR(JINJA_VERSION_PATCH)

#ifdef JINJA_DEBUG
    #define JINJA_LOG(x) std::cerr << "[JINJA_DEBUG] " << x << std::endl
#else
    #define JINJA_LOG(x) do {} while(0)
#endif

namespace jinja {

using json = ujson::json;


using Argument = std::pair<std::string, json>;
using UserFunction = std::function<json(const std::vector<Argument>&)>;

/**
 * @brief A lightweight, C++11 compatible Jinja2 template renderer.
 *
 * Designed specifically for LLM chat templates (HuggingFace style).
 * It supports a subset of Jinja2 syntax used in modern models.
 */
class Template {
public:
    /**
     * @brief Construct and compile a Jinja template.
     *
     * @param template_str The Jinja2 template string.
     * @param default_context Optional global variables.
     */
    inline Template(const std::string& template_str, const json& default_context = json::object());

    /**
     * @brief Destructor.
     */
    inline ~Template();

    // Move semantics
    Template(Template&&) noexcept;
    Template& operator=(Template&&) noexcept;

    // Copy semantics deleted
    Template(const Template&) = delete;
    Template& operator=(const Template&) = delete;

    /**
     * @brief Core rendering function.
     */
    inline std::string render(const json& context) const;

    /**
     * @brief Register a custom function.
     */
    inline void add_function(const std::string& name, UserFunction func);

    /**
     * @brief Helper function mimicking HuggingFace's `apply_chat_template`.
     */
    inline std::string apply_chat_template(
        const json& messages,
        bool add_generation_prompt = true,
        const json& tools = json::array(),
        const json& extra_context = json::object()
    ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    void register_builtins();
};

} // namespace jinja

// --- Implementation ---

#include <utility>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cctype>

namespace jinja {


// Use mnn_make_unique to avoid ADL ambiguity with std::make_unique on MSVC C++14+
template<typename T, typename... Args>
std::unique_ptr<T> mnn_make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

inline std::string to_python_string(const json& val);

inline std::string token_type_to_string(int type) {
    switch (type) {
        case 0: return "Text";
        case 1: return "ExprStart";
        case 2: return "ExprEnd";
        case 3: return "BlockStart";
        case 4: return "BlockEnd";
        case 5: return "Identifier";
        case 6: return "String";
        case 7: return "Number";
        case 8: return "Operator";
        case 9: return "Punctuation";
        case 10: return "Eof";
        default: return "Unknown";
    }
}

inline std::string to_python_repr(const json& val) {
    if (val.is_string()) {
         std::string s = val.get<std::string>();
         std::string out = "'";
         for (char c : s) {
             if (c == '\'') out += "\\'";
             else if (c == '\\') out += "\\\\";
             else if (c == '\n') out += "\\n";
             else if (c == '\r') out += "\\r";
             else out += c;
         }
         out += "'";
         return out;
    }
    return to_python_string(val);
}

inline std::string to_json_string(const json& val, int indent = -1, int level = 0) {
    if (val.is_null()) return "null";
    if (val.is_boolean()) return val.get<bool>() ? "true" : "false";
    if (val.is_number()) return val.dump();
    if (val.is_string()) return val.dump();

    std::string nl = (indent >= 0) ? "\n" : "";
    std::string sp = (indent >= 0) ? std::string((level + 1) * indent, ' ') : "";
    std::string sp_closing = (indent >= 0) ? std::string(level * indent, ' ') : "";
    std::string sep = (indent >= 0) ? ", " : ", ";
    if (indent >= 0) sep = "," + nl + sp;

    if (val.is_array()) {
        if (val.empty()) return "[]";
        std::string res = "[" + nl + sp;
        bool first = true;
        for (const auto& el : val) {
            if (!first) res += sep;
            res += to_json_string(el, indent, level + 1);
            first = false;
        }
        res += nl + sp_closing + "]";
        return res;
    }

    if (val.is_object()) {
        if (val.empty()) return "{}";
        std::string res = "{" + nl + sp;

        std::vector<std::string> keys;
        for (json::const_iterator it = val.begin(); it != val.end(); ++it) keys.push_back(it.key());

        auto get_prio = [](const std::string& k) -> int {
            if (k == "type") return 1;
            if (k == "function") return 2;
            if (k == "name") return 3;
            if (k == "description") return 4;
            if (k == "parameters") return 5;
            if (k == "properties") return 6;
            if (k == "required") return 7;
            if (k == "enum") return 8;
            return 100;
        };

        std::sort(keys.begin(), keys.end(), [&](const std::string& a, const std::string& b){
            int pa = get_prio(a);
            int pb = get_prio(b);
            if (pa != pb) return pa < pb;
            return a < b;
        });

        bool first = true;
        for (const auto& key : keys) {
            if (!first) res += sep;
            res += "\"" + key + "\": ";
            // if (indent >= 0) res += " "; // Already have space in ": "
            res += to_json_string(val.at(key), indent, level + 1);
            first = false;
        }
        res += nl + sp_closing + "}";
        return res;
    }
    return val.dump();
}

inline std::string to_python_string(const json& val) {
    if (val.is_null()) return "None";
    if (val.is_boolean()) return val.get<bool>() ? "True" : "False";
    if (val.is_number()) return val.dump();
    if (val.is_string()) return val.get<std::string>();

    if (val.is_array()) {
        std::string res = "[";
        bool first = true;
        for (const auto& el : val) {
            if (!first) res += ", ";
            res += to_python_repr(el);
            first = false;
        }
        res += "]";
        return res;
    }
    if (val.is_object()) {
        std::string res = "{";

        std::vector<std::string> keys;
        for (json::const_iterator it = val.begin(); it != val.end(); ++it) keys.push_back(it.key());

        auto get_prio = [](const std::string& k) -> int {
            if (k == "type") return 1;
            if (k == "function") return 2;
            if (k == "name") return 3;
            if (k == "description") return 4;
            if (k == "parameters") return 5;
            if (k == "properties") return 6;
            if (k == "required") return 7;
            if (k == "enum") return 8;
            return 100;
        };

        std::sort(keys.begin(), keys.end(), [&](const std::string& a, const std::string& b){
            int pa = get_prio(a);
            int pb = get_prio(b);
            if (pa != pb) return pa < pb;
            return a < b;
        });

        bool first = true;
        for (const auto& key : keys) {
            if (!first) res += ", ";
            std::string key_repr = "'";
            for (char c : key) {
                if (c == '\'') key_repr += "\\'";
                else if (c == '\\') key_repr += "\\\\";
                else key_repr += c;
            }
            key_repr += "'";
            res += key_repr + ": " + to_python_repr(val.at(key));
            first = false;
        }
        res += "}";
        return res;
    }
    return val.dump();
}


struct Token {
    enum Type {
        Text,
        ExpressionStart, // {{
        ExpressionEnd,   // }}
        BlockStart,      // {%
        BlockEnd,        // %}
        Identifier,
        String,
        Number,
        Operator,
        Punctuation,
        Eof
    };
    Type type;
    std::string value;
};

class Lexer {
public:
    explicit Lexer(const std::string& input) : m_input(input), m_cursor(0) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        bool trim_next = false;

        auto add_token = [&](Token::Type t, const std::string& v) {
            tokens.push_back({t, v});
            JINJA_LOG("Lexer Token: [" << token_type_to_string(t) << "] "
                      << (v == "\n" ? "\\n" : v));
        };
        JINJA_LOG("Lexer: Start tokenizing input length " << m_input.length());

        while (m_cursor < m_input.length()) {
            if (m_state == State::Text) {
                // Find next {{ or {% or {#
                size_t next_expr = m_input.find("{{", m_cursor);
                size_t next_block = m_input.find("{%", m_cursor);
                size_t next_comment = m_input.find("{#", m_cursor);

                size_t next = std::string::npos;
                int token_kind = 0; // 1: expr, 2: block, 3: comment

                // Find the nearest one
                if (next_expr != std::string::npos) { next = next_expr; token_kind = 1; }
                if (next_block != std::string::npos && (next == std::string::npos || next_block < next)) { next = next_block; token_kind = 2; }
                if (next_comment != std::string::npos && (next == std::string::npos || next_comment < next)) { next = next_comment; token_kind = 3; }

                if (next == std::string::npos) {
                    std::string text = std::string(m_input.substr(m_cursor));
                    if (trim_next) {
                         text.erase(0, text.find_first_not_of(" \n\r\t"));
                         trim_next = false;
                    }
                    if (!text.empty()) add_token(Token::Text, text);
                    m_cursor = m_input.length();
                } else {
                    std::string text = std::string(m_input.substr(m_cursor, next - m_cursor));
                    if (trim_next) {
                        text.erase(0, text.find_first_not_of(" \n\r\t"));
                        trim_next = false;
                    }

                    // Check logic for trim_prev is shared? Comments also support -# ?
                    bool trim_prev = false;
                    if (next + 3 <= m_input.length()) { // {{x or {%x or {#x
                        if (m_input[next+2] == '-') {
                            trim_prev = true;
                        }
                    }

                    if (trim_prev) {
                        size_t last_non_space = text.find_last_not_of(" \n\r\t");
                        if (last_non_space == std::string::npos) text.clear();
                        else text.erase(last_non_space + 1);
                    } else if (token_kind == 2) {
                        // lstrip_blocks (default true): strip indentation
                        size_t last_newline = text.rfind('\n');
                        size_t search_start = (last_newline == std::string::npos) ? 0 : last_newline + 1;
                        bool only_spaces = true;
                        for (size_t i = search_start; i < text.length(); ++i) {
                            if (text[i] != ' ' && text[i] != '\t') {
                                only_spaces = false;
                                break;
                            }
                        }
                        if (only_spaces && text.length() > search_start) {
                            text.erase(search_start);
                        }
                    }

                    if (!text.empty()) add_token(Token::Text, text);

                    m_cursor = next;
                    if (token_kind == 1) { // Expr
                        add_token(Token::ExpressionStart, "{{");
                        m_state = State::Expression;
                         if (trim_prev) m_cursor++;
                    } else if (token_kind == 2) { // Block
                        add_token(Token::BlockStart, "{%");
                        m_state = State::Block;
                        if (trim_prev) m_cursor++;
                    } else if (token_kind == 3) { // Comment
                        // Skip until #}
                        m_cursor += 2;
                        if (trim_prev) m_cursor++; // skip -

                        // Scan for #} or -#}
                        // Note: Jinja comments can contain anything.
                        while (m_cursor < m_input.length()) {
                            if (m_input.substr(m_cursor, 2) == "#}") {
                                m_cursor += 2;
                                break;
                            }
                            if (m_input.substr(m_cursor, 3) == "-#}") {
                                m_cursor += 3;
                                trim_next = true;
                                break;
                            }
                            m_cursor++;
                        }
                        // Continue to next iteration in State::Text
                        continue;
                    }
                    m_cursor += 2; // for {{ or {%
                }
            } else {
                // Inside Expression or Block
                skip_whitespace();
                if (m_cursor >= m_input.length()) break;

                // Check for end tags with potential trim modifier
                // -}} or -%}
                bool trim_current = false;
                if (m_input[m_cursor] == '-') {
                    if (m_state == State::Expression && m_input.substr(m_cursor, 3) == "-}}") {
                        trim_current = true;
                    } else if (m_state == State::Block && m_input.substr(m_cursor, 3) == "-%}") {
                        trim_current = true;
                    }
                }

                if (trim_current) {
                    if (m_state == State::Expression) add_token(Token::ExpressionEnd, "}}");
                    else add_token(Token::BlockEnd, "%}");
                    m_cursor += 3;
                    m_state = State::Text;
                    trim_next = true;
                    continue;
                }

                // Normal end tags
                if (m_state == State::Expression && m_input.substr(m_cursor, 2) == "}}") {
                    add_token(Token::ExpressionEnd, "}}");
                    m_cursor += 2;
                    m_state = State::Text;
                    continue;
                }
                if (m_state == State::Block && m_input.substr(m_cursor, 2) == "%}") {
                    add_token(Token::BlockEnd, "%}");
                    m_cursor += 2;
                    m_state = State::Text;

                    // trim_blocks: remove first newline
                    if (m_cursor < m_input.length() && m_input[m_cursor] == '\n') {
                        m_cursor++;
                    }
                    continue;
                }

                char c = m_input[m_cursor];
                if (isalpha(c) || c == '_') {
                    add_token(Token::Identifier, read_identifier());
                } else if (isdigit(c)) {
                    add_token(Token::Number, read_number());
                } else if (c == '\'' || c == '"') {
                    add_token(Token::String, read_string(c));
                } else if (strchr("[](){}:.,", c)) {
                    std::string op(1, c);
                    add_token(Token::Punctuation, op);
                    m_cursor++;
                } else {
                    // Operator or other symbols
                    std::string op(1, c);
                    if (m_cursor + 1 < m_input.length()) {
                        char next = m_input[m_cursor + 1];
                        if (c == '=' && next == '=') op = "==";
                        else if (c == '!' && next == '=') op = "!=";
                        else if (c == '<' && next == '=') op = "<=";
                        else if (c == '>' && next == '=') op = ">=";
                    }
                    if (op.length() > 1) m_cursor++; // Consume extra char
                    add_token(Token::Operator, op);
                    m_cursor++;
                }
            }
        }
        add_token(Token::Eof, "");
        return tokens;
    }

private:
    void skip_whitespace() {
        while (m_cursor < m_input.length() && isspace(m_input[m_cursor])) {
            m_cursor++;
        }
    }

    std::string read_identifier() {
        size_t start = m_cursor;
        while (m_cursor < m_input.length() && (isalnum(m_input[m_cursor]) || m_input[m_cursor] == '_')) {
            m_cursor++;
        }
        return std::string(m_input.substr(start, m_cursor - start));
    }

    std::string read_number() {
        size_t start = m_cursor;
        while (m_cursor < m_input.length() && isdigit(m_input[m_cursor])) {
            m_cursor++;
        }
        return std::string(m_input.substr(start, m_cursor - start));
    }

    std::string read_string(char quote) {
        m_cursor++; // skip opening quote
        std::string val;
        while (m_cursor < m_input.length() && m_input[m_cursor] != quote) {
            char c = m_input[m_cursor];
            if (c == '\\') {
                if (m_cursor + 1 < m_input.length()) {
                    char next = m_input[m_cursor + 1];
                    switch (next) {
                        case 'n': val += '\n'; break;
                        case 'r': val += '\r'; break;
                        case 't': val += '\t'; break;
                        case '\\': val += '\\'; break;
                        case '"': val += '"'; break;
                        case '\'': val += '\''; break;
                        case 'b': val += '\b'; break;
                        case 'f': val += '\f'; break;
                        default: val += next; break;
                    }
                    m_cursor += 2;
                } else {
                    val += '\\'; // trailing backslash
                    m_cursor++;
                }
            } else {
                val += c;
                m_cursor++;
            }
        }
        if (m_cursor < m_input.length()) m_cursor++; // skip closing quote
        return val;
    }

    enum class State {
        Text,
        Expression,
        Block
    };

    const std::string& m_input;
    size_t m_cursor;
    State m_state = State::Text;
};


// --- Helpers ---
class Context; // Forward declaration

struct Node {
    virtual ~Node() = default;
    virtual void render(Context& context, std::string& out) = 0;
};

struct Macro;

// Forward declarations
static bool is_truthy(const json& val);
static json undefined_init() {
    json j = json::object();
    j["__jinja_undefined__"] = true;
    return j;
}
static const json UNDEFINED = undefined_init();

inline bool is_undefined(const json& val) {
    return val.is_object() && val.contains("__jinja_undefined__");
}

inline bool is_truthy(const json& val) {
    if (is_undefined(val)) return false;
    if (val.is_boolean()) return val.get<bool>();
    if (val.is_string()) return !val.get<std::string>().empty();
    if (val.is_number_integer()) return val.get<int64_t>() != 0;
    if (val.is_number_float()) return val.get<double>() != 0.0;
    if (val.is_array() || val.is_object()) return !val.empty();
    if (val.is_null()) return false;
    return true;
}


class Context {
    std::vector<json> scopes;
    std::map<std::string, Macro*> macros;
    const std::map<std::string, UserFunction>* functions = nullptr;

public:
    explicit Context(const json& global) {
        scopes.push_back(global);
    }

    void set_functions(const std::map<std::string, UserFunction>* funcs) {
        functions = funcs;
    }

    UserFunction get_function(const std::string& name) const {
        if (functions && functions->count(name)) {
            return functions->at(name);
        }
        return nullptr;
    }

    void register_macro(const std::string& name, Macro* macro) {
        macros[name] = macro;
    }

    Macro* get_macro(const std::string& name) {
        if (macros.count(name)) return macros[name];
        return nullptr;
    }

    json get(const std::string& name) {
        // Search from top to bottom
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            if (it->contains(name)) {
                return (*it)[name];
            }
        }
        JINJA_LOG("Context: Variable '" << name << "' not found, returning UNDEFINED");
        return UNDEFINED;
    }

    json get(const std::string& name) const {
        // Const version
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            if (it->contains(name)) {
                return (*it)[name];
            }
        }
        return UNDEFINED;
    }

    void set(const std::string& name, json val) {
        // Set in current scope
        scopes.back()[name] = std::move(val);
    }

    // Update variable in the scope where it was originally defined (for namespace mutations).
    void update(const std::string& name, json val) {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            if (it->contains(name)) {
                (*it)[name] = std::move(val);
                return;
            }
        }
        // Not found, set in current scope
        scopes.back()[name] = std::move(val);
    }

    void push_scope(json scope = json::object()) {
        scopes.push_back(std::move(scope));
    }

    void pop_scope() {
        if (scopes.size() > 1) {
            scopes.pop_back();
        }
    }

};

// --- AST for Expressions ---

struct Expr {
    enum ExprKind { EXPR_OTHER, EXPR_VAR, EXPR_GETATTR };
    ExprKind kind_ = EXPR_OTHER;
    virtual ~Expr() = default;
    virtual json evaluate(Context& context) = 0;
    virtual std::string dump() const { return "???"; }
};

struct LiteralExpr : Expr {
    json value;
    explicit LiteralExpr(json v) : value(std::move(v)) {}
    json evaluate(Context& context) override { return value; }
    std::string dump() const override { return value.dump(); }
};

struct VarExpr : Expr {
    std::string name;
    explicit VarExpr(std::string n) : name(std::move(n)) { kind_ = EXPR_VAR; }
    json evaluate(Context& context) override {
        json val = context.get(name);
        JINJA_LOG("Eval Var: '" << name << "' -> " << (val.is_string() ? val.get<std::string>() : val.dump()));
        return val;
    }
    std::string dump() const override { return name; }
};

struct GetAttrExpr : Expr {
    std::unique_ptr<Expr> object;
    std::string name;
    GetAttrExpr(std::unique_ptr<Expr> obj, std::string n)
        : object(std::move(obj)), name(std::move(n)) { kind_ = EXPR_GETATTR; }

    json evaluate(Context& context) override {
        json obj_val = object->evaluate(context);
        if (obj_val.is_object() && obj_val.contains(name)) {
            return obj_val[name];
        } else if (obj_val.is_array() && name == "length") {
             return obj_val.size();
        }
        return UNDEFINED;
    }
    std::string dump() const override { return object->dump() + "." + name; }
};


struct SliceExpr : Expr {
    std::unique_ptr<Expr> start, stop, step;
    SliceExpr(std::unique_ptr<Expr> s, std::unique_ptr<Expr> e, std::unique_ptr<Expr> st)
        : start(std::move(s)), stop(std::move(e)), step(std::move(st)) {}

    json evaluate(Context& context) override {
        json j = json::object();
        j["_type"] = "slice";
        if (start) j["start"] = start->evaluate(context);
        else j["start"] = nullptr;

        if (stop) j["stop"] = stop->evaluate(context);
        else j["stop"] = nullptr;

        if (step) j["step"] = step->evaluate(context);
        else j["step"] = nullptr;

        return j;
    }
};

struct GetItemExpr : Expr {
    std::unique_ptr<Expr> object;
    std::unique_ptr<Expr> key;
    GetItemExpr(std::unique_ptr<Expr> obj, std::unique_ptr<Expr> k)
        : object(std::move(obj)), key(std::move(k)) {}

    json evaluate(Context& context) override {
        json obj_val = object->evaluate(context);
        json key_val = key->evaluate(context);

        if (obj_val.is_array()) {
            if (key_val.is_object() && key_val.value("_type", "") == "slice") {
                 // Implement slicing
                 long len = static_cast<long>(obj_val.size());

                 // Decode slice
                 long start = 0, stop = len, step = 1;
                 bool has_start = !key_val["start"].is_null();
                 bool has_stop = !key_val["stop"].is_null();
                 bool has_step = !key_val["step"].is_null();

                 if (has_step) step = key_val["step"].get<long>();
                 if (step == 0) step = 1; // avoid infinite loop

                 if (has_start) {
                     start = key_val["start"].get<long>();
                     if (start < 0) start += len;
                     if (start < 0) start = 0;
                     if (start > len) start = len; // Clamping depends on step sign usually
                 } else {
                     start = (step > 0) ? 0 : len - 1;
                 }

                 if (has_stop) {
                     stop = key_val["stop"].get<long>();
                     if (stop < 0) stop += len;
                 } else {
                     stop = (step > 0) ? len : -1;
                 }

                 // Clamping for safety
                 // Simple iteration
                 json result = json::array();
                 if (step > 0) {
                     for (long i = start; i < stop && i < len; i += step) {
                         result.push_back(obj_val[static_cast<size_t>(i)]);
                     }
                 } else {
                     for (long i = start; i > stop && i >= 0 && i < len; i += step) {
                         result.push_back(obj_val[static_cast<size_t>(i)]);
                     }
                 }
                 return result;
            }

            if (key_val.is_number_integer()) {
                long idx = key_val.get<long>();
                if (idx < 0) idx += (long)obj_val.size();
                if (idx >= 0 && idx < (long)obj_val.size()) return obj_val[static_cast<size_t>(idx)];
            }
        } else if (obj_val.is_object()) {
            if (key_val.is_string()) {
                std::string k = key_val.get<std::string>();
                if (obj_val.contains(k)) return obj_val[k];
            }
        }
        return "";
    }
};

struct CallExpr : Expr {
    std::string func_name;
    std::vector<std::pair<std::string, std::unique_ptr<Expr>>> args; // named args only for now (namespace) or mixed?
    // Simplified: namespace(a=1, b=2) -> named args.

    CallExpr(std::string name, std::vector<std::pair<std::string, std::unique_ptr<Expr>>> a)
        : func_name(std::move(name)), args(std::move(a)) {}

    json evaluate(Context& context) override;
    std::string dump() const override { return func_name + "(...)"; }
};

struct MethodCallExpr : Expr {
    std::unique_ptr<Expr> object;
    std::string method;
    std::vector<std::unique_ptr<Expr>> args;

    MethodCallExpr(std::unique_ptr<Expr> obj, std::string m, std::vector<std::unique_ptr<Expr>> a)
        : object(std::move(obj)), method(std::move(m)), args(std::move(a)) {}

    json evaluate(Context& context) override {
        json obj_val = object->evaluate(context);

        if (obj_val.is_string()) {
            std::string s = obj_val.get<std::string>();
            if (method == "startswith") {
                if (!args.empty()) {
                    std::string arg = args[0]->evaluate(context).get<std::string>();
                    if (s.length() >= arg.length()) {
                        return (0 == s.compare(0, arg.length(), arg));
                    }
                    return false;
                }
            } else if (method == "endswith") {
                 if (!args.empty()) {
                    std::string arg = args[0]->evaluate(context).get<std::string>();
                    if (s.length() >= arg.length()) {
                        return (0 == s.compare(s.length() - arg.length(), arg.length(), arg));
                    }
                    return false;
                }
            } else if (method == "split") {
                 std::string delim = " ";
                 if (!args.empty()) delim = args[0]->evaluate(context).get<std::string>();
                 json arr = json::array();
                 size_t pos = 0;
                 std::string token;
                 // Simple split
                 size_t found = 0;
                 while ((found = s.find(delim, pos)) != std::string::npos) {
                     arr.push_back(s.substr(pos, found - pos));
                     pos = found + delim.length();
                 }
                 arr.push_back(s.substr(pos));
                 return arr;
            } else if (method == "lstrip") {
                // Simplified lstrip (whitespace or chars?)
                // Python lstrip() removes whitespace, lstrip(chars) removes chars.
                 std::string chars = " \n\r\t";
                 if (!args.empty()) chars = args[0]->evaluate(context).get<std::string>();
                 size_t start = s.find_first_not_of(chars);
                 return (start == std::string::npos) ? "" : s.substr(start);
            } else if (method == "rstrip") {
                 std::string chars = " \n\r\t";
                 if (!args.empty()) chars = args[0]->evaluate(context).get<std::string>();
                 size_t end = s.find_last_not_of(chars);
                 return (end == std::string::npos) ? "" : s.substr(0, end + 1);
            } else if (method == "strip") {
                 std::string chars = " \n\r\t";
                 if (!args.empty()) chars = args[0]->evaluate(context).get<std::string>();
                 size_t start = s.find_first_not_of(chars);
                 if (start == std::string::npos) return "";
                 size_t end = s.find_last_not_of(chars);
                 return s.substr(start, end - start + 1);
            } else if (method == "replace") {
                 if (args.size() >= 2) {
                     std::string from = args[0]->evaluate(context).get<std::string>();
                     std::string to = args[1]->evaluate(context).get<std::string>();
                     std::string res = s;
                     if (from.empty()) return res;
                     size_t start_pos = 0;
                     while((start_pos = res.find(from, start_pos)) != std::string::npos) {
                         res.replace(start_pos, from.length(), to);
                         start_pos += to.length();
                     }
                     return res;
                 }
            }
        }
        return "";
    }
};

struct FilterExpr : Expr {
    std::unique_ptr<Expr> left;
    std::string name;
    std::vector<std::pair<std::string, std::unique_ptr<Expr>>> args;

    FilterExpr(std::unique_ptr<Expr> l, std::string n, std::vector<std::pair<std::string, std::unique_ptr<Expr>>> a = {})
        : left(std::move(l)), name(std::move(n)), args(std::move(a)) {}

    json evaluate(Context& context) override {
        json val = left->evaluate(context);
        if (name == "tojson") {
             int indent = -1;
             for (const auto& arg : args) {
                 if (arg.first == "indent") {
                     json v = arg.second->evaluate(context);
                     if (v.is_number_integer()) indent = v.get<int>();
                 }
             }
             if (indent >= 0) return to_json_string(val, indent);
             // Use our sorted serialization for default dump to support Tool ordering expectations
             return to_json_string(val);
        } else if (name == "string") {
            return to_python_string(val);
        } else if (name == "capitalize") {
             if (val.is_string()) {
                 std::string s = val.get<std::string>();
                 if (!s.empty()) {
                     s[0] = std::toupper(s[0]);
                     for (size_t i = 1; i < s.length(); ++i) {
                         s[i] = std::tolower(s[i]);
                     }
                 }
                 return s;
             }
        } else if (name == "length") {
            if (val.is_array() || val.is_object()) return val.size();
            if (val.is_string()) return val.get<std::string>().length();
            return 0;
        } else if (name == "trim") {
            if (val.is_string()) {
                std::string s = val.get<std::string>();
                // Trim logic from MethodCallExpr?
                // Minimal trim:
                auto start = s.find_first_not_of(" \n\r\t");
                if (start == std::string::npos) return "";
                auto end = s.find_last_not_of(" \n\r\t");
                return s.substr(start, end - start + 1);
            }
        } else if (name == "items") {
             if (val.is_object()) {
                 json arr = json::array();
                  for (json::const_iterator it = val.begin(); it != val.end(); ++it) {
                      json pair = json::array();
                      pair.push_back(it.key());
                      pair.push_back(it.value());
                      arr.push_back(pair);
                  }
                 return arr;
             }
             return json::array();
        } else if (name == "lower") {
             if (val.is_string()) {
                 std::string s = val.get<std::string>();
                 std::transform(s.begin(), s.end(), s.begin(),
                                [](unsigned char c){ return std::tolower(c); });
                 return s;
             }
        } else if (name == "upper") {
             if (val.is_string()) {
                 std::string s = val.get<std::string>();
                 std::transform(s.begin(), s.end(), s.begin(),
                                [](unsigned char c){ return std::toupper(c); });
                 return s;
             }
        } else if (name == "map") {
             if (val.is_array()) {
                 std::string attr;
                 for (const auto& arg : args) {
                     // Support map(attribute='name') or map('name')
                     if (arg.first == "attribute" || (arg.first.empty() && attr.empty())) {
                         json v = arg.second->evaluate(context);
                         if (v.is_string()) attr = v.get<std::string>();
                     }
                 }

                 if (attr.empty()) return val;

                 json res = json::array();
                 for (const auto& item : val) {
                     if (item.is_object() && item.contains(attr)) {
                         res.push_back(item.at(attr));
                     } else {
                         res.push_back(nullptr); // undefined behavior: usage usually implies valid
                     }
                 }
                 return res;
             }
        }
        return val; // Unknown filter pass-through
    }
    std::string dump() const override { return left->dump() + "|" + name; }
};

struct BinaryExpr : Expr {
    std::string op;
    std::unique_ptr<Expr> left, right;
    BinaryExpr(std::string o, std::unique_ptr<Expr> l, std::unique_ptr<Expr> r)
        : op(std::move(o)), left(std::move(l)), right(std::move(r)) {}

    json evaluate(Context& context) override {
        json l = left->evaluate(context);
        json r = right->evaluate(context);


        // Basic ops
        if (op == "+") {
            if (l.is_string() && r.is_string()) return l.get<std::string>() + r.get<std::string>();
            if (l.is_number() && r.is_number()) {
                 if (l.is_number_float() || r.is_number_float()) return l.get<double>() + r.get<double>();
                 return l.get<int64_t>() + r.get<int64_t>();
            }
            if (l.is_array() && r.is_array()) {
                json result = json::array();
                for (const auto& item : l) result.push_back(item);
                for (const auto& item : r) result.push_back(item);
                return result;
            }
            if (l.is_array() && !r.is_array()) {
                json result = json::array();
                for (const auto& item : l) result.push_back(item);
                result.push_back(r);
                return result;
            }
            if (!l.is_array() && r.is_array()) {
                json result = json::array();
                result.push_back(l);
                for (const auto& item : r) result.push_back(item);
                return result;
            }
        }
        if (op == "-") {
            if (l.is_number() && r.is_number()) {
                 if (l.is_number_float() || r.is_number_float()) return l.get<double>() - r.get<double>();
                 return l.get<int64_t>() - r.get<int64_t>();
            }
        }
        if (op == "*") {
             if (l.is_number() && r.is_number()) {
                 if (l.is_number_float() || r.is_number_float()) return l.get<double>() * r.get<double>();
                 return l.get<int64_t>() * r.get<int64_t>();
            }
        }
         if (op == "/") {
             if (l.is_number() && r.is_number()) {
                 if (r.get<double>() != 0) return l.get<double>() / r.get<double>();
                 return 0; // div by zero safety?
            }
        }
        if (op == "%") {
             if (l.is_number_integer() && r.is_number_integer()) {
                 if (r.get<int64_t>() != 0) return l.get<int64_t>() % r.get<int64_t>();
            }
        }

        if (op == "==") return l == r;
        if (op == "!=") return l != r;
        if (op == "<") return l < r;
        if (op == ">") return l > r;
        if (op == "<=") return l <= r;
        if (op == ">=") return l >= r;

        if (op == "in") {
             if (r.is_array()) {
                 for (const auto& el : r) if (el == l) return true;
                 return false;
             }
             if (r.is_object() && l.is_string()) {
                 return r.contains(l.get<std::string>());
             }
             if (r.is_string() && l.is_string()) {
                 return r.get<std::string>().find(l.get<std::string>()) != std::string::npos;
             }
             return false;
        }
        if (op == "not in") {
             if (r.is_array()) {
                 for (const auto& el : r) {
                     if (el == l) return false;
                 }
                 return true;
             }
             if (r.is_object() && l.is_string()) {
                 return !r.contains(l.get<std::string>());
             }
             if (r.is_string() && l.is_string()) {
                 return r.get<std::string>().find(l.get<std::string>()) == std::string::npos;
             }
             return true;
        }
        if (op == ">=") return l >= r;

        if (op == "and") {
             return is_truthy(l) && is_truthy(r);
        }
        if (op == "or") {
             return is_truthy(l) || is_truthy(r);
        }
        if (op == "~") {
             std::ostringstream ss;
             ss << to_python_string(l);
             ss << to_python_string(r);
             return ss.str();
        }

        return "";
    }
    std::string dump() const override { return "(" + left->dump() + " " + op + " " + right->dump() + ")"; }
};

struct TestExpr : Expr {
    std::unique_ptr<Expr> expr;
    std::string test_name;
    bool is_not;

    TestExpr(std::unique_ptr<Expr> e, std::string name, bool n)
        : expr(std::move(e)), test_name(std::move(name)), is_not(n) {}

    json evaluate(Context& context) override {
        json val = expr->evaluate(context);
        bool result = false;

        if (test_name == "defined") {
            result = !is_undefined(val);
        } else if (test_name == "undefined") {
            result = is_undefined(val);
        } else if (test_name == "none") {
            result = val.is_null() && !is_undefined(val);
        } else if (test_name == "boolean") {
            result = val.is_boolean();
        } else if (test_name == "string") {
            result = val.is_string();
        } else if (test_name == "number") {
            result = val.is_number();
        } else if (test_name == "sequence" || test_name == "iterable") {
            result = val.is_array() || val.is_string();
        } else if (test_name == "mapping") {
            result = val.is_object();
        } else if (test_name == "true") {
            result = val.is_boolean() && val.get<bool>() == true;
        } else if (test_name == "false") {
            result = val.is_boolean() && val.get<bool>() == false;
        }
        // TODO: other tests
        if (is_not) result = !result;
        JINJA_LOG("TestExpr: " << expr->dump() << " is " << (is_not ? "not " : "") << test_name << " -> " << (result ? "TRUE" : "FALSE"));
        return result;
    }
    std::string dump() const override { return "(" + expr->dump() + " is " + (is_not ? "not " : "") + test_name + ")"; }
};

struct TernaryExpr : Expr {
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Expr> true_expr;
    std::unique_ptr<Expr> false_expr;

    TernaryExpr(std::unique_ptr<Expr> c, std::unique_ptr<Expr> t, std::unique_ptr<Expr> f)
        : condition(std::move(c)), true_expr(std::move(t)), false_expr(std::move(f)) {}

    json evaluate(Context& context) override {
        json c = condition->evaluate(context);
        if (is_truthy(c)) return true_expr->evaluate(context);
        return false_expr->evaluate(context);
    }
    std::string dump() const override { return "(" + true_expr->dump() + " if " + condition->dump() + " else " + false_expr->dump() + ")"; }
};

struct ListExpr : Expr {
    std::vector<std::unique_ptr<Expr>> items;
    explicit ListExpr(std::vector<std::unique_ptr<Expr>> i) : items(std::move(i)) {}

    json evaluate(Context& context) override {
        json arr = json::array();
        for (const auto& item : items) {
             arr.push_back(item->evaluate(context));
        }
        return arr;
    }
    std::string dump() const override { return "[...]"; }
};

struct ObjectExpr : Expr {
    std::vector<std::pair<std::unique_ptr<Expr>, std::unique_ptr<Expr>>> items;
    explicit ObjectExpr(std::vector<std::pair<std::unique_ptr<Expr>, std::unique_ptr<Expr>>> i) : items(std::move(i)) {}

    json evaluate(Context& context) override {
        json obj = json::object();
        for (const auto& item : items) {
             std::string key = item.first->evaluate(context).get<std::string>(); // Keys must be strings in JSON
             obj[key] = item.second->evaluate(context);
        }
        return obj;
    }
    std::string dump() const override { return "{...}"; }
};

// --- AST for Template Nodes ---

// Node definition moved to top
struct Macro {
    std::string name;
    std::vector<std::string> args;
    std::vector<std::unique_ptr<Node>> body;
    ~Macro(); // Defined just below
};

// Macro implementation
Macro::~Macro() = default;

struct MacroNode : Node {
    Macro macro;

    MacroNode(std::string name, std::vector<std::string> args, std::vector<std::unique_ptr<Node>> body) {
        macro.name = std::move(name);
        macro.args = std::move(args);
        macro.body = std::move(body);
    }

    void render(Context& context, std::string& out) override {
        context.register_macro(macro.name, &macro);
    }
};

inline json CallExpr::evaluate(Context& context) {
    if (auto func = context.get_function(func_name)) {
        std::vector<Argument> arg_vals;
        for (auto& arg : args) {
            arg_vals.push_back({arg.first, arg.second->evaluate(context)});
        }
        return func(arg_vals);
    }

    if (auto macro = context.get_macro(func_name)) {
            json scope = json::object();
            size_t i = 0;
            for (const auto& call_arg : args) {
               if (!call_arg.first.empty()) {
                   scope[call_arg.first] = call_arg.second->evaluate(context);
               } else {
                   if (i < macro->args.size()) {
                       scope[macro->args[i]] = call_arg.second->evaluate(context);
                       i++;
                   }
               }
            }
            context.push_scope(scope);
            std::string out;
            for (const auto& node : macro->body) {
                node->render(context, out);
            }
            context.pop_scope();
            return out;
    }

    return "";
}

struct TextNode : Node {
    std::string text;
    explicit TextNode(std::string t) : text(std::move(t)) {}

    void render(Context& context, std::string& out) override {
        out += text;
    }
};

struct PrintNode : Node {
    std::unique_ptr<Expr> expr;
    explicit PrintNode(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
    void render(Context& context, std::string& out) override {
        json val = expr->evaluate(context);
        if (is_undefined(val)) return; // Print nothing
        if (val.is_string()) out += val.get<std::string>();
        else out += val.dump();
    }
};

struct SetNode : Node {
    std::vector<std::string> targets;
    // Qwen uses "set ns.x = ..." which is setting attribute of object.
    // Standard set is "set x = ...".
    // We need to handle both.
    // For now, let's treat target as Expr? Or just handle VarExpr and GetAttrExpr as targets?
    std::unique_ptr<Expr> target;
    std::unique_ptr<Expr> value;

    SetNode(std::unique_ptr<Expr> t, std::unique_ptr<Expr> v)
        : target(std::move(t)), value(std::move(v)) {}

    void render(Context& context, std::string& out) override {
        json val = value->evaluate(context);

        if (target->kind_ == Expr::EXPR_VAR) {
            auto* var = static_cast<VarExpr*>(target.get());
            context.set(var->name, val);
        } else if (target->kind_ == Expr::EXPR_GETATTR) {
            auto* attr = static_cast<GetAttrExpr*>(target.get());
            if (attr->object->kind_ == Expr::EXPR_VAR) {
                auto* var = static_cast<VarExpr*>(attr->object.get());
                json obj = context.get(var->name);
                if (!obj.is_null()) {
                    obj[attr->name] = val;
                    context.update(var->name, obj);
                }
            }
        }
        // If target is tuple, not supported yet.
    }
};

struct ForStmt : Node {
    std::vector<std::string> loop_vars;
    std::unique_ptr<Expr> iterable;
    std::vector<std::unique_ptr<Node>> body;
    std::unique_ptr<Expr> filter_expr;

    ForStmt(std::vector<std::string> vars, std::unique_ptr<Expr> iter, std::vector<std::unique_ptr<Node>> b, std::unique_ptr<Expr> filter = nullptr)
        : loop_vars(std::move(vars)), iterable(std::move(iter)), body(std::move(b)), filter_expr(std::move(filter)) {}

    void render(Context& context, std::string& out) override {
        JINJA_LOG("Render For: Start loop processing");
        json iter_val = iterable->evaluate(context);
        if (is_undefined(iter_val)) {
            JINJA_LOG("Render For: Iterable is undefined, skipping.");
            return;
        }

        std::vector<json> items;

        if (iter_val.is_array()) {
            for (const auto& item : iter_val) items.push_back(item);
        } else if (iter_val.is_object()) {
            // Iterate keys
            std::vector<std::string> keys;
            for (json::iterator it = iter_val.begin(); it != iter_val.end(); ++it) {
                keys.push_back(it.key());
            }
            // Sort keys to be deterministic/consistent with map behavior
            std::sort(keys.begin(), keys.end());
            for (const auto& key : keys) items.push_back(key);
        }

        size_t len = items.size();
        size_t index = 0; // Index of *accepted* items

        // Pre-calculate acceptable items to correctly set loop.length, loop.last etc?
        // Jinja2 loop variables (index, first, last) refer to the ITERATION, but if filtered,
        // they usually refer to the filtered set?
        // Actually in Jinja2: "The loop variable ... is available inside the loop."
        // If 'if' is used: "{% for user in users if user.active %}"
        // The loop object tracks the *filtered* iterations.

        // So we should filter FIRST.
        std::vector<json> filtered_items;
        for (const auto& item : items) {
             // Create temporary scope to evaluate filter
             json temp_scope;
             if (loop_vars.size() == 1) {
                temp_scope[loop_vars[0]] = item;
             } else {
                 if (item.is_array() && item.size() >= loop_vars.size()) {
                     for (size_t i = 0; i < loop_vars.size(); ++i) temp_scope[loop_vars[i]] = item[i];
                 } else {
                     // fallback
                     if (loop_vars.size() > 0) temp_scope[loop_vars[0]] = item;
                 }
             }

             if (filter_expr) {
                 context.push_scope(temp_scope);
                 json res = filter_expr->evaluate(context);
                 context.pop_scope();
                 if (!is_truthy(res)) continue;
             }
             filtered_items.push_back(item);
        }

        len = filtered_items.size();
        index = 0;
        JINJA_LOG("Render For: Iterating " << len << " items.");

        for (const auto& item : filtered_items) {
             json loop_scope;
             if (loop_vars.size() == 1) {
                loop_scope[loop_vars[0]] = item;
             } else {
                 if (item.is_array() && item.size() >= loop_vars.size()) {
                     for (size_t i = 0; i < loop_vars.size(); ++i) loop_scope[loop_vars[i]] = item[i];
                 } else {
                     if (loop_vars.size() > 0) loop_scope[loop_vars[0]] = item;
                 }
             }

             json loop_obj;
             loop_obj["index0"] = index;
             loop_obj["index"] = index + 1;
             loop_obj["first"] = (index == 0);
             loop_obj["last"] = (index == len - 1);
             loop_obj["length"] = len;
             loop_scope["loop"] = loop_obj;

             context.push_scope(std::move(loop_scope));
             for (const auto& node : body) node->render(context, out);
             context.pop_scope();
             index++;
        }
    }
};

struct IfNode : Node {
    std::unique_ptr<Expr> condition;
    std::vector<std::unique_ptr<Node>> true_body;
    std::vector<std::unique_ptr<Node>> false_body;

    IfNode(std::unique_ptr<Expr> cond, std::vector<std::unique_ptr<Node>> tb, std::vector<std::unique_ptr<Node>> fb = {})
        : condition(std::move(cond)), true_body(std::move(tb)), false_body(std::move(fb)) {
        }

    void render(Context& context, std::string& out) override {
        bool res = is_truthy(condition->evaluate(context));
        JINJA_LOG("Render If: (" << condition->dump() << ") evaluated to " << (res ? "TRUE" : "FALSE"));
        if (res) {
             for (const auto& node : true_body) node->render(context, out);
        } else {
             for (const auto& node : false_body) node->render(context, out);
        }
    }
};


// --- Parser ---

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : m_tokens(std::move(tokens)), m_cursor(0) {}

    std::vector<std::unique_ptr<Node>> parse() {
        JINJA_LOG("Parser: Start parsing");
        std::vector<std::unique_ptr<Node>> nodes;
        while (!is_at_end()) {
            if (check(Token::Text)) {
                nodes.push_back(mnn_make_unique<TextNode>(advance().value));
            } else if (check(Token::ExpressionStart)) {
                nodes.push_back(parse_print());
            } else if (check(Token::BlockStart)) {
                nodes.push_back(parse_block());
            } else {
                advance(); // Skip unknown or stray tokens
            }
        }
        return nodes;
    }

private:
   std::unique_ptr<Node> parse_block() {
        advance(); // eat {%

        Token cmd = advance();
        if (cmd.type == Token::Identifier) {
            if (cmd.value == "for") {
                return parse_for();
            } else if (cmd.value == "if") {
                return parse_if();
            } else if (cmd.value == "set") {
                return parse_set();
            } else if (cmd.value == "macro") {
                return parse_macro();
            }
        }

        // Unknown block, skip until end
        while (!check(Token::BlockEnd) && !is_at_end()) {
            advance();
        }
        if (!is_at_end()) advance(); // eat %}
        return mnn_make_unique<TextNode>("");
    }

    std::unique_ptr<Node> parse_macro() {
        std::string name = advance().value;
        std::vector<std::string> args;
        if (check(Token::Punctuation) && peek().value == "(") {
            advance();
            while (!check(Token::Punctuation) || peek().value != ")") {
                if (check(Token::Identifier)) {
                    args.push_back(advance().value);
                    // Handle default values (ignore expression for now)
                    if (check(Token::Identifier) && m_tokens.size() > m_cursor+1 && m_tokens[m_cursor+1].value == "=") {
                         // kwarg? Lexer might split = ?
                         // Actually Lexer returns Token::Operator for = ?
                         // Let's check operator.
                    }
                    if (check(Token::Operator) && peek().value == "=") {
                         advance();
                         parse_expression();
                    }
                }
                if (check(Token::Punctuation) && peek().value == ",") advance();
                else break;
            }
            if (check(Token::Punctuation) && peek().value == ")") advance();
        }
        if (check(Token::BlockEnd)) advance(); // %}

        std::vector<std::unique_ptr<Node>> body;
        while (!is_at_end()) {
             if (check(Token::BlockStart)) {
                 size_t look = m_cursor + 1;
                 if (look < m_tokens.size() && m_tokens[look].value == "endmacro") {
                     advance(); // {%
                     advance(); // endmacro
                     if (check(Token::BlockEnd)) advance(); // %}
                     break;
                 }
             }
             if (check(Token::Text)) body.push_back(mnn_make_unique<TextNode>(advance().value));
             else if (check(Token::ExpressionStart)) body.push_back(parse_print());
             else if (check(Token::BlockStart)) body.push_back(parse_block());
             else advance();
        }
        return mnn_make_unique<MacroNode>(std::move(name), std::move(args), std::move(body));
    }

    std::unique_ptr<Node> parse_set() {
        // We consumed {% and 'set'
        std::unique_ptr<Expr> target = parse_expression();

        if (check(Token::Operator) && peek().value == "=") {
            advance();
        } else {
            // Error expect =
        }

        std::unique_ptr<Expr> value = parse_expression();

        if (check(Token::BlockEnd)) advance(); // eat %}

        return mnn_make_unique<SetNode>(std::move(target), std::move(value));
    }

    std::unique_ptr<Node> parse_if() {
        JINJA_LOG("Parser: Parsing IF block");
        // We consumed {% and 'if'
        std::unique_ptr<Expr> condition = parse_expression();
        if (check(Token::BlockEnd)) advance(); // eat %}

        std::vector<std::unique_ptr<Node>> true_body;
        std::vector<std::unique_ptr<Node>> false_body;
        std::vector<std::unique_ptr<Node>>* current_body = &true_body;

        while (!is_at_end()) {
             if (check(Token::BlockStart)) {
                 // Check for endif or else
                 // Need to peek ahead to see if it's endif/else/elif
                 // But parse_block logic usually consumes {%...
                 // Here we are inside the body loop.

                 // Lookahead:
                 size_t look = m_cursor;
                 if (m_tokens[look].type == Token::BlockStart) {
                     look++;
                     if (look < m_tokens.size() && m_tokens[look].type == Token::Identifier) {
                         std::string tag = m_tokens[look].value;
                         if (tag == "endif") {
                             advance(); advance();
                             if (check(Token::BlockEnd)) advance();
                             break;
                         } else if (tag == "else") {
                             advance(); advance();
                             if (check(Token::BlockEnd)) advance();
                             current_body = &false_body;
                             continue;
                         } else if (tag == "elif") {
                              // Elif is tricky: needs to close current if true_body, and start a new if in false_body?
                              // Or treat as Else { If ... }
                              // Simplest: treat as else block containing a single IfNode.
                              advance(); advance(); // eat {% elif
                              std::unique_ptr<Expr> elif_cond = parse_expression();
                              if (check(Token::BlockEnd)) advance();

                              // Recursion again
                              current_body = &false_body;
                              current_body->push_back(parse_if_from_elif(std::move(elif_cond)));
                              break; // The recursive call consumes everything up to matching endif
                         }
                     }
                 }
             }

             if (check(Token::Text)) {
                current_body->push_back(mnn_make_unique<TextNode>(advance().value));
            } else if (check(Token::ExpressionStart)) {
                current_body->push_back(parse_print());
            } else if (check(Token::BlockStart)) {
                 current_body->push_back(parse_block());
            } else {
                 advance();
            }
        }

        return mnn_make_unique<IfNode>(std::move(condition), std::move(true_body), std::move(false_body));
    }

    std::unique_ptr<Node> parse_if_from_elif(std::unique_ptr<Expr> condition) {
         // Similar to parse_if but condition is passed in.
         std::vector<std::unique_ptr<Node>> true_body;
         std::vector<std::unique_ptr<Node>> false_body;
         std::vector<std::unique_ptr<Node>>* current_body = &true_body;

         while (!is_at_end()) {
             if (check(Token::BlockStart)) {
                 size_t look = m_cursor;
                 if (m_tokens[look].type == Token::BlockStart) {
                     look++;
                     if (look < m_tokens.size() && m_tokens[look].type == Token::Identifier) {
                         std::string tag = m_tokens[look].value;
                         if (tag == "endif") {
                             advance(); // eat {%
                             advance(); // eat endif
                             if (check(Token::BlockEnd)) advance(); // eat %}
                             break;
                         } else if (tag == "else") {
                             advance(); advance();
                             if (check(Token::BlockEnd)) advance();
                             current_body = &false_body;
                             continue;
                         } else if (tag == "elif") {
                             advance(); advance();
                             std::unique_ptr<Expr> elif_cond = parse_expression();
                             if (check(Token::BlockEnd)) advance();

                             // Recursion again
                             current_body = &false_body;
                             current_body->push_back(parse_if_from_elif(std::move(elif_cond)));
                             break;
                         }
                     }
                 }
             }

             if (check(Token::Text)) {
                current_body->push_back(mnn_make_unique<TextNode>(advance().value));
            } else if (check(Token::ExpressionStart)) {
                current_body->push_back(parse_print());
            } else if (check(Token::BlockStart)) {
                 current_body->push_back(parse_block());
            } else {
                 advance();
            }
         }
         return mnn_make_unique<IfNode>(std::move(condition), std::move(true_body), std::move(false_body));
    }

    std::unique_ptr<Node> parse_for() {
        // We already consumed {% and 'for'
        std::vector<std::string> vars;
        while (check(Token::Identifier)) {
            vars.push_back(advance().value);
            if (check(Token::Punctuation) && peek().value == ",") {
                advance();
            } else {
                break;
            }
        }

        // Expect 'in'
        if (check(Token::Identifier) && peek().value == "in") {
            advance();
        }

        // Parse iterable. Disable ternary parsing to avoid consuming 'if' filter.
        std::unique_ptr<Expr> iterable = parse_expression(false);
        std::unique_ptr<Expr> filter_expr = nullptr;

        // Check for 'if' filter
        if (check(Token::Identifier) && peek().value == "if") {
            advance(); // eat 'if'
            filter_expr = parse_expression();
        }

        if (check(Token::BlockEnd)) {
            advance(); // eat %}
        }

        std::vector<std::unique_ptr<Node>> body;
        while (!is_at_end()) {
            if (check(Token::BlockStart)) {
                 if (m_cursor + 2 < m_tokens.size() &&
                     m_tokens[m_cursor+1].type == Token::Identifier &&
                     m_tokens[m_cursor+1].value == "endfor") {
                         advance(); // eat {%
                         advance(); // eat endfor
                         if (check(Token::BlockEnd)) advance(); // eat %}
                         break;
                 }
            }
            if (check(Token::Text)) {
                body.push_back(mnn_make_unique<TextNode>(advance().value));
            } else if (check(Token::ExpressionStart)) {
                body.push_back(parse_print());
            } else if (check(Token::BlockStart)) {
                 body.push_back(parse_block());
            } else {
                 advance();
            }
        }

        return mnn_make_unique<ForStmt>(std::move(vars), std::move(iterable), std::move(body), std::move(filter_expr));
    }

    std::unique_ptr<Node> parse_print() {
        advance(); // eat {{
        std::unique_ptr<Expr> expr = parse_expression();
        if (check(Token::ExpressionEnd)) {
            advance(); // eat }}
        }
        return mnn_make_unique<PrintNode>(std::move(expr));
    }

    // Expression Parsing with Precedence
    // logic > not > compare > add > filter

    std::unique_ptr<Expr> parse_expression(bool allow_ternary = true) {
        auto expr = parse_or();
        if (allow_ternary && check(Token::Identifier) && peek().value == "if") {
            advance(); // if
            auto cond = parse_or();
            if (check(Token::Identifier) && peek().value == "else") {
                advance(); // else
                auto else_expr = parse_expression();
                return mnn_make_unique<TernaryExpr>(std::move(cond), std::move(expr), std::move(else_expr));
            } else {
                 // Syntax error: expected else
            }
        }
        return expr;
    }

    std::unique_ptr<Expr> parse_or() {
        auto left = parse_and();
        while (check(Token::Identifier) && peek().value == "or") {
            advance();
            auto right = parse_and();
            left = mnn_make_unique<BinaryExpr>("or", std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<Expr> parse_and() {
        auto left = parse_not();
        while (check(Token::Identifier) && peek().value == "and") {
            advance();
            auto right = parse_not(); // logic op precedence
            left = mnn_make_unique<BinaryExpr>("and", std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<Expr> parse_not() {
        if (check(Token::Identifier) && peek().value == "not") {
            advance();
            // (not Expr) -> (Expr ? false : true)。
            auto expr = parse_not();
            return mnn_make_unique<TernaryExpr>(
                std::move(expr),
                mnn_make_unique<LiteralExpr>(false),
                mnn_make_unique<LiteralExpr>(true)
            );
         }
         return parse_compare();
    }

    std::unique_ptr<Expr> parse_compare() {
        auto left = parse_add();
        while ((check(Token::Operator) && (peek().value == "==" || peek().value == "!=" || peek().value == "<" || peek().value == ">" || peek().value == "<=" || peek().value == ">=")) ||
               (check(Token::Identifier) && (peek().value == "in" || peek().value == "is" || peek().value == "not"))) {
            std::string op = advance().value;
            if (op == "is") {
                bool is_not = false;
                if (check(Token::Identifier) && peek().value == "not") {
                    advance();
                    is_not = true;
                }
                std::string test_name = advance().value;
                left = mnn_make_unique<TestExpr>(std::move(left), test_name, is_not);
            } else if (op == "not") {
                if (check(Token::Identifier) && peek().value == "in") {
                    advance(); // eat in
                    auto right = parse_add();
                    left = mnn_make_unique<BinaryExpr>("not in", std::move(left), std::move(right));
                } else {
                     // Error or treat as unary not binding weirdly?
                     // Assuming 'not in' is the only valid infix 'not'.
                     break;
                }
            } else {
                auto right = parse_add();
                left = mnn_make_unique<BinaryExpr>(op, std::move(left), std::move(right));
            }
        }
        return left;
    }

    std::unique_ptr<Expr> parse_add() {
        auto left = parse_filter();
        while (check(Token::Operator) && (peek().value == "+" || peek().value == "-" || peek().value == "~")) {
            std::string op = advance().value;
            auto right = parse_filter();
            left = mnn_make_unique<BinaryExpr>(op, std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<Expr> parse_filter() {
        auto left = parse_unary();
        while (check(Token::Operator) && peek().value == "|") {
            advance(); // eat |
            if (check(Token::Identifier)) {
                std::string name = advance().value;
                std::vector<std::pair<std::string, std::unique_ptr<Expr>>> args;

                if (check(Token::Punctuation) && peek().value == "(") {
                    advance(); // eat (
                    while (!check(Token::Punctuation) || peek().value != ")") {
                        std::string arg_name;
                        std::unique_ptr<Expr> val;
                        // Peek for kwarg: identifier =
                        if (check(Token::Identifier) && m_tokens.size() > m_cursor+1 && m_tokens[m_cursor+1].value == "=") {
                            arg_name = advance().value;
                            advance(); // =
                            val = parse_expression();
                        } else {
                            val = parse_expression();
                        }
                        args.emplace_back(arg_name, std::move(val));
                        if (check(Token::Punctuation) && peek().value == ",") advance();
                        else break;
                    }
                    if (check(Token::Punctuation) && peek().value == ")") advance();
                }
                left = mnn_make_unique<FilterExpr>(std::move(left), name, std::move(args));
            }
        }
        return left;
    }

    std::unique_ptr<Expr> parse_unary() {
        if (check(Token::Operator) && (peek().value == "+" || peek().value == "-")) {
            std::string op = advance().value;
            auto right = parse_unary();
             // Treat as 0 + val or 0 - val
             return mnn_make_unique<BinaryExpr>(op, mnn_make_unique<LiteralExpr>(0), std::move(right));
        }
        return parse_primary();
    }

    std::unique_ptr<Expr> parse_primary() {
        if (check(Token::Punctuation) && peek().value == "[") {
             advance(); // [
             std::vector<std::unique_ptr<Expr>> items;
             while (!check(Token::Punctuation) || peek().value != "]") {
                 items.push_back(parse_expression());
                 if (check(Token::Punctuation) && peek().value == ",") advance();
             }
             if (check(Token::Punctuation) && peek().value == "]") advance();
             return mnn_make_unique<ListExpr>(std::move(items));
        }

        if (check(Token::Punctuation) && peek().value == "{") {
             advance(); // {
             std::vector<std::pair<std::unique_ptr<Expr>, std::unique_ptr<Expr>>> items;
             while (!check(Token::Punctuation) || peek().value != "}") {
                 auto key = parse_expression();
                 if (check(Token::Punctuation) && peek().value == ":") advance();
                 auto val = parse_expression();
                 items.emplace_back(std::move(key), std::move(val));
                 if (check(Token::Punctuation) && peek().value == ",") advance();
             }
             if (check(Token::Punctuation) && peek().value == "}") advance();
             return mnn_make_unique<ObjectExpr>(std::move(items));
        }

        if (check(Token::String)) {
            return mnn_make_unique<LiteralExpr>(advance().value);
        }
        if (check(Token::Identifier)) {
            std::string name = advance().value;
             if (name == "true") return mnn_make_unique<LiteralExpr>(true);
             if (name == "false") return mnn_make_unique<LiteralExpr>(false);
             if (name == "none") return mnn_make_unique<LiteralExpr>(nullptr);

            std::unique_ptr<Expr> expr;

            // Check for Call (
            if (check(Token::Punctuation) && peek().value == "(") {
                advance(); // eat (
                std::vector<std::pair<std::string, std::unique_ptr<Expr>>> args;
                while (!check(Token::Punctuation) || peek().value != ")") {
                    // Argument parsing: name=val or val
                    std::string arg_name;
                    std::unique_ptr<Expr> val;

                    // Peek for identifier =
                    if (check(Token::Identifier) && m_tokens.size() > m_cursor+1 && m_tokens[m_cursor+1].value == "=") {
                        arg_name = advance().value;
                        advance(); // eat =
                        val = parse_expression();
                    } else {
                        val = parse_expression();
                    }
                    args.emplace_back(arg_name, std::move(val));

                    if (check(Token::Punctuation) && peek().value == ",") advance();
                    else break;
                }
                if (check(Token::Punctuation) && peek().value == ")") advance();
                expr = mnn_make_unique<CallExpr>(name, std::move(args));
            } else {
                expr = mnn_make_unique<VarExpr>(name);
            }

            while (true) { // Chain of accesses
                if (check(Token::Punctuation) && peek().value == "[") {
                     advance(); // [
                     // Check for slice
                     std::unique_ptr<Expr> start, stop, step;
                     bool is_slice = false;

                     // Parse start
                     if (!check(Token::Punctuation) || peek().value != ":" ) {
                         if (!check(Token::Punctuation) || peek().value != "]") { // [expr] or [expr:]
                             start = parse_expression();
                         }
                     }

                     if (check(Token::Punctuation) && peek().value == ":") {
                         is_slice = true;
                         advance(); // eat :
                         // Parse stop
                         if (!check(Token::Punctuation) || (peek().value != ":" && peek().value != "]")) {
                              stop = parse_expression();
                         }

                         if (check(Token::Punctuation) && peek().value == ":") {
                             advance(); // eat second :
                             if (!check(Token::Punctuation) || peek().value != "]") {
                                 step = parse_expression();
                             }
                         }
                     }

                     if (check(Token::Punctuation) && peek().value == "]") advance();

                     if (is_slice) {
                         auto slice_expr = mnn_make_unique<SliceExpr>(std::move(start), std::move(stop), std::move(step));
                         expr = mnn_make_unique<GetItemExpr>(std::move(expr), std::move(slice_expr));
                     } else {
                         // Normal index
                         // if start is null effectively (e.g. []) - invalid in Jinja usually but maybe empty key used?
                         if (!start) start = mnn_make_unique<LiteralExpr>("");
                         expr = mnn_make_unique<GetItemExpr>(std::move(expr), std::move(start));
                     }
                } else if (check(Token::Punctuation) && peek().value == ".") {
                    advance(); // .
                    if (check(Token::Identifier)) {
                        std::string member = advance().value;
                        if (check(Token::Punctuation) && peek().value == "(") {
                             // Method call
                             advance(); // (
                             std::vector<std::unique_ptr<Expr>> method_args;
                             while (!check(Token::Punctuation) || peek().value != ")") {
                                 method_args.push_back(parse_expression());
                                 if (check(Token::Punctuation) && peek().value == ",") advance();
                                 else break;
                             }
                             if (check(Token::Punctuation) && peek().value == ")") advance();
                             expr = mnn_make_unique<MethodCallExpr>(std::move(expr), member, std::move(method_args));
                        } else {
                            expr = mnn_make_unique<GetAttrExpr>(std::move(expr), member);
                        }
                    }
                } else {
                    break;
                }
            }
            return expr;
        }
        if (check(Token::Number)) {
            std::string val = advance().value;
             {
                char* end = nullptr;
                if (val.find('.') != std::string::npos) {
                    float f = std::strtof(val.c_str(), &end);
                    if (end && end != val.c_str()) return mnn_make_unique<LiteralExpr>(f);
                } else {
                    long l = std::strtol(val.c_str(), &end, 10);
                    if (end && end != val.c_str()) return mnn_make_unique<LiteralExpr>((int)l);
                }
                return mnn_make_unique<LiteralExpr>(0);
             }
        }
        if (check(Token::Punctuation) && peek().value == "(") {
            advance();
            auto expr = parse_expression();
            if (check(Token::Punctuation) && peek().value == ")") advance();
            return expr;
        }
        return mnn_make_unique<LiteralExpr>("");
    }

    // Helper methods
    const Token& peek() const { return m_tokens[m_cursor]; }
    const Token& previous() const { return m_tokens[m_cursor - 1]; }
    bool is_at_end() const { return peek().type == Token::Eof; }

    bool check(Token::Type type) const {
        if (is_at_end()) return false;
        return peek().type == type;
    }

    Token advance() {
        if (!is_at_end()) m_cursor++;
        return previous();
    }

    bool match(Token::Type type) {
        if (check(type)) {
            advance();
            return true;
        }
        return false;
    }

    std::vector<Token> m_tokens;
    size_t m_cursor;
};

struct Template::Impl {
    std::string template_str;
    std::vector<std::unique_ptr<Node>> root_nodes;
    json default_context; // Changed from global_vars to default_context
    std::map<std::string, UserFunction> functions;

    void parse() {
        Lexer lexer(template_str);
        auto tokens = lexer.tokenize();
        Parser parser(std::move(tokens));
        root_nodes = parser.parse();
    }
};

inline Template::Template(const std::string& template_str, const json& default_context)
    : m_impl(mnn_make_unique<Impl>()) {
    m_impl->template_str = template_str;
    m_impl->default_context = default_context;

    register_builtins();

    m_impl->parse();
}

inline void Template::register_builtins() {
    // Built-in: range
    add_function("range", [](const std::vector<Argument>& args) -> json {
         long start=0, stop=0, step=1;
         if (args.size() == 1) {
             stop = args[0].second.get<long>();
         } else if (args.size() == 2) {
             start = args[0].second.get<long>();
             stop = args[1].second.get<long>();
         } else if (args.size() >= 3) {
             start = args[0].second.get<long>();
             stop = args[1].second.get<long>();
             step = args[2].second.get<long>();
         }
         json arr = json::array();
         if (step > 0) {
             for (long i = start; i < stop; i += step) arr.push_back(i);
         } else if (step < 0) {
             for (long i = start; i > stop; i += step) arr.push_back(i);
         }
         return arr;
    });

    // Built-in: namespace
    add_function("namespace", [](const std::vector<Argument>& args) -> json {
        json ns = json::object();
        for (const auto& arg : args) {
            if (!arg.first.empty()) {
                ns[arg.first] = arg.second;
            }
        }
        return ns;
    });

    // Built-in: strftime_now
    add_function("strftime_now", [](const std::vector<Argument>& args) -> json {
        std::string format = "%Y-%m-%d";
        if (args.size() > 0) {
            format = args[0].second.get<std::string>();
        }
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        std::stringstream ss;
        ss << std::put_time(&tm, format.c_str());
        return ss.str();
    });
}

inline Template::~Template() = default;

inline Template::Template(Template&& other) noexcept = default;
inline Template& Template::operator=(Template&& other) noexcept = default;

inline std::string Template::render(const json& context) const {
    Context ctx(m_impl->default_context);
    ctx.set_functions(&m_impl->functions);
    if (!context.empty()) {
        ctx.push_scope(context);
    }
    std::string output;
    for (const auto& node : m_impl->root_nodes) {
        node->render(ctx, output);
    }
    return output;
}

inline void Template::add_function(const std::string& name, UserFunction func) {
    m_impl->functions[name] = std::move(func);
    if (!m_impl->default_context.contains(name)) {
        m_impl->default_context[name] = "<function " + name + ">";
    }
}

inline std::string Template::apply_chat_template(
    const json& messages,
    bool add_generation_prompt,
    const json& tools,
    const json& extra_context
) const {
    // Detect if template requires typed content (array of {type,text} parts)
    // by trial-rendering with string vs array content and checking output.
    bool requires_typed_content = false;
    {
        const std::string needle = "<__jinja_content_probe__>";
        json str_msg = json::array();
        {
            json m = json::object();
            m["role"] = "user";
            m["content"] = needle;
            str_msg.push_back(m);
        }
        json typed_msg = json::array();
        {
            json m = json::object();
            m["role"] = "user";
            json arr = json::array();
            json part = json::object();
            part["type"] = "text";
            part["text"] = needle;
            arr.push_back(part);
            m["content"] = arr;
            typed_msg.push_back(m);
        }
        auto try_render = [&](const json& msgs) -> std::string {
            json ctx = json::object();
            ctx["messages"] = msgs;
            return render(ctx);
        };
        auto str_result = try_render(str_msg);
        auto typed_result = try_render(typed_msg);
        requires_typed_content = (str_result.find(needle) == std::string::npos) &&
                                 (typed_result.find(needle) != std::string::npos);
    }

    json msgs = messages;
    if (requires_typed_content) {
        for (size_t i = 0; i < msgs.size(); i++) {
            if (msgs[i].contains("content") && msgs[i]["content"].is_string()) {
                auto text = msgs[i]["content"].get<std::string>();
                if (text.empty()) continue;
                json arr = json::array();
                json part = json::object();
                part["type"] = "text";
                part["text"] = text;
                arr.push_back(part);
                msgs[i]["content"] = arr;
            }
        }
    }

    json context = extra_context;
    context["messages"] = msgs;
    if (!tools.empty()) context["tools"] = tools;
    if (add_generation_prompt) context["add_generation_prompt"] = true;

    return render(context);
}

} // namespace jinja