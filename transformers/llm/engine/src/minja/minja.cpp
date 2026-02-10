/*
    Copyright 2024 Google LLC

    Use of this source code is governed by an MIT-style
    license that can be found in the LICENSE file or at
    https://opensource.org/licenses/MIT.
*/
// SPDX-License-Identifier: MIT
#ifdef LLM_USE_MINJA
#include "minja.hpp"
namespace minja {
enum SpaceHandling { Keep, Strip, StripSpaces, StripNewline };

class TemplateToken {
public:
    enum class Type { Text, Expression, If, Else, Elif, EndIf, For, EndFor, Generation, EndGeneration, Set, EndSet, Comment, Macro, EndMacro, Filter, EndFilter, Break, Continue };

    static std::string typeToString(Type t) {
        switch (t) {
            case Type::Text: return "text";
            case Type::Expression: return "expression";
            case Type::If: return "if";
            case Type::Else: return "else";
            case Type::Elif: return "elif";
            case Type::EndIf: return "endif";
            case Type::For: return "for";
            case Type::EndFor: return "endfor";
            case Type::Set: return "set";
            case Type::EndSet: return "endset";
            case Type::Comment: return "comment";
            case Type::Macro: return "macro";
            case Type::EndMacro: return "endmacro";
            case Type::Filter: return "filter";
            case Type::EndFilter: return "endfilter";
            case Type::Generation: return "generation";
            case Type::EndGeneration: return "endgeneration";
            case Type::Break: return "break";
            case Type::Continue: return "continue";
        }
        return "Unknown";
    }

    TemplateToken(Type type, const Location & location, SpaceHandling pre, SpaceHandling post) : type(type), location(location), pre_space(pre), post_space(post) {}
    virtual ~TemplateToken() = default;

    Type type;
    Location location;
    SpaceHandling pre_space = SpaceHandling::Keep;
    SpaceHandling post_space = SpaceHandling::Keep;
};

struct TextTemplateToken : public TemplateToken {
    std::string text;
    TextTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, const std::string& t) : TemplateToken(Type::Text, loc, pre, post), text(t) {}
};

struct ExpressionTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> expr;
    ExpressionTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && e) : TemplateToken(Type::Expression, loc, pre, post), expr(std::move(e)) {}
};

struct IfTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> condition;
    IfTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && c) : TemplateToken(Type::If, loc, pre, post), condition(std::move(c)) {}
};

struct ElifTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> condition;
    ElifTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && c) : TemplateToken(Type::Elif, loc, pre, post), condition(std::move(c)) {}
};

struct ElseTemplateToken : public TemplateToken {
    ElseTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::Else, loc, pre, post) {}
};

struct EndIfTemplateToken : public TemplateToken {
    EndIfTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndIf, loc, pre, post) {}
};

struct MacroTemplateToken : public TemplateToken {
    std::shared_ptr<VariableExpr> name;
    Expression::Parameters params;
    MacroTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<VariableExpr> && n, Expression::Parameters && p)
      : TemplateToken(Type::Macro, loc, pre, post), name(std::move(n)), params(std::move(p)) {}
};

struct EndMacroTemplateToken : public TemplateToken {
    EndMacroTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndMacro, loc, pre, post) {}
};

struct FilterTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> filter;
    FilterTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && filter)
      : TemplateToken(Type::Filter, loc, pre, post), filter(std::move(filter)) {}
};

struct EndFilterTemplateToken : public TemplateToken {
    EndFilterTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndFilter, loc, pre, post) {}
};

struct ForTemplateToken : public TemplateToken {
    std::vector<std::string> var_names;
    std::shared_ptr<Expression> iterable;
    std::shared_ptr<Expression> condition;
    bool recursive;
    ForTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, const std::vector<std::string> & vns, std::shared_ptr<Expression> && iter,
      std::shared_ptr<Expression> && c, bool r)
      : TemplateToken(Type::For, loc, pre, post), var_names(vns), iterable(std::move(iter)), condition(std::move(c)), recursive(r) {}
};

struct EndForTemplateToken : public TemplateToken {
    EndForTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndFor, loc, pre, post) {}
};

struct GenerationTemplateToken : public TemplateToken {
    GenerationTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::Generation, loc, pre, post) {}
};

struct EndGenerationTemplateToken : public TemplateToken {
    EndGenerationTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndGeneration, loc, pre, post) {}
};

struct SetTemplateToken : public TemplateToken {
    std::string ns;
    std::vector<std::string> var_names;
    std::shared_ptr<Expression> value;
    SetTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, const std::string & ns, const std::vector<std::string> & vns, std::shared_ptr<Expression> && v)
      : TemplateToken(Type::Set, loc, pre, post), ns(ns), var_names(vns), value(std::move(v)) {}
};

struct EndSetTemplateToken : public TemplateToken {
    EndSetTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndSet, loc, pre, post) {}
};

struct CommentTemplateToken : public TemplateToken {
    std::string text;
    CommentTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, const std::string& t) : TemplateToken(Type::Comment, loc, pre, post), text(t) {}
};



struct LoopControlTemplateToken : public TemplateToken {
    LoopControlType control_type;
    LoopControlTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, LoopControlType control_type) : TemplateToken(Type::Break, loc, pre, post), control_type(control_type) {}
};
    class Parser {
        private:
            using CharIterator = std::string::const_iterator;

            std::shared_ptr<std::string> template_str;
            CharIterator start, end, it;
            Options options;

            Parser(const std::shared_ptr<std::string>& template_str, const Options & options) : template_str(template_str), options(options) {
                if (!template_str) _printlog("Template string is null");
                start = it = this->template_str->begin();
                end = this->template_str->end();
            }

            bool consumeSpaces(SpaceHandling space_handling = SpaceHandling::Strip) {
                if (space_handling == SpaceHandling::Strip) {
                    while (it != end && std::isspace(*it)) ++it;
                }
                return true;
            }

            std::shared_ptr<std::string> parseString() {
                auto doParse = [&](char quote) -> std::shared_ptr<std::string> {
                    if (it == end || *it != quote) return nullptr;
                    std::string result;
                    bool escape = false;
                    for (++it; it != end; ++it) {
                        if (escape) {
                            escape = false;
                            switch (*it) {
                                case 'n': result += '\n'; break;
                                case 'r': result += '\r'; break;
                                case 't': result += '\t'; break;
                                case 'b': result += '\b'; break;
                                case 'f': result += '\f'; break;
                                case '\\': result += '\\'; break;
                                default:
                                           if (*it == quote) {
                                               result += quote;
                                           } else {
                                               result += *it;
                                           }
                                           break;
                            }
                        } else if (*it == '\\') {
                            escape = true;
                        } else if (*it == quote) {
                            ++it;
                            std::shared_ptr<std::string> res(new std::string);
                            *res = result;
                            return res;
                        } else {
                            result += *it;
                        }
                    }
                    return nullptr;
                };

                consumeSpaces();
                if (it == end) return nullptr;
                if (*it == '"') return doParse('"');
                if (*it == '\'') return doParse('\'');
                return nullptr;
            }

            json parseNumber(CharIterator& it, const CharIterator& end) {
                auto before = it;
                consumeSpaces();
                auto start = it;
                bool hasDecimal = false;
                bool hasExponent = false;

                if (it != end && (*it == '-' || *it == '+')) ++it;

                while (it != end) {
                    if (std::isdigit(*it)) {
                        ++it;
                    } else if (*it == '.') {
                        if (hasDecimal) {
                            _printlog("Multiple decimal points");
                            return json();
                        }
                        hasDecimal = true;
                        ++it;
                    } else if (it != start && (*it == 'e' || *it == 'E')) {
                        if (hasExponent) {
                            _printlog("Multiple exponents");
                            return json();
                        }
                        hasExponent = true;
                        ++it;
                    } else {
                        break;
                    }
                }
                if (start == it) {
                    it = before;
                    return json(); // No valid characters found
                }
                std::string str(start, it);
                if (hasExponent || hasDecimal) {
                    double v = std::stof(str);
                    return json(v);
                }
                int64_t v = std::stoi(str);
                return json(v);
            }

            /** integer, float, bool, string */
            std::shared_ptr<Value> parseConstant() {
                auto start = it;
                consumeSpaces();
                if (it == end) return nullptr;
                if (*it == '"' || *it == '\'') {
                    auto str = parseString();
                    if (str) return std::make_shared<Value>(*str);
                }
                static std::regex prim_tok(R"(true\b|True\b|false\b|False\b|None\b)");
                auto token = consumeToken(prim_tok);
                if (!token.empty()) {
                    if (token == "true" || token == "True") return std::make_shared<Value>(true);
                    if (token == "false" || token == "False") return std::make_shared<Value>(false);
                    if (token == "None") return std::make_shared<Value>(nullptr);
                    _printlog("Unknown constant token: " + token);
                }

                auto number = parseNumber(it, end);
                if (!number.is_null()) return std::make_shared<Value>(number);

                it = start;
                return nullptr;
            }

            bool peekSymbols(const std::vector<std::string> & symbols) const {
                for (const auto & symbol : symbols) {
                    if (std::distance(it, end) >= (int64_t) symbol.size() && std::string(it, it + symbol.size()) == symbol) {
                        return true;
                    }
                }
                return false;
            }

            std::vector<std::string> consumeTokenGroups(const std::regex & regex, SpaceHandling space_handling = SpaceHandling::Strip) {
                auto start = it;
                consumeSpaces(space_handling);
                std::smatch match;
                if (std::regex_search(it, end, match, regex) && match.position() == 0) {
                    it += match[0].length();
                    std::vector<std::string> ret;
                    for (size_t i = 0, n = match.size(); i < n; ++i) {
                        ret.push_back(match[i].str());
                    }
                    return ret;
                }
                it = start;
                return {};
            }
            std::string consumeToken(const std::regex & regex, SpaceHandling space_handling = SpaceHandling::Strip) {
                auto start = it;
                consumeSpaces(space_handling);
                std::smatch match;
                if (std::regex_search(it, end, match, regex) && match.position() == 0) {
                    it += match[0].length();
                    return match[0].str();
                }
                it = start;
                return "";
            }

            std::string consumeToken(const std::string & token, SpaceHandling space_handling = SpaceHandling::Strip) {
                auto start = it;
                consumeSpaces(space_handling);
                if (std::distance(it, end) >= (int64_t) token.size() && std::string(it, it + token.size()) == token) {
                    it += token.size();
                    return token;
                }
                it = start;
                return "";
            }

            std::shared_ptr<Expression> parseExpression(bool allow_if_expr = true) {
                auto left = parseLogicalOr();
                if (it == end) return left;

                if (!allow_if_expr) return left;

                static std::regex if_tok(R"(if\b)");
                if (consumeToken(if_tok).empty()) {
                    return left;
                }

                auto location = get_location();
                auto cepair = parseIfExpression();
                auto condition = cepair.first;
                auto else_expr = cepair.second;
                return std::make_shared<IfExpr>(location, std::move(condition), std::move(left), std::move(else_expr));
            }

            Location get_location() const {
                return {template_str, (size_t) std::distance(start, it)};
            }

            std::pair<std::shared_ptr<Expression>, std::shared_ptr<Expression>> parseIfExpression() {
                auto condition = parseLogicalOr();
                if (!condition) _printlog("Expected condition expression");

                static std::regex else_tok(R"(else\b)");
                std::shared_ptr<Expression> else_expr;
                if (!consumeToken(else_tok).empty()) {
                    else_expr = parseExpression();
                    if (!else_expr) _printlog("Expected 'else' expression");
                }
                return std::make_pair(std::move(condition), std::move(else_expr));
            }

            std::shared_ptr<Expression> parseLogicalOr() {
                auto left = parseLogicalAnd();
                if (!left) _printlog("Expected left side of 'logical or' expression");

                static std::regex or_tok(R"(or\b)");
                auto location = get_location();
                while (!consumeToken(or_tok).empty()) {
                    auto right = parseLogicalAnd();
                    if (!right) _printlog("Expected right side of 'or' expression");
                    left = std::make_shared<BinaryOpExpr>(location, std::move(left), std::move(right), BinaryOpExpr::Op::Or);
                }
                return left;
            }

            std::shared_ptr<Expression> parseLogicalNot() {
                static std::regex not_tok(R"(not\b)");
                auto location = get_location();

                if (!consumeToken(not_tok).empty()) {
                    auto sub = parseLogicalNot();
                    if (!sub) _printlog("Expected expression after 'not' keyword");
                    return std::make_shared<UnaryOpExpr>(location, std::move(sub), UnaryOpExpr::Op::LogicalNot);
                }
                return parseLogicalCompare();
            }

            std::shared_ptr<Expression> parseLogicalAnd() {
                auto left = parseLogicalNot();
                if (!left) _printlog("Expected left side of 'logical and' expression");

                static std::regex and_tok(R"(and\b)");
                auto location = get_location();
                while (!consumeToken(and_tok).empty()) {
                    auto right = parseLogicalNot();
                    if (!right) _printlog("Expected right side of 'and' expression");
                    left = std::make_shared<BinaryOpExpr>(location, std::move(left), std::move(right), BinaryOpExpr::Op::And);
                }
                return left;
            }

            std::shared_ptr<Expression> parseLogicalCompare() {
                auto left = parseStringConcat();
                if (!left) _printlog("Expected left side of 'logical compare' expression");

                static std::regex compare_tok(R"(==|!=|<=?|>=?|in\b|is\b|not\s+in\b)");
                static std::regex not_tok(R"(not\b)");
                std::string op_str;
                while (!(op_str = consumeToken(compare_tok)).empty()) {
                    auto location = get_location();
                    if (op_str == "is") {
                        auto negated = !consumeToken(not_tok).empty();

                        auto identifier = parseIdentifier();
                        if (!identifier) _printlog("Expected identifier after 'is' keyword");

                        return std::make_shared<BinaryOpExpr>(
                                left->location,
                                std::move(left), std::move(identifier),
                                negated ? BinaryOpExpr::Op::IsNot : BinaryOpExpr::Op::Is);
                    }
                    auto right = parseStringConcat();
                    if (!right) _printlog("Expected right side of 'logical compare' expression");
                    BinaryOpExpr::Op op;
                    if (op_str == "==") op = BinaryOpExpr::Op::Eq;
                    else if (op_str == "!=") op = BinaryOpExpr::Op::Ne;
                    else if (op_str == "<") op = BinaryOpExpr::Op::Lt;
                    else if (op_str == ">") op = BinaryOpExpr::Op::Gt;
                    else if (op_str == "<=") op = BinaryOpExpr::Op::Le;
                    else if (op_str == ">=") op = BinaryOpExpr::Op::Ge;
                    else if (op_str == "in") op = BinaryOpExpr::Op::In;
                    else if (op_str.substr(0, 3) == "not") op = BinaryOpExpr::Op::NotIn;
                    else _printlog("Unknown comparison operator: " + op_str);
                    left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), op);
                }
                return left;
            }

            Expression::Parameters parseParameters() {
                consumeSpaces();
                if (consumeToken("(").empty()) _printlog("Expected opening parenthesis in param list");

                Expression::Parameters result;

                while (it != end) {
                    if (!consumeToken(")").empty()) {
                        return result;
                    }
                    auto expr = parseExpression();
                    if (!expr) _printlog("Expected expression in call args");
                    if (expr->mType == Expression::Type_Variable) {
                        auto ident = (VariableExpr*)(expr.get());
                        if (!consumeToken("=").empty()) {
                            auto value = parseExpression();
                            if (!value) _printlog("Expected expression in for named arg");
                            result.emplace_back(ident->get_name(), std::move(value));
                        } else {
                            result.emplace_back(ident->get_name(), nullptr);
                        }
                    } else {
                        result.emplace_back(std::string(), std::move(expr));
                    }
                    if (consumeToken(",").empty()) {
                        if (consumeToken(")").empty()) {
                            _printlog("Expected closing parenthesis in call args");
                        }
                        return result;
                    }
                }
                _printlog("Expected closing parenthesis in call args");
                return result;
            }

            ArgumentsExpression parseCallArgs() {
                consumeSpaces();
                if (consumeToken("(").empty()) _printlog("Expected opening parenthesis in call args");

                ArgumentsExpression result;

                while (it != end) {
                    if (!consumeToken(")").empty()) {
                        return result;
                    }
                    auto expr = parseExpression();
                    if (!expr) _printlog("Expected expression in call args");

                    if (expr->mType == Expression::Type_Variable) {
                        auto ident = (VariableExpr*)(expr.get());
                        if (!consumeToken("=").empty()) {
                            auto value = parseExpression();
                            if (!value) _printlog("Expected expression in for named arg");
                            result.kwargs.emplace_back(ident->get_name(), std::move(value));
                        } else {
                            result.args.emplace_back(std::move(expr));
                        }
                    } else {
                        result.args.emplace_back(std::move(expr));
                    }
                    if (consumeToken(",").empty()) {
                        if (consumeToken(")").empty()) {
                            _printlog("Expected closing parenthesis in call args");
                        }
                        return result;
                    }
                }
                _printlog("Expected closing parenthesis in call args");
                return result;
            }

            std::shared_ptr<VariableExpr> parseIdentifier() {
                static std::regex ident_regex(R"((?!(?:not|is|and|or|del)\b)[a-zA-Z_]\w*)");
                auto location = get_location();
                auto ident = consumeToken(ident_regex);
                if (ident.empty())
                    return nullptr;
                return std::make_shared<VariableExpr>(location, ident);
            }

            std::shared_ptr<Expression> parseStringConcat() {
                auto left = parseMathPow();
                if (!left) _printlog("Expected left side of 'string concat' expression");

                static std::regex concat_tok(R"(~(?!\}))");
                if (!consumeToken(concat_tok).empty()) {
                    auto right = parseLogicalAnd();
                    if (!right) _printlog("Expected right side of 'string concat' expression");
                    left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), BinaryOpExpr::Op::StrConcat);
                }
                return left;
            }

            std::shared_ptr<Expression> parseMathPow() {
                auto left = parseMathPlusMinus();
                if (!left) _printlog("Expected left side of 'math pow' expression");

                while (!consumeToken("**").empty()) {
                    auto right = parseMathPlusMinus();
                    if (!right) _printlog("Expected right side of 'math pow' expression");
                    left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), BinaryOpExpr::Op::MulMul);
                }
                return left;
            }

            std::shared_ptr<Expression> parseMathPlusMinus() {
                static std::regex plus_minus_tok(R"(\+|-(?![}%#]\}))");

                auto left = parseMathMulDiv();
                if (!left) _printlog("Expected left side of 'math plus/minus' expression");
                std::string op_str;
                while (!(op_str = consumeToken(plus_minus_tok)).empty()) {
                    auto right = parseMathMulDiv();
                    if (!right) _printlog("Expected right side of 'math plus/minus' expression");
                    auto op = op_str == "+" ? BinaryOpExpr::Op::Add : BinaryOpExpr::Op::Sub;
                    left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), op);
                }
                return left;
            }

            std::shared_ptr<Expression> parseMathMulDiv() {
                auto left = parseMathUnaryPlusMinus();
                if (!left) _printlog("Expected left side of 'math mul/div' expression");

                static std::regex mul_div_tok(R"(\*\*?|//?|%(?!\}))");
                std::string op_str;
                while (!(op_str = consumeToken(mul_div_tok)).empty()) {
                    auto right = parseMathUnaryPlusMinus();
                    if (!right) _printlog("Expected right side of 'math mul/div' expression");
                    auto op = op_str == "*" ? BinaryOpExpr::Op::Mul
                        : op_str == "**" ? BinaryOpExpr::Op::MulMul
                        : op_str == "/" ? BinaryOpExpr::Op::Div
                        : op_str == "//" ? BinaryOpExpr::Op::DivDiv
                        : BinaryOpExpr::Op::Mod;
                    left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), op);
                }

                if (!consumeToken("|").empty()) {
                    auto expr = parseMathMulDiv();
                    if (expr->mType == Expression::Type_Filter) {
                        auto filter = (FilterExpr*)(expr.get());
                        filter->prepend(std::move(left));
                        return expr;
                    } else {
                        std::vector<std::shared_ptr<Expression>> parts;
                        parts.emplace_back(std::move(left));
                        parts.emplace_back(std::move(expr));
                        return std::make_shared<FilterExpr>(get_location(), std::move(parts));
                    }
                }
                return left;
            }

            std::shared_ptr<Expression> call_func(const std::string & name, ArgumentsExpression && args) const {
                return std::make_shared<CallExpr>(get_location(), std::make_shared<VariableExpr>(get_location(), name), std::move(args));
            }

            std::shared_ptr<Expression> parseMathUnaryPlusMinus() {
                static std::regex unary_plus_minus_tok(R"(\+|-(?![}%#]\}))");
                auto op_str = consumeToken(unary_plus_minus_tok);
                auto expr = parseExpansion();
                if (!expr) _printlog("Expected expr of 'unary plus/minus/expansion' expression");

                if (!op_str.empty()) {
                    auto op = op_str == "+" ? UnaryOpExpr::Op::Plus : UnaryOpExpr::Op::Minus;
                    return std::make_shared<UnaryOpExpr>(get_location(), std::move(expr), op);
                }
                return expr;
            }

            std::shared_ptr<Expression> parseExpansion() {
                static std::regex expansion_tok(R"(\*\*?)");
                auto op_str = consumeToken(expansion_tok);
                auto expr = parseValueExpression();
                if (op_str.empty()) return expr;
                if (!expr) {
                    _printlog("Expected expr of 'expansion' expression");
                    return nullptr;
                }
                return std::make_shared<UnaryOpExpr>(get_location(), std::move(expr), op_str == "*" ? UnaryOpExpr::Op::Expansion : UnaryOpExpr::Op::ExpansionDict);
            }

            std::shared_ptr<Expression> parseValueExpression() {
                auto parseValue = [&]() -> std::shared_ptr<Expression> {
                    auto location = get_location();
                    auto constant = parseConstant();
                    if (constant) return std::make_shared<LiteralExpr>(location, *constant);

                    static std::regex null_regex(R"(null\b)");
                    if (!consumeToken(null_regex).empty()) return std::make_shared<LiteralExpr>(location, Value());

                    auto identifier = parseIdentifier();
                    if (identifier) return identifier;

                    auto braced = parseBracedExpressionOrArray();
                    if (braced) return braced;

                    auto array = parseArray();
                    if (array) return array;

                    auto dictionary = parseDictionary();
                    if (dictionary) return dictionary;

                    _printlog("Expected value expression");
                    return nullptr;
                };

                auto value = parseValue();

                while (it != end && consumeSpaces() && peekSymbols({ "[", "." })) {
                    if (!consumeToken("[").empty()) {
                        std::shared_ptr<Expression> index;
                        auto slice_loc = get_location();
                        std::shared_ptr<Expression> start, end, step;
                        bool c1 = false, c2 = false;

                        if (!peekSymbols({ ":" })) {
                            start = parseExpression();
                        }

                        if (!consumeToken(":").empty()) {
                            c1 = true;
                            if (!peekSymbols({ ":", "]" })) {
                                end = parseExpression();
                            }
                            if (!consumeToken(":").empty()) {
                                c2 = true;
                                if (!peekSymbols({ "]" })) {
                                    step = parseExpression();
                                }
                            }
                        }

                        if ((c1 || c2) && (start || end || step)) {
                            index = std::make_shared<SliceExpr>(slice_loc, std::move(start), std::move(end), std::move(step));
                        } else {
                            index = std::move(start);
                        }
                        if (consumeToken("]").empty()) {
                            MNN_ERROR("Expected closing bracket in subscript");
                        }
                        if (index != nullptr) {
                            value = std::make_shared<SubscriptExpr>(value->location, std::move(value), std::move(index));
                        }
                    } else if (!consumeToken(".").empty()) {
                        auto identifier = parseIdentifier();
                        if (!identifier) _printlog("Expected identifier in subscript");

                        consumeSpaces();
                        if (peekSymbols({ "(" })) {
                            auto callParams = parseCallArgs();
                            value = std::make_shared<MethodCallExpr>(identifier->location, std::move(value), std::move(identifier), std::move(callParams));
                        } else {
                            auto key = std::make_shared<LiteralExpr>(identifier->location, Value(identifier->get_name()));
                            value = std::make_shared<SubscriptExpr>(identifier->location, std::move(value), std::move(key));
                        }
                    }
                    consumeSpaces();
                }

                if (peekSymbols({ "(" })) {
                    auto location = get_location();
                    auto callParams = parseCallArgs();
                    value = std::make_shared<CallExpr>(location, std::move(value), std::move(callParams));
                }
                return value;
            }

            std::shared_ptr<Expression> parseBracedExpressionOrArray() {
                if (consumeToken("(").empty()) return nullptr;

                auto expr = parseExpression();
                if (!expr) _printlog("Expected expression in braced expression");

                if (!consumeToken(")").empty()) {
                    return expr;  // Drop the parentheses
                }

                std::vector<std::shared_ptr<Expression>> tuple;
                tuple.emplace_back(std::move(expr));

                while (it != end) {
                    if (consumeToken(",").empty()) _printlog("Expected comma in tuple");
                    auto next = parseExpression();
                    if (!next) _printlog("Expected expression in tuple");
                    tuple.push_back(std::move(next));

                    if (!consumeToken(")").empty()) {
                        return std::make_shared<ArrayExpr>(get_location(), std::move(tuple));
                    }
                }
                _printlog("Expected closing parenthesis");
                return nullptr;
            }

            std::shared_ptr<Expression> parseArray() {
                if (consumeToken("[").empty()) return nullptr;

                std::vector<std::shared_ptr<Expression>> elements;
                if (!consumeToken("]").empty()) {
                    return std::make_shared<ArrayExpr>(get_location(), std::move(elements));
                }
                auto first_expr = parseExpression();
                if (!first_expr) _printlog("Expected first expression in array");
                elements.push_back(std::move(first_expr));

                while (it != end) {
                    if (!consumeToken(",").empty()) {
                        auto expr = parseExpression();
                        if (!expr) _printlog("Expected expression in array");
                        elements.push_back(std::move(expr));
                    } else if (!consumeToken("]").empty()) {
                        return std::make_shared<ArrayExpr>(get_location(), std::move(elements));
                    } else {
                        _printlog("Expected comma or closing bracket in array");
                    }
                }
                _printlog("Expected closing bracket");
                return nullptr;
            }

            std::shared_ptr<Expression> parseDictionary() {
                if (consumeToken("{").empty()) return nullptr;

                std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<Expression>>> elements;
                if (!consumeToken("}").empty()) {
                    return std::make_shared<DictExpr>(get_location(), std::move(elements));
                }

                auto parseKeyValuePair = [&]() {
                    auto key = parseExpression();
                    if (!key) _printlog("Expected key in dictionary");
                    if (consumeToken(":").empty()) _printlog("Expected colon betweek key & value in dictionary");
                    auto value = parseExpression();
                    if (!value) _printlog("Expected value in dictionary");
                    elements.emplace_back(std::make_pair(std::move(key), std::move(value)));
                };

                parseKeyValuePair();

                while (it != end) {
                    if (!consumeToken(",").empty()) {
                        parseKeyValuePair();
                    } else if (!consumeToken("}").empty()) {
                        return std::make_shared<DictExpr>(get_location(), std::move(elements));
                    } else {
                        _printlog("Expected comma or closing brace in dictionary");
                    }
                }
                _printlog("Expected closing brace");
                return nullptr;
            }

            SpaceHandling parsePreSpace(const std::string& s) const {
                if (s == "-")
                    return SpaceHandling::Strip;
                return SpaceHandling::Keep;
            }

            SpaceHandling parsePostSpace(const std::string& s) const {
                if (s == "-") return SpaceHandling::Strip;
                return SpaceHandling::Keep;
            }

            using TemplateTokenVector = std::vector<std::shared_ptr<TemplateToken>>;
            using TemplateTokenIterator = TemplateTokenVector::const_iterator;

            std::vector<std::string> parseVarNames() {
                static std::regex varnames_regex(R"(((?:\w+)(?:\s*,\s*(?:\w+))*)\s*)");

                std::vector<std::string> group;
                if ((group = consumeTokenGroups(varnames_regex)).empty()) _printlog("Expected variable names");
                std::vector<std::string> varnames;
                std::istringstream iss(group[1]);
                std::string varname;
                while (std::getline(iss, varname, ',')) {
                    varnames.push_back(strip(varname));
                }
                return varnames;
            }

            std::string unexpected(const TemplateToken & token) const {
                return std::string("Unexpected " + TemplateToken::typeToString(token.type)
                        + error_location_suffix(*template_str, token.location.pos));
            }
            std::string unterminated(const TemplateToken & token) const {
                return std::string("Unterminated " + TemplateToken::typeToString(token.type)
                        + error_location_suffix(*template_str, token.location.pos));
            }

            TemplateTokenVector tokenize() {
                static std::regex comment_tok(R"(\{#([-~]?)([\s\S]*?)([-~]?)#\})");
                static std::regex expr_open_regex(R"(\{\{([-~])?)");
                static std::regex block_open_regex(R"(^\{%([-~])?\s*)");
                static std::regex block_keyword_tok(R"((if|else|elif|endif|for|endfor|generation|endgeneration|set|endset|block|endblock|macro|endmacro|filter|endfilter|break|continue)\b)");
                static std::regex non_text_open_regex(R"(\{\{|\{%|\{#)");
                static std::regex expr_close_regex(R"(\s*([-~])?\}\})");
                static std::regex block_close_regex(R"(\s*([-~])?%\})");

                TemplateTokenVector tokens;
                std::vector<std::string> group;
                std::string text;
                std::smatch match;

                while (it != end) {
                    auto location = get_location();

                    if (!(group = consumeTokenGroups(comment_tok, SpaceHandling::Keep)).empty()) {
                        auto pre_space = parsePreSpace(group[1]);
                        auto content = group[2];
                        auto post_space = parsePostSpace(group[3]);
                        tokens.push_back(std::make_shared<CommentTemplateToken>(location, pre_space, post_space, content));
                    } else if (!(group = consumeTokenGroups(expr_open_regex, SpaceHandling::Keep)).empty()) {
                        auto pre_space = parsePreSpace(group[1]);
                        auto expr = parseExpression();

                        if ((group = consumeTokenGroups(expr_close_regex)).empty()) {
                            _printlog("Expected closing expression tag");
                        }

                        auto post_space = parsePostSpace(group[1]);
                        tokens.push_back(std::make_shared<ExpressionTemplateToken>(location, pre_space, post_space, std::move(expr)));
                    } else if (!(group = consumeTokenGroups(block_open_regex, SpaceHandling::Keep)).empty()) {
                        auto pre_space = parsePreSpace(group[1]);

                        std::string keyword;

                        auto parseBlockClose = [&]() -> SpaceHandling {
                            if ((group = consumeTokenGroups(block_close_regex)).empty()) _printlog("Expected closing block tag");
                            return parsePostSpace(group[1]);
                        };

                        if ((keyword = consumeToken(block_keyword_tok)).empty()) _printlog("Expected block keyword");

                        if (keyword == "if") {
                            auto condition = parseExpression();
                            if (!condition) _printlog("Expected condition in if block");

                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<IfTemplateToken>(location, pre_space, post_space, std::move(condition)));
                        } else if (keyword == "elif") {
                            auto condition = parseExpression();
                            if (!condition) _printlog("Expected condition in elif block");

                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<ElifTemplateToken>(location, pre_space, post_space, std::move(condition)));
                        } else if (keyword == "else") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<ElseTemplateToken>(location, pre_space, post_space));
                        } else if (keyword == "endif") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<EndIfTemplateToken>(location, pre_space, post_space));
                        } else if (keyword == "for") {
                            static std::regex recursive_tok(R"(recursive\b)");
                            static std::regex if_tok(R"(if\b)");

                            auto varnames = parseVarNames();
                            static std::regex in_tok(R"(in\b)");
                            if (consumeToken(in_tok).empty()) _printlog("Expected 'in' keyword in for block");
                            auto iterable = parseExpression(/* allow_if_expr = */ false);
                            if (!iterable) _printlog("Expected iterable in for block");

                            std::shared_ptr<Expression> condition;
                            if (!consumeToken(if_tok).empty()) {
                                condition = parseExpression();
                            }
                            auto recursive = !consumeToken(recursive_tok).empty();

                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<ForTemplateToken>(location, pre_space, post_space, std::move(varnames), std::move(iterable), std::move(condition), recursive));
                        } else if (keyword == "endfor") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<EndForTemplateToken>(location, pre_space, post_space));
                        } else if (keyword == "generation") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<GenerationTemplateToken>(location, pre_space, post_space));
                        } else if (keyword == "endgeneration") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<EndGenerationTemplateToken>(location, pre_space, post_space));
                        } else if (keyword == "set") {
                            static std::regex namespaced_var_regex(R"((\w+)\s*\.\s*(\w+))");

                            std::string ns;
                            std::vector<std::string> var_names;
                            std::shared_ptr<Expression> value;
                            if (!(group = consumeTokenGroups(namespaced_var_regex)).empty()) {
                                ns = group[1];
                                var_names.push_back(group[2]);

                                if (consumeToken("=").empty()) _printlog("Expected equals sign in set block");

                                value = parseExpression();
                                if (!value) _printlog("Expected value in set block");
                            } else {
                                var_names = parseVarNames();

                                if (!consumeToken("=").empty()) {
                                    value = parseExpression();
                                    if (!value) _printlog("Expected value in set block");
                                }
                            }
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<SetTemplateToken>(location, pre_space, post_space, ns, var_names, std::move(value)));
                        } else if (keyword == "endset") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<EndSetTemplateToken>(location, pre_space, post_space));
                        } else if (keyword == "macro") {
                            auto macroname = parseIdentifier();
                            if (!macroname) _printlog("Expected macro name in macro block");
                            auto params = parseParameters();

                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<MacroTemplateToken>(location, pre_space, post_space, std::move(macroname), std::move(params)));
                        } else if (keyword == "endmacro") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<EndMacroTemplateToken>(location, pre_space, post_space));
                        } else if (keyword == "filter") {
                            auto filter = parseExpression();
                            if (!filter) _printlog("Expected expression in filter block");

                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<FilterTemplateToken>(location, pre_space, post_space, std::move(filter)));
                        } else if (keyword == "endfilter") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<EndFilterTemplateToken>(location, pre_space, post_space));
                        } else if (keyword == "break" || keyword == "continue") {
                            auto post_space = parseBlockClose();
                            tokens.push_back(std::make_shared<LoopControlTemplateToken>(location, pre_space, post_space, keyword == "break" ? LoopControlType::Break : LoopControlType::Continue));
                        } else {
                            _printlog("Unexpected block: " + keyword);
                        }
                    } else if (std::regex_search(it, end, match, non_text_open_regex)) {
                        if (!match.position()) {
                            if (match[0] != "{#")
                                _printlog("Internal error: Expected a comment");
                            _printlog("Missing end of comment tag");
                        }
                        auto text_end = it + match.position();
                        text = std::string(it, text_end);
                        it = text_end;
                        tokens.push_back(std::make_shared<TextTemplateToken>(location, SpaceHandling::Keep, SpaceHandling::Keep, text));
                    } else {
                        text = std::string(it, end);
                        it = end;
                        tokens.push_back(std::make_shared<TextTemplateToken>(location, SpaceHandling::Keep, SpaceHandling::Keep, text));
                    }
                }
                return tokens;
            }

            std::shared_ptr<TemplateNode> parseTemplate(
                    const TemplateTokenIterator & begin,
                    TemplateTokenIterator & it,
                    const TemplateTokenIterator & end,
                    bool fully = false) const {
                std::vector<std::shared_ptr<TemplateNode>> children;
                while (it != end) {
                    const auto start = it;
                    const auto & token = *(it++);
                    if (token->type == TemplateToken::Type::If) {
                        auto if_token = (IfTemplateToken*)(token.get());
                        std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<TemplateNode>>> cascade;
                        cascade.emplace_back(std::move(if_token->condition), parseTemplate(begin, it, end));

                        while (it != end && (*it)->type == TemplateToken::Type::Elif) {
                            auto elif_token = (ElifTemplateToken*)((*(it++)).get());
                            cascade.emplace_back(std::move(elif_token->condition), parseTemplate(begin, it, end));
                        }

                        if (it != end && (*it)->type == TemplateToken::Type::Else) {
                            cascade.emplace_back(nullptr, parseTemplate(begin, ++it, end));
                        }
                        if (it == end || (*(it++))->type != TemplateToken::Type::EndIf) {
                            MNN_ERROR("%s\n", unterminated(**start).c_str());
                        }
                        children.emplace_back(std::make_shared<IfNode>(token->location, std::move(cascade)));
                    } else if (token->type == TemplateToken::Type::For) {
                        auto for_token = (ForTemplateToken*)(token.get());
                        auto body = parseTemplate(begin, it, end);
                        auto else_body = std::shared_ptr<TemplateNode>();
                        if (it != end && (*it)->type == TemplateToken::Type::Else) {
                            else_body = parseTemplate(begin, ++it, end);
                        }
                        if (it == end || (*(it++))->type != TemplateToken::Type::EndFor) {
                            MNN_ERROR("%s\n", unterminated(**start).c_str());
                        }
                        children.emplace_back(std::make_shared<ForNode>(token->location, std::move(for_token->var_names), std::move(for_token->iterable), std::move(for_token->condition), std::move(body), for_token->recursive, std::move(else_body)));
                    } else if(token->type == TemplateToken::Type::Generation) {
                        auto body = parseTemplate(begin, it, end);
                        if (it == end || (*(it++))->type != TemplateToken::Type::EndGeneration) {
                            MNN_ERROR("%s\n", unterminated(**start).c_str());
                        }
                        // Treat as a no-op, as our scope is templates for inference, not training (`{% generation %}` wraps generated tokens for masking).
                        children.emplace_back(std::move(body));
                    } else if(token->type == TemplateToken::Type::Text) {
                        auto text_token = (TextTemplateToken*)(token.get());
                        SpaceHandling pre_space = (it - 1) != begin ? (*(it - 2))->post_space : SpaceHandling::Keep;
                        SpaceHandling post_space = it != end ? (*it)->pre_space : SpaceHandling::Keep;

                        auto text = text_token->text;
                        if (post_space == SpaceHandling::Strip) {
                            static std::regex trailing_space_regex(R"(\s+$)");
                            text = std::regex_replace(text, trailing_space_regex, "");
                        } else if (options.lstrip_blocks && it != end) {
                            auto i = text.size();
                            while (i > 0 && (text[i - 1] == ' ' || text[i - 1] == '\t')) i--;
                            if ((i == 0 && (it - 1) == begin) || (i > 0 && text[i - 1] == '\n')) {
                                text.resize(i);
                            }
                        }
                        if (pre_space == SpaceHandling::Strip) {
                            static std::regex leading_space_regex(R"(^\s+)");
                            text = std::regex_replace(text, leading_space_regex, "");
                        } else if (options.trim_blocks && (it - 1) != begin && (*(it - 2))->type != TemplateToken::Type::Expression) {
                            if (!text.empty() && text[0] == '\n') {
                                text.erase(0, 1);
                            }
                        }
                        if (it == end && !options.keep_trailing_newline) {
                            auto i = text.size();
                            if (i > 0 && text[i - 1] == '\n') {
                                i--;
                                if (i > 0 && text[i - 1] == '\r') i--;
                                text.resize(i);
                            }
                        }
                        children.emplace_back(std::make_shared<TextNode>(token->location, text));
                    } else if(token->type == TemplateToken::Type::Expression) {
                        auto expr_token = (ExpressionTemplateToken*)(token.get());
                        children.emplace_back(std::make_shared<ExpressionNode>(token->location, std::move(expr_token->expr)));
                    } else if(token->type == TemplateToken::Type::Set) {
                        auto set_token = (SetTemplateToken*)(token.get());
                        if (set_token->value) {
                            children.emplace_back(std::make_shared<SetNode>(token->location, set_token->ns, set_token->var_names, std::move(set_token->value)));
                        } else {
                            auto value_template = parseTemplate(begin, it, end);
                            if (it == end || (*(it++))->type != TemplateToken::Type::EndSet) {
                                MNN_ERROR("%s\n", unterminated(**start).c_str());
                            }
                            if (!set_token->ns.empty()) _printlog("Namespaced set not supported in set with template value");
                            if (set_token->var_names.size() != 1) _printlog("Structural assignment not supported in set with template value");
                            auto & name = set_token->var_names[0];
                            children.emplace_back(std::make_shared<SetTemplateNode>(token->location, name, std::move(value_template)));
                        }
                    } else if(token->type == TemplateToken::Type::Macro) {
                        auto macro_token = (MacroTemplateToken*)(token.get());
                        auto body = parseTemplate(begin, it, end);
                        if (it == end || (*(it++))->type != TemplateToken::Type::EndMacro) {
                            MNN_ERROR("%s\n", unterminated(**start).c_str());
                        }
                        children.emplace_back(std::make_shared<MacroNode>(token->location, std::move(macro_token->name), std::move(macro_token->params), std::move(body)));
                    } else if(token->type == TemplateToken::Type::Filter) {
                        auto filter_token = (FilterTemplateToken*)(token.get());
                        auto body = parseTemplate(begin, it, end);
                        if (it == end || (*(it++))->type != TemplateToken::Type::EndFilter) {
                            MNN_ERROR("%s\n", unterminated(**start).c_str());
                        }
                        children.emplace_back(std::make_shared<FilterNode>(token->location, std::move(filter_token->filter), std::move(body)));
                    } else if(token->type == TemplateToken::Type::Comment) {
                        // Ignore comments
                    } else if(token->type == TemplateToken::Type::Break) {
                        auto ctrl_token = (LoopControlTemplateToken*)(token.get());
                        children.emplace_back(std::make_shared<LoopControlNode>(token->location, ctrl_token->control_type));
                    } else {
                        bool needBreak = false;
                        switch (token->type) {
                            case TemplateToken::Type::EndSet:
                            case TemplateToken::Type::EndFor:
                            case TemplateToken::Type::EndMacro:
                            case TemplateToken::Type::EndFilter:
                            case TemplateToken::Type::EndIf:
                            case TemplateToken::Type::Else:
                            case TemplateToken::Type::Elif:
                            case TemplateToken::Type::EndGeneration:
                                it--;
                                needBreak = true;
                                break;
                            default:
                                MNN_ERROR("%s\n", unexpected(**(it-1)).c_str());
                        }
                        if (needBreak) {
                            break;
                        }
                    }
                }
                if (fully && it != end) {
                    MNN_ERROR("%s\n", unexpected(**it).c_str());
                }
                if (children.empty()) {
                    return std::make_shared<TextNode>(Location { template_str, 0 }, std::string());
                } else if (children.size() == 1) {
                    return std::move(children[0]);
                } else {
                    return std::make_shared<SequenceNode>(children[0]->location(), std::move(children));
                }
            }

        public:

            static std::shared_ptr<TemplateNode> parse(const std::string& template_str, const Options & options) {
                Parser parser(std::make_shared<std::string>(normalize_newlines(template_str)), options);
                auto tokens = parser.tokenize();
                TemplateTokenIterator begin = tokens.begin();
                auto it = begin;
                TemplateTokenIterator end = tokens.end();
                return parser.parseTemplate(begin, it, end, /* fully= */ true);
            }
    };
    std::shared_ptr<TemplateNode> parse(const std::string& template_str, const Options & options) {
        return Parser::parse(template_str, options);
    }

    std::shared_ptr<Context> Context::builtins() {
        auto globals = Value::object();

        //  globals.set("raise_exception", simple_function("raise_exception", { "message" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
        //    _printlog(args.at("message").get<std::string>());
        //  }));
        globals.set("tojson", simple_function("tojson", { "value", "indent" }, [](const std::shared_ptr<Context> &, Value & args) {
                    return Value(args.at("value").dump(args.get<int64_t>("indent", -1), /* to_json= */ true));
                    }));
        globals.set("items", simple_function("items", { "object" }, [](const std::shared_ptr<Context> &, Value & args) {
                    auto items = Value::array();
                    if (args.contains("object")) {
                    auto & obj = args.at("object");
                    if (obj.is_string()) {
                    rapidjson::Document doc;
                    doc.Parse(obj.get<std::string>().c_str());
                    for (auto& kv : doc.GetObject()) {
                    items.push_back(Value::array({kv.name, kv.value}));
                    }
                    } else if (!obj.is_null()) {
                    for (auto & key : obj.keys()) {
                    items.push_back(Value::array({key, obj.at(key)}));
                    }
                    }
                    }
                    return items;
                    }));
        globals.set("last", simple_function("last", { "items" }, [](const std::shared_ptr<Context> &, Value & args) {
                    auto items = args.at("items");
                    if (!items.is_array()) _printlog("object is not a list");
                    if (items.empty()) return Value();
                    return items.at(items.size() - 1);
                    }));
        globals.set("trim", simple_function("trim", { "text" }, [](const std::shared_ptr<Context> &, Value & args) {
                    auto & text = args.at("text");
                    return text.is_null() ? text : Value(strip(text.get<std::string>()));
                    }));
        auto char_transform_function = [](const std::string & name, const std::function<char(char)> & fn) {
            return simple_function(name, { "text" }, [=](const std::shared_ptr<Context> &, Value & args) {
                    auto text = args.at("text");
                    if (text.is_null()) return text;
                    std::string res;
                    auto str = text.get<std::string>();
                    std::transform(str.begin(), str.end(), std::back_inserter(res), fn);
                    return Value(res);
                    });
        };
        globals.set("lower", char_transform_function("lower", ::tolower));
        globals.set("upper", char_transform_function("upper", ::toupper));
        globals.set("default", Value::callable([=](const std::shared_ptr<Context> &, ArgumentsValue & args) {
                    args.expectArgs("default", {2, 3}, {0, 1});
                    auto & value = args.args[0];
                    auto & default_value = args.args[1];
                    bool boolean = false;
                    if (args.args.size() == 3) {
                    boolean = args.args[2].get<bool>();
                    } else {
                    Value bv = args.get_named("boolean");
                    if (!bv.is_null()) {
                    boolean = bv.get<bool>();
                    }
                    }
                    return boolean ? (value.to_bool() ? value : default_value) : value.is_null() ? default_value : value;
                    }));
        auto escape = simple_function("escape", { "text" }, [](const std::shared_ptr<Context> &, Value & args) {
                return Value(html_escape(args.at("text").get<std::string>()));
                });
        globals.set("e", escape);
        globals.set("escape", escape);
        globals.set("joiner", simple_function("joiner", { "sep" }, [](const std::shared_ptr<Context> &, Value & args) {
                    auto sep = args.get<std::string>("sep", "");
                    auto first = std::make_shared<bool>(true);
                    return simple_function("", {}, [sep, first](const std::shared_ptr<Context> &, const Value &) -> Value {
                            if (*first) {
                            *first = false;
                            return "";
                            }
                            return sep;
                            });
                    return Value(html_escape(args.at("text").get<std::string>()));
                    }));
        globals.set("count", simple_function("count", { "items" }, [](const std::shared_ptr<Context> &, Value & args) {
                    return Value((int64_t) args.at("items").size());
                    }));
        globals.set("dictsort", simple_function("dictsort", { "value" }, [](const std::shared_ptr<Context> &, Value & args) {
                    if (args.size() != 1) _printlog("dictsort expects exactly 1 argument (TODO: fix implementation)");
                    auto & value = args.at("value");
                    auto keys = value.keys();
                    std::sort(keys.begin(), keys.end());
                    auto res = Value::array();
                    for (auto & key : keys) {
                    res.push_back(Value::array({key, value.at(key)}));
                    }
                    return res;
                    }));
        globals.set("join", simple_function("join", { "items", "d" }, [](const std::shared_ptr<Context> &, Value & args) {
                    auto do_join = [](Value & items, const std::string & sep) {
                    if (!items.is_array()) _printlog("object is not iterable: " + items.dump());
                    std::ostringstream oss;
                    auto first = true;
                    for (size_t i = 0, n = items.size(); i < n; ++i) {
                    if (first) first = false;
                    else oss << sep;
                    oss << items.at(i).to_str();
                    }
                    return Value(oss.str());
                    };
                    auto sep = args.get<std::string>("d", "");
                    if (args.contains("items")) {
                    auto & items = args.at("items");
                    return do_join(items, sep);
                    } else {
                    return simple_function("", {"items"}, [sep, do_join](const std::shared_ptr<Context> &, Value & args) {
                            auto & items = args.at("items");
                            if (!items.to_bool() || !items.is_array()) _printlog("join expects an array for items, got: " + items.dump());
                            return do_join(items, sep);
                            });
                    }
        }));
        globals.set("namespace", Value::callable([=](const std::shared_ptr<Context> &, ArgumentsValue & args) {
                    auto ns = Value::object();
                    args.expectArgs("namespace", {0, 0}, {0, (std::numeric_limits<size_t>::max)()});
                    for (auto & iter : args.kwargs) {
                    auto& name = iter.first;
                    auto& value = iter.second;
                    ns.set(name, value);
                    }
                    return ns;
                    }));
        auto equalto = simple_function("equalto", { "expected", "actual" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
                return args.at("actual") == args.at("expected");
                });
        globals.set("equalto", equalto);
        globals.set("==", equalto);
        globals.set("length", simple_function("length", { "items" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
                    auto & items = args.at("items");
                    return (int64_t) items.size();
                    }));
        globals.set("safe", simple_function("safe", { "value" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
                    return args.at("value").to_str();
                    }));
        globals.set("string", simple_function("string", { "value" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
                    return args.at("value").to_str();
                    }));
        globals.set("int", simple_function("int", { "value" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
                    return args.at("value").to_int();
                    }));
        globals.set("list", simple_function("list", { "items" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
                    auto & items = args.at("items");
                    if (!items.is_array()) _printlog("object is not iterable");
                    return items;
                    }));
        globals.set("unique", simple_function("unique", { "items" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
                    auto & items = args.at("items");
                    if (!items.is_array()) _printlog("object is not iterable");
                    std::unordered_set<Value> seen;
                    auto result = Value::array();
                    for (size_t i = 0, n = items.size(); i < n; i++) {
                    auto pair = seen.insert(items.at(i));
                    if (pair.second) {
                    result.push_back(items.at(i));
                    }
                    }
                    return result;
                    }));
        auto make_filter = [](const Value & filter, Value & extra_args) -> Value {
            return simple_function("", { "value" }, [=](const std::shared_ptr<Context> & context, Value & args) {
                    auto & value = args.at("value");
                    ArgumentsValue actual_args;
                    actual_args.args.emplace_back(value);
                    for (size_t i = 0, n = extra_args.size(); i < n; i++) {
                    actual_args.args.emplace_back(extra_args.at(i));
                    }
                    return filter.call(context, actual_args);
                    });
        };
        auto select_or_reject = [make_filter](bool is_select) {
            return Value::callable([=](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
                    args.expectArgs(is_select ? "select" : "reject", {2, (std::numeric_limits<size_t>::max)()}, {0, 0});
                    auto & items = args.args[0];
                    if (items.is_null()) {
                    return Value::array();
                    }
                    if (!items.is_array()) {
                    _printlog("object is not iterable: " + items.dump());
                    }

                    auto filter_fn = context->get(args.args[1]);
                    if (filter_fn.is_null()) {
                    _printlog("Undefined filter: " + args.args[1].dump());
                    }

                    auto filter_args = Value::array();
                    for (size_t i = 2, n = args.args.size(); i < n; i++) {
                    filter_args.push_back(args.args[i]);
                    }
                    auto filter = make_filter(filter_fn, filter_args);

                    auto res = Value::array();
                    for (size_t i = 0, n = items.size(); i < n; i++) {
                        auto & item = items.at(i);
                        ArgumentsValue filter_args;
                        filter_args.args.emplace_back(item);
                        auto pred_res = filter.call(context, filter_args);
                        if (pred_res.to_bool() == (is_select ? true : false)) {
                            res.push_back(item);
                        }
                    }
                    return res;
            });
        };
        globals.set("select", select_or_reject(/* is_select= */ true));
        globals.set("reject", select_or_reject(/* is_select= */ false));
        globals.set("map", Value::callable([=](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
                    auto res = Value::array();
                    if (args.args.size() == 1 &&
                            ((args.has_named("attribute") && args.kwargs.size() == 1) || (args.has_named("default") && args.kwargs.size() == 2))) {
                    auto & items = args.args[0];
                    auto attr_name = args.get_named("attribute");
                    auto default_value = args.get_named("default");
                    for (size_t i = 0, n = items.size(); i < n; i++) {
                    auto & item = items.at(i);
                    auto attr = item.get(attr_name);
                    res.push_back(attr.is_null() ? default_value : attr);
                    }
                    } else if (args.kwargs.empty() && args.args.size() >= 2) {
                    auto fn = context->get(args.args[1]);
                    if (fn.is_null()) _printlog("Undefined filter: " + args.args[1].dump());
                    ArgumentsValue filter_args { {Value()}, {} };
                    for (size_t i = 2, n = args.args.size(); i < n; i++) {
                    filter_args.args.emplace_back(args.args[i]);
                    }
                    for (size_t i = 0, n = args.args[0].size(); i < n; i++) {
                    auto & item = args.args[0].at(i);
                    filter_args.args[0] = item;
                    res.push_back(fn.call(context, filter_args));
                    }
                    } else {
                        _printlog("Invalid or unsupported arguments for map");
                    }
                    return res;
        }));
        globals.set("indent", simple_function("indent", { "text", "indent", "first" }, [](const std::shared_ptr<Context> &, Value & args) {
                    auto text = args.at("text").get<std::string>();
                    auto first = args.get<bool>("first", false);
                    std::string out;
                    std::string indent(args.get<int64_t>("indent", 0), ' ');
                    std::istringstream iss(text);
                    std::string line;
                    auto is_first = true;
                    while (std::getline(iss, line, '\n')) {
                    auto needs_indent = !is_first || first;
                    if (is_first) is_first = false;
                    else out += "\n";
                    if (needs_indent) out += indent;
                    out += line;
                    }
                    if (!text.empty() && text.back() == '\n') out += "\n";
                    return out;
                    }));
        auto select_or_reject_attr = [](bool is_select) {
            return Value::callable([=](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
                    args.expectArgs(is_select ? "selectattr" : "rejectattr", {2, (std::numeric_limits<size_t>::max)()}, {0, 0});
                    auto & items = args.args[0];
                    if (items.is_null())
                    return Value::array();
                    if (!items.is_array()) _printlog("object is not iterable: " + items.dump());
                    auto attr_name = args.args[1].get<std::string>();

                    bool has_test = false;
                    Value test_fn;
                    ArgumentsValue test_args {{Value()}, {}};
                    if (args.args.size() >= 3) {
                    has_test = true;
                    test_fn = context->get(args.args[2]);
                    if (test_fn.is_null()) _printlog("Undefined test: " + args.args[2].dump());
                    for (size_t i = 3, n = args.args.size(); i < n; i++) {
                    test_args.args.emplace_back(args.args[i]);
                    }
                    test_args.kwargs = args.kwargs;
                    }

                    auto res = Value::array();
                    for (size_t i = 0, n = items.size(); i < n; i++) {
                        auto & item = items.at(i);
                        auto attr = item.get(attr_name);
                        if (has_test) {
                            test_args.args[0] = attr;
                            if (test_fn.call(context, test_args).to_bool() == (is_select ? true : false)) {
                                res.push_back(item);
                            }
                        } else {
                            res.push_back(attr);
                        }
                    }
                    return res;
            });
        };
        globals.set("selectattr", select_or_reject_attr(/* is_select= */ true));
        globals.set("rejectattr", select_or_reject_attr(/* is_select= */ false));
        globals.set("range", Value::callable([=](const std::shared_ptr<Context> &, ArgumentsValue & args) {
                    std::vector<int64_t> startEndStep(3);
                    std::vector<bool> param_set(3);
                    if (args.args.size() == 1) {
                    startEndStep[1] = args.args[0].get<int64_t>();
                    param_set[1] = true;
                    } else {
                    for (size_t i = 0; i < args.args.size(); i++) {
                    auto & arg = args.args[i];
                    auto v = arg.get<int64_t>();
                    startEndStep[i] = v;
                    param_set[i] = true;
                    }
                    }
                    for (auto & iter : args.kwargs) {
                    auto& name = iter.first;
                    auto& value = iter.second;
                    size_t i;
                    if (name == "start") {
                    i = 0;
                    } else if (name == "end") {
                        i = 1;
                    } else if (name == "step") {
                        i = 2;
                    } else {
                        _printlog("Unknown argument " + name + " for function range");
                    }

                    if (param_set[i]) {
                        _printlog("Duplicate argument " + name + " for function range");
                    }
                    startEndStep[i] = value.get<int64_t>();
                    param_set[i] = true;
                    }
                    if (!param_set[1]) {
                        _printlog("Missing required argument 'end' for function range");
                    }
                    int64_t start = param_set[0] ? startEndStep[0] : 0;
                    int64_t end = startEndStep[1];
                    int64_t step = param_set[2] ? startEndStep[2] : 1;

                    auto res = Value::array();
                    if (step > 0) {
                        for (int64_t i = start; i < end; i += step) {
                            res.push_back(Value(i));
                        }
                    } else {
                        for (int64_t i = start; i > end; i += step) {
                            res.push_back(Value(i));
                        }
                    }
                    return res;
        }));

        return std::make_shared<Context>(std::move(globals));
    }


};

#endif
