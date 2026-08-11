//
//  unicode.cpp
//
//  Created by MNN on 2024/01/01.
//  ZhaodeWang
//

#include "unicode.hpp"
#include <cstring>
#include <regex>
#include <unordered_map>

namespace MNN {
namespace Unicode {

// ==========================================
// Binary search helpers
// ==========================================

Category get_category(int32_t cp) {
    if (cp < 0 || cp >= 0x110000) return CAT_Cn;
    uint32_t ucp = (uint32_t)cp;
    size_t lo = 0, hi = kCategoryRangesSize;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (ucp > kCategoryRanges[mid].end) lo = mid + 1;
        else if (ucp < kCategoryRanges[mid].start) hi = mid;
        else return (Category)kCategoryRanges[mid].category;
    }
    return CAT_Cn;
}

bool is_letter(int32_t cp) {
    Category c = get_category(cp);
    return c == CAT_Lu || c == CAT_Ll || c == CAT_Lt || c == CAT_Lm || c == CAT_Lo;
}

bool is_number(int32_t cp) {
    Category c = get_category(cp);
    return c == CAT_Nd || c == CAT_Nl || c == CAT_No;
}

bool is_punctuation(int32_t cp) {
    Category c = get_category(cp);
    return c == CAT_Pc || c == CAT_Pd || c == CAT_Ps || c == CAT_Pe ||
           c == CAT_Pi || c == CAT_Pf || c == CAT_Po;
}

bool is_whitespace(int32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v') return true;
    return get_category(cp) == CAT_Zs;
}

bool is_mark(int32_t cp) {
    Category c = get_category(cp);
    return c == CAT_Mn || c == CAT_Mc || c == CAT_Me;
}

int32_t to_lower(int32_t cp) {
    size_t lo = 0, hi = kToLowerSize;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if ((uint32_t)cp > kToLower[mid].from) lo = mid + 1;
        else if ((uint32_t)cp < kToLower[mid].from) hi = mid;
        else return (int32_t)kToLower[mid].to;
    }
    return cp;
}

// ==========================================
// Text folding (Unicode → single-byte)
// ==========================================

// Maps Unicode text to single-byte representation for regex matching.
// ASCII chars are preserved as-is; non-ASCII codepoints are folded to category markers:
//   \x01=letter, \x02=number, \x03=whitespace, \x04=other, \x05=mark, \x06=punct, \x07=symbol
static std::string fold_text(const std::string& text, std::vector<size_t>& byte_offsets) {
    std::string folded;
    byte_offsets.clear();
    const uint8_t* ptr = (const uint8_t*)text.data();
    size_t len = text.size(), i = 0;
    while (i < len) {
        byte_offsets.push_back(i);
        int32_t cp;
        int r = utf8_decode(ptr + i, len - i, &cp);
        if (cp < 0x80) {
            folded += (char)cp;
        } else if (is_letter(cp)) {
            folded += '\x01';
        } else if (is_number(cp)) {
            folded += '\x02';
        } else if (is_whitespace(cp)) {
            folded += '\x03';
        } else if (is_mark(cp)) {
            folded += '\x05';
        } else {
            Category cat = get_category(cp);
            if (cat >= CAT_Pc && cat <= CAT_Po) folded += '\x06';
            else if (cat >= CAT_Sm && cat <= CAT_So) folded += '\x07';
            else folded += '\x04';
        }
        i += r;
    }
    byte_offsets.push_back(len);
    return folded;
}

// ==========================================
// Lightweight byte-regex engine
// ==========================================

namespace {

// 256-bit character set for single-byte matching
struct CharSet {
    uint64_t bits[4] = {};
    void set(uint8_t b) { bits[b >> 6] |= 1ULL << (b & 63); }
    bool test(uint8_t b) const { return (bits[b >> 6] >> (b & 63)) & 1; }
    void negate() { for (auto& w : bits) w = ~w; }
    void merge(const CharSet& o) { for (int i = 0; i < 4; i++) bits[i] |= o.bits[i]; }
};

// Predefined character sets for folded text
static CharSet cs_letter() {
    CharSet cs;
    cs.set(0x01);
    for (int c = 'A'; c <= 'Z'; c++) cs.set(c);
    for (int c = 'a'; c <= 'z'; c++) cs.set(c);
    return cs;
}
static CharSet cs_number() {
    CharSet cs;
    cs.set(0x02);
    for (int c = '0'; c <= '9'; c++) cs.set(c);
    return cs;
}
static CharSet cs_whitespace() {
    CharSet cs;
    cs.set(0x03); cs.set(' '); cs.set('\t'); cs.set('\n'); cs.set('\r'); cs.set('\f'); cs.set('\v');
    return cs;
}
static CharSet cs_mark() { CharSet cs; cs.set(0x05); return cs; }
static CharSet cs_punct() {
    CharSet cs;
    cs.set(0x06);
    const char* p = "!\"#%&'()*,-./:;?@[\\]_{}";
    while (*p) cs.set((uint8_t)*p++);
    return cs;
}
static CharSet cs_symbol() {
    CharSet cs;
    cs.set(0x07);
    const char* p = "$+<=>^`|~";
    while (*p) cs.set((uint8_t)*p++);
    return cs;
}
static CharSet cs_dot() {
    CharSet cs;
    for (int i = 0; i < 4; i++) cs.bits[i] = ~0ULL;
    cs.bits['\n' >> 6] &= ~(1ULL << ('\n' & 63));
    return cs;
}

// Regex AST node
struct RNode {
    enum Type { CS, GRP, NEG_LA };
    Type type = CS;
    CharSet cs;
    std::vector<std::vector<RNode>> alts;
    int qmin = 1, qmax = 1; // quantifier: -1 = unlimited
};

// Compiled pattern: top-level alternation of branches
struct CompiledPattern {
    std::vector<std::vector<RNode>> branches;
};

// Map a Unicode codepoint to its folded single-byte representation (matching fold_text)
static uint8_t cp_to_folded(int32_t cp) {
    if (cp < 0x80) return (uint8_t)cp;
    if (is_letter(cp)) return 0x01;
    if (is_number(cp)) return 0x02;
    if (is_whitespace(cp)) return 0x03;
    if (is_mark(cp)) return 0x05;
    Category cat = get_category(cp);
    if (cat >= CAT_Pc && cat <= CAT_Po) return 0x06;
    if (cat >= CAT_Sm && cat <= CAT_So) return 0x07;
    return 0x04;
}

// Pattern parser: recursive descent parser for pretokenize regex patterns
class PatternParser {
    const std::string& pat_;
    size_t pos_;
    bool icase_;
    bool supported_;

    char peek() const { return pos_ < pat_.size() ? pat_[pos_] : 0; }
    char next() { return pos_ < pat_.size() ? pat_[pos_++] : 0; }
    bool at_end() const { return pos_ >= pat_.size(); }

    // Read a full UTF-8 codepoint from the pattern, returns the codepoint
    int32_t read_cp() {
        if (pos_ >= pat_.size()) return 0;
        int32_t cp;
        int r = utf8_decode((const uint8_t*)pat_.data() + pos_, pat_.size() - pos_, &cp);
        pos_ += r;
        return cp;
    }

    // Peek at the next codepoint without consuming
    int32_t peek_cp() const {
        if (pos_ >= pat_.size()) return 0;
        int32_t cp;
        utf8_decode((const uint8_t*)pat_.data() + pos_, pat_.size() - pos_, &cp);
        return cp;
    }

    // Check if current position starts a multi-byte UTF-8 sequence
    bool is_multibyte() const {
        return pos_ < pat_.size() && (uint8_t)pat_[pos_] >= 0x80;
    }

    static int hex_val(char h) {
        if (h >= '0' && h <= '9') return h - '0';
        if (h >= 'a' && h <= 'f') return h - 'a' + 10;
        if (h >= 'A' && h <= 'F') return h - 'A' + 10;
        return 0;
    }

    CharSet prop_charset(const std::string& prop) {
        if (prop.size() >= 1 && prop[0] == 'L') return cs_letter();
        if (prop.size() >= 1 && prop[0] == 'N') return cs_number();
        if (prop.size() >= 1 && prop[0] == 'M') return cs_mark();
        if (prop.size() >= 1 && prop[0] == 'P') return cs_punct();
        if (prop.size() >= 1 && prop[0] == 'S') return cs_symbol();
        if (prop.size() >= 1 && prop[0] == 'Z') return cs_whitespace();
        return CharSet();
    }

    void add_literal(CharSet& cs, char c) {
        if (icase_ && c >= 'a' && c <= 'z') { cs.set(c); cs.set(c - 32); }
        else if (icase_ && c >= 'A' && c <= 'Z') { cs.set(c); cs.set(c + 32); }
        else cs.set((uint8_t)c);
    }

    void add_escape_to_cs(CharSet& cs) {
        next(); // skip backslash
        char c = next();
        switch (c) {
            case 'p': case 'P': {
                bool neg = (c == 'P');
                if (peek() == '{') {
                    next();
                    std::string prop;
                    while (!at_end() && peek() != '}') prop += next();
                    if (peek() == '}') next();
                    CharSet pcs = prop_charset(prop);
                    if (neg) pcs.negate();
                    cs.merge(pcs);
                }
                break;
            }
            case 's': cs.merge(cs_whitespace()); break;
            case 'S': { CharSet t = cs_whitespace(); t.negate(); cs.merge(t); break; }
            case 'd': cs.merge(cs_number()); break;
            case 'D': { CharSet t = cs_number(); t.negate(); cs.merge(t); break; }
            case 'w': { cs.merge(cs_letter()); cs.merge(cs_number()); cs.set('_'); break; }
            case 'W': { CharSet t = cs_letter(); t.merge(cs_number()); t.set('_'); t.negate(); cs.merge(t); break; }
            case 'r': cs.set('\r'); break;
            case 'n': cs.set('\n'); break;
            case 't': cs.set('\t'); break;
            case 'f': cs.set('\f'); break;
            case 'v': cs.set('\v'); break;
            case 'x': {
                if (pos_ + 1 < pat_.size()) {
                    char h1 = next(), h2 = next();
                    cs.set((uint8_t)(hex_val(h1) * 16 + hex_val(h2)));
                }
                break;
            }
            case 'b': case 'B': case 'A': case 'Z':
            case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                // Unsupported zero-width assertions / backreferences
                supported_ = false;
                add_literal(cs, c);
                break;
            default: add_literal(cs, c); break;
        }
    }

    RNode parse_class() {
        next(); // skip '['
        bool negated = (peek() == '^');
        if (negated) next();
        CharSet cs;
        // Handle ']' as first char in class (literal)
        if (peek() == ']') { cs.set((uint8_t)']'); next(); }
        while (!at_end() && peek() != ']') {
            if (peek() == '\\') {
                add_escape_to_cs(cs);
            } else if (is_multibyte()) {
                // Multi-byte UTF-8 character in class
                int32_t cp = read_cp();
                uint8_t fb = cp_to_folded(cp);
                if (peek() == '-' && pos_ + 1 < pat_.size() && pat_[pos_ + 1] != ']') {
                    next(); // skip '-'
                    int32_t end_cp;
                    if (is_multibyte()) {
                        end_cp = read_cp();
                    } else {
                        end_cp = (int32_t)(uint8_t)next();
                    }
                    uint8_t fb_end = cp_to_folded(end_cp);
                    // Set all folded bytes in the range
                    for (int b = (fb < fb_end ? fb : fb_end); b <= (fb < fb_end ? fb_end : fb); b++) {
                        cs.set((uint8_t)b);
                    }
                } else {
                    cs.set(fb);
                }
            } else {
                char c = next();
                if (peek() == '-' && pos_ + 1 < pat_.size() && pat_[pos_ + 1] != ']') {
                    next(); // skip '-'
                    char end_c;
                    if (peek() == '\\') {
                        // Escaped range end
                        next(); end_c = next();
                        if (end_c == 'n') end_c = '\n';
                        else if (end_c == 'r') end_c = '\r';
                        else if (end_c == 't') end_c = '\t';
                    } else if (is_multibyte()) {
                        int32_t end_cp = read_cp();
                        uint8_t fb = cp_to_folded((int32_t)(uint8_t)c);
                        uint8_t fb_end = cp_to_folded(end_cp);
                        for (int b = (fb < fb_end ? fb : fb_end); b <= (fb < fb_end ? fb_end : fb); b++) {
                            cs.set((uint8_t)b);
                        }
                        continue;
                    } else {
                        end_c = next();
                    }
                    for (int b = (uint8_t)c; b <= (uint8_t)end_c; b++) {
                        add_literal(cs, (char)b);
                    }
                } else {
                    add_literal(cs, c);
                }
            }
        }
        if (peek() == ']') next();
        if (negated) cs.negate();
        RNode n; n.type = RNode::CS; n.cs = cs;
        return n;
    }

    RNode parse_group() {
        next(); // skip '('
        RNode n;
        if (peek() == '?') {
            next();
            if (peek() == '!') {
                next();
                n.type = RNode::NEG_LA;
                n.alts = parse_alt();
                if (peek() == ')') next();
                return n;
            }
            if (peek() == '=' || peek() == '<') {
                // (?=...) positive lookahead, (?<=...) lookbehind, (?<name>...) named group
                supported_ = false;
                // Skip to matching ')' to avoid parse confusion
                int depth = 1;
                while (!at_end() && depth > 0) {
                    char ch = next();
                    if (ch == '(') depth++;
                    else if (ch == ')') depth--;
                }
                return n; // return empty node
            }
            bool old_icase = icase_;
            // Parse flags before ':'
            while (!at_end() && peek() != ':' && peek() != ')') {
                if (peek() == 'i') icase_ = true;
                next();
            }
            if (peek() == ':') next();
            n.type = RNode::GRP;
            n.alts = parse_alt();
            if (peek() == ')') next();
            icase_ = old_icase;
        } else {
            n.type = RNode::GRP;
            n.alts = parse_alt();
            if (peek() == ')') next();
        }
        return n;
    }

    RNode parse_atom() {
        char c = peek();
        if (c == '(') return parse_group();
        if (c == '[') return parse_class();
        if (c == '\\') {
            RNode n; n.type = RNode::CS;
            add_escape_to_cs(n.cs);
            return n;
        }
        if (c == '.') {
            next();
            RNode n; n.type = RNode::CS; n.cs = cs_dot();
            return n;
        }
        // Literal character (handle multi-byte UTF-8)
        if (is_multibyte()) {
            int32_t cp = read_cp();
            RNode n; n.type = RNode::CS;
            n.cs.set(cp_to_folded(cp));
            return n;
        }
        next();
        RNode n; n.type = RNode::CS;
        add_literal(n.cs, c);
        return n;
    }

    void parse_quantifier(RNode& node) {
        if (at_end()) return;
        char c = peek();
        if (c == '+') { next(); node.qmin = 1; node.qmax = -1; }
        else if (c == '*') { next(); node.qmin = 0; node.qmax = -1; }
        else if (c == '?') { next(); node.qmin = 0; node.qmax = 1; }
        else if (c == '{') {
            next();
            std::string num;
            while (!at_end() && peek() >= '0' && peek() <= '9') num += next();
            node.qmin = num.empty() ? 0 : std::stoi(num);
            if (peek() == ',') {
                next();
                std::string num2;
                while (!at_end() && peek() >= '0' && peek() <= '9') num2 += next();
                node.qmax = num2.empty() ? -1 : std::stoi(num2);
            } else {
                node.qmax = node.qmin;
            }
            if (peek() == '}') next();
        }
        else return;
        // Skip lazy '?' modifier (we always match greedy)
        if (!at_end() && peek() == '?') next();
    }

    std::vector<RNode> parse_branch() {
        std::vector<RNode> nodes;
        while (!at_end() && peek() != '|' && peek() != ')') {
            nodes.push_back(parse_atom());
            parse_quantifier(nodes.back());
        }
        return nodes;
    }

    std::vector<std::vector<RNode>> parse_alt() {
        std::vector<std::vector<RNode>> alts;
        alts.push_back(parse_branch());
        while (peek() == '|') {
            next();
            alts.push_back(parse_branch());
        }
        return alts;
    }

public:
    PatternParser(const std::string& pat) : pat_(pat), pos_(0), icase_(false), supported_(true) {}

    // Parse top-level alternation, skipping unsupported branches with a warning
    std::vector<std::vector<RNode>> parse() {
        std::vector<std::vector<RNode>> alts;
        // Parse first branch
        supported_ = true;
        auto branch = parse_branch();
        if (supported_) {
            alts.push_back(std::move(branch));
        } else {
            // Extract the branch text for warning (approximate: from start to current '|' or end)
            printf("[Tokenizer] WARNING: skipping unsupported regex branch in pattern\n");
        }
        // Parse remaining branches
        while (peek() == '|') {
            next();
            supported_ = true;
            branch = parse_branch();
            if (supported_) {
                alts.push_back(std::move(branch));
            } else {
                printf("[Tokenizer] WARNING: skipping unsupported regex branch in pattern\n");
            }
        }
        return alts;
    }
};

// ==========================================
// Matcher: greedy with backtracking
// ==========================================

static int match_seq(const char* text, int len, int pos,
                     const std::vector<RNode>& nodes, int idx);

static int try_branches(const char* text, int len, int pos,
                        const std::vector<std::vector<RNode>>& alts) {
    for (const auto& branch : alts) {
        int r = match_seq(text, len, pos, branch, 0);
        if (r >= 0) return r;
    }
    return -1;
}

static int match_seq(const char* text, int len, int pos,
                     const std::vector<RNode>& nodes, int idx) {
    if (idx >= (int)nodes.size()) return pos;
    const RNode& node = nodes[idx];

    if (node.type == RNode::NEG_LA) {
        // Zero-width negative lookahead: fails if any alt matches
        if (try_branches(text, len, pos, node.alts) >= 0) return -1;
        return match_seq(text, len, pos, nodes, idx + 1);
    }

    if (node.type == RNode::CS) {
        int qmax = node.qmax < 0 ? (len - pos) : node.qmax;
        // Greedily match as many as possible
        int count = 0;
        while (count < qmax && pos + count < len && node.cs.test((uint8_t)text[pos + count])) {
            count++;
        }
        // Backtrack from max to min
        for (int c = count; c >= node.qmin; c--) {
            int r = match_seq(text, len, pos + c, nodes, idx + 1);
            if (r >= 0) return r;
        }
        return -1;
    }

    if (node.type == RNode::GRP) {
        int qmax = node.qmax < 0 ? 10000 : node.qmax;
        // Greedily collect group repetitions
        std::vector<int> positions;
        positions.push_back(pos);
        int p = pos;
        int reps = 0;
        while (reps < qmax) {
            int r = try_branches(text, len, p, node.alts);
            if (r <= p) break; // must consume something to avoid infinite loop
            p = r;
            reps++;
            positions.push_back(p);
        }
        // Backtrack from max reps to min
        for (int c = reps; c >= node.qmin; c--) {
            int r = match_seq(text, len, positions[c], nodes, idx + 1);
            if (r >= 0) return r;
        }
        return -1;
    }

    return -1;
}

} // anonymous namespace

// ==========================================
// Universal regex scanner
// ==========================================

// GPT-2 pattern used as fallback when all branches are unsupported
static const char* kGPT2Pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// Check if pattern contains non-ASCII bytes (Unicode literal ranges like [一-龥])
// AND is safe for std::wregex (no \p{} Unicode properties which wregex doesn't support)
static bool use_wregex(const std::string& pattern) {
    bool has_nonascii = false;
    for (unsigned char c : pattern) {
        if (c >= 0x80) { has_nonascii = true; break; }
    }
    if (!has_nonascii) return false;
    // \p{} not supported by std::wregex, fall back to custom engine
    if (pattern.find("\\p{") != std::string::npos || pattern.find("\\P{") != std::string::npos) {
        return false;
    }
    return true;
}

static const CompiledPattern& get_compiled(const std::string& pattern) {
    static std::unordered_map<std::string, CompiledPattern> cache;
    auto it = cache.find(pattern);
    if (it == cache.end()) {
        PatternParser parser(pattern);
        CompiledPattern cp;
        cp.branches = parser.parse();
        if (cp.branches.empty()) {
            printf("[Tokenizer] WARNING: all branches unsupported in pattern, falling back to GPT-2 pattern\n");
            auto fallback_it = cache.find(kGPT2Pattern);
            if (fallback_it == cache.end()) {
                PatternParser gpt2_parser(kGPT2Pattern);
                CompiledPattern gpt2_cp;
                gpt2_cp.branches = gpt2_parser.parse();
                fallback_it = cache.emplace(kGPT2Pattern, std::move(gpt2_cp)).first;
            }
            cp.branches = fallback_it->second.branches;
        }
        it = cache.emplace(pattern, std::move(cp)).first;
    }
    return it->second;
}

// Convert UTF-8 string to wstring, tracking byte offsets per wchar
static std::wstring utf8_to_wstring(const std::string& s, std::vector<size_t>& byte_offsets) {
    std::wstring ws;
    byte_offsets.clear();
    const uint8_t* p = (const uint8_t*)s.data();
    size_t len = s.size(), i = 0;
    while (i < len) {
        byte_offsets.push_back(i);
        int32_t cp;
        int r = utf8_decode(p + i, len - i, &cp);
        ws += (wchar_t)cp;
        i += r;
    }
    byte_offsets.push_back(len); // sentinel
    return ws;
}

static std::wstring utf8_to_wstring(const std::string& s) {
    std::vector<size_t> unused;
    return utf8_to_wstring(s, unused);
}

// Get cached std::wregex for patterns with non-ASCII content (codepoint-level matching)
static const std::wregex& get_wregex(const std::string& pattern) {
    static std::unordered_map<std::string, std::wregex> cache;
    auto it = cache.find(pattern);
    if (it == cache.end()) {
        std::wstring wpat = utf8_to_wstring(pattern);
        it = cache.emplace(pattern, std::wregex(wpat)).first;
    }
    return it->second;
}

// Find all non-overlapping matches using std::wregex, returns (byte_start, byte_end) pairs
static std::vector<std::pair<size_t, size_t>> unicode_regex_find_all(const std::string& text, const std::string& pattern) {
    std::vector<size_t> byte_offsets;
    std::wstring wtext = utf8_to_wstring(text, byte_offsets);
    const auto& re = get_wregex(pattern);

    std::vector<std::pair<size_t, size_t>> matches;
    auto begin = std::wsregex_iterator(wtext.begin(), wtext.end(), re);
    auto end_it = std::wsregex_iterator();
    for (auto it = begin; it != end_it; ++it) {
        size_t ws = (size_t)it->position();
        size_t we = ws + (size_t)it->length();
        matches.push_back({byte_offsets[ws], byte_offsets[we]});
    }
    return matches;
}

// Apply split behavior on segments
struct Seg { std::string str; bool is_match; };

static std::vector<std::string> apply_behavior(std::vector<Seg>& segs, bool invert, const std::string& behavior) {
    if (invert) {
        for (auto& seg : segs) seg.is_match = !seg.is_match;
    }
    std::vector<std::string> result;
    if (behavior == "Removed") {
        for (const auto& seg : segs) {
            if (!seg.is_match && !seg.str.empty()) result.push_back(seg.str);
        }
    } else if (behavior == "MergedWithPrevious") {
        for (const auto& seg : segs) {
            if (seg.is_match && !result.empty()) {
                result.back() += seg.str;
            } else if (!seg.str.empty()) {
                result.push_back(seg.str);
            }
        }
    } else if (behavior == "MergedWithNext") {
        std::string pending;
        for (const auto& seg : segs) {
            if (seg.is_match) {
                pending += seg.str;
            } else {
                result.push_back(pending + seg.str);
                pending.clear();
            }
        }
        if (!pending.empty()) {
            if (!result.empty()) result.back() += pending;
            else result.push_back(pending);
        }
    } else {
        // "Isolated" or default
        for (const auto& seg : segs) {
            if (!seg.str.empty()) result.push_back(seg.str);
        }
    }
    return result;
}

std::vector<std::string> regex_scanner(const std::string& text, const std::string& pattern) {
    if (text.empty()) return {};

    // Non-ASCII pattern: use std::regex for accurate Unicode range matching
    if (use_wregex(pattern)) {
        auto matches = unicode_regex_find_all(text, pattern);
        std::vector<std::string> result;
        size_t prev = 0;
        for (const auto& m : matches) {
            // Emit unmatched chars individually
            while (prev < m.first) {
                int32_t cp;
                int r = utf8_decode((const uint8_t*)text.data() + prev, text.size() - prev, &cp);
                result.push_back(text.substr(prev, r));
                prev += r;
            }
            result.push_back(text.substr(m.first, m.second - m.first));
            prev = m.second;
        }
        while (prev < text.size()) {
            int32_t cp;
            int r = utf8_decode((const uint8_t*)text.data() + prev, text.size() - prev, &cp);
            result.push_back(text.substr(prev, r));
            prev += r;
        }
        return result;
    }

    const auto& branches = get_compiled(pattern).branches;
    std::vector<size_t> byte_offsets;
    std::string folded = fold_text(text, byte_offsets);

    std::vector<std::string> result;
    int pos = 0, flen = (int)folded.size();
    while (pos < flen) {
        int end = try_branches(folded.c_str(), flen, pos, branches);
        if (end > pos) {
            result.push_back(text.substr(byte_offsets[pos], byte_offsets[end] - byte_offsets[pos]));
            pos = end;
        } else {
            result.push_back(text.substr(byte_offsets[pos], byte_offsets[pos + 1] - byte_offsets[pos]));
            pos++;
        }
    }
    return result;
}

std::vector<std::string> regex_split(const std::string& text, const std::string& pattern,
                                      bool invert, const std::string& behavior) {
    if (text.empty()) return {};

    // Build segments from match positions
    std::vector<Seg> segs;

    if (use_wregex(pattern)) {
        // Non-ASCII pattern: use std::regex
        auto matches = unicode_regex_find_all(text, pattern);
        size_t prev = 0;
        for (const auto& m : matches) {
            if (m.first > prev) {
                segs.push_back({text.substr(prev, m.first - prev), false});
            }
            segs.push_back({text.substr(m.first, m.second - m.first), true});
            prev = m.second;
        }
        if (prev < text.size()) {
            segs.push_back({text.substr(prev), false});
        }
    } else {
        // ASCII pattern: use custom fold engine
        const auto& branches = get_compiled(pattern).branches;
        std::vector<size_t> byte_offsets;
        std::string folded = fold_text(text, byte_offsets);

        std::vector<std::pair<int,int>> matches;
        int pos = 0, flen = (int)folded.size();
        while (pos < flen) {
            int end = try_branches(folded.c_str(), flen, pos, branches);
            if (end > pos) {
                matches.push_back({pos, end});
                pos = end;
            } else {
                pos++;
            }
        }

        int prev = 0;
        for (const auto& m : matches) {
            if (m.first > prev) {
                segs.push_back({text.substr(byte_offsets[prev], byte_offsets[m.first] - byte_offsets[prev]), false});
            }
            segs.push_back({text.substr(byte_offsets[m.first], byte_offsets[m.second] - byte_offsets[m.first]), true});
            prev = m.second;
        }
        if (prev < flen) {
            segs.push_back({text.substr(byte_offsets[prev]), false});
        }
    }

    return apply_behavior(segs, invert, behavior);
}

// ==========================================
// Multi-string matching for added tokens
// ==========================================

std::pair<int, int> multi_string_find(const std::string& text, int start,
                                       const std::vector<std::string>& candidates) {
    int best_pos = -1;
    int best_idx = -1;
    for (int idx = 0; idx < (int)candidates.size(); idx++) {
        if (candidates[idx].empty()) continue;
        size_t pos = text.find(candidates[idx], start);
        if (pos != std::string::npos) {
            int ipos = (int)pos;
            if (best_pos == -1 || ipos < best_pos ||
                (ipos == best_pos && (int)candidates[idx].size() > (int)candidates[best_idx].size())) {
                best_pos = ipos;
                best_idx = idx;
            }
        }
    }
    return {best_pos, best_idx};
}

} // namespace Unicode
} // namespace MNN
