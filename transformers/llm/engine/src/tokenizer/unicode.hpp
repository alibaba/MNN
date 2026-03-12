//
//  unicode.hpp
//
//  Created by MNN on 2024/01/01.
//  ZhaodeWang
//

#ifndef UNICODE_HPP
#define UNICODE_HPP

#include "unicode_data.hpp"
#include <string>
#include <vector>

namespace MNN {
namespace Unicode {

// UTF-8 decode: returns bytes consumed (1-4), writes codepoint to *cp.
// On error returns 1 and writes 0xFFFD.
static inline int utf8_decode(const uint8_t* src, size_t len, int32_t* cp) {
    if (len == 0) { *cp = 0xFFFD; return 1; }
    uint8_t b0 = src[0];
    if (b0 < 0x80) { *cp = b0; return 1; }
    if ((b0 & 0xE0) == 0xC0 && len >= 2 && (src[1] & 0xC0) == 0x80) {
        *cp = ((int32_t)(b0 & 0x1F) << 6) | (src[1] & 0x3F);
        return (*cp >= 0x80) ? 2 : (*cp = 0xFFFD, 1);
    }
    if ((b0 & 0xF0) == 0xE0 && len >= 3 && (src[1] & 0xC0) == 0x80 && (src[2] & 0xC0) == 0x80) {
        *cp = ((int32_t)(b0 & 0x0F) << 12) | ((int32_t)(src[1] & 0x3F) << 6) | (src[2] & 0x3F);
        return (*cp >= 0x800 && (*cp < 0xD800 || *cp > 0xDFFF)) ? 3 : (*cp = 0xFFFD, 1);
    }
    if ((b0 & 0xF8) == 0xF0 && len >= 4 && (src[1] & 0xC0) == 0x80 && (src[2] & 0xC0) == 0x80 && (src[3] & 0xC0) == 0x80) {
        *cp = ((int32_t)(b0 & 0x07) << 18) | ((int32_t)(src[1] & 0x3F) << 12) | ((int32_t)(src[2] & 0x3F) << 6) | (src[3] & 0x3F);
        return (*cp >= 0x10000 && *cp <= 0x10FFFF) ? 4 : (*cp = 0xFFFD, 1);
    }
    *cp = 0xFFFD; return 1;
}

// UTF-8 encode: writes codepoint to buf, returns bytes written (1-4).
static inline int utf8_encode(int32_t cp, char* buf) {
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    if (cp < 0x800) { buf[0] = (char)(0xC0 | (cp >> 6)); buf[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { buf[0] = (char)(0xE0 | (cp >> 12)); buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    buf[0] = (char)(0xF0 | (cp >> 18)); buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

// Unicode property queries
Category get_category(int32_t cp);
bool is_letter(int32_t cp);
bool is_number(int32_t cp);
bool is_punctuation(int32_t cp);
bool is_whitespace(int32_t cp);
bool is_mark(int32_t cp);
int32_t to_lower(int32_t cp);

// Universal pre-tokenizer scanner: split text into tokens using any pretokenize regex pattern.
// Uses text folding (Unicode → single-byte) + lightweight byte-regex engine (no std::regex).
std::vector<std::string> regex_scanner(const std::string& text, const std::string& pattern);

// Split text by regex pattern with behavior control.
// behavior: "Isolated" | "MergedWithPrevious" | "MergedWithNext" | "Removed"
// invert: if true, swap the roles of matched and non-matched parts
std::vector<std::string> regex_split(const std::string& text, const std::string& pattern,
                                      bool invert, const std::string& behavior);

// Multi-string match: find the earliest and longest match among candidates
// Returns (position, index) or (-1, -1) if no match
std::pair<int, int> multi_string_find(const std::string& text, int start,
                                       const std::vector<std::string>& candidates);
} // namespace Unicode
} // namespace MNN

#endif // UNICODE_HPP
