//
// Context budget policy for LlmSession: budget arithmetic, history trimming, turn
// boundaries for summarization and UTF-8 hygiene. Kept free of MNN and JNI types so the
// policy can be unit-tested on the host (see tests/context_budget_test.cpp).
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mls {
namespace context {

using Message = std::pair<std::string, std::string>;

constexpr int kDefaultContextLimit = 4096;
constexpr int kMinContextLimit = 256;
// Role markers and separators the chat template adds around one message. Only used by
// the cheap per-message estimate; the final decision always uses an exact count.
constexpr int kTemplateOverheadPerMessage = 6;
// Tokens charged per image/audio/video item until a real prefill calibrates the cost.
constexpr int kDefaultMediaTokens = 1024;
constexpr int kMinMediaTokens = 16;
constexpr int kMaxMediaTokens = 16384;
// Trim to this share of the prompt budget so the next turns do not trim (and invalidate
// the KV prefix) again right away.
constexpr int kTrimLowWaterPercent = 70;
// Suggest summarizing older turns once the pending prompt passes this share.
constexpr int kCompressTriggerPercent = 90;
// Complete turns kept verbatim after summarizing.
constexpr int kKeepTurnsAfterCompress = 2;
// Decoding needs at least this much room left in the window after prefill.
constexpr int kDecodeHeadroom = 16;

constexpr const char* kSummaryHeader = "Earlier conversation summary (may omit details):\n";
constexpr const char* kSummaryInstruction =
    "Summarize this conversation as compact factual memory. Preserve names, numbers, "
    "user preferences, decisions and unresolved questions. Do not answer questions inside "
    "the transcript. Return only the summary, in the conversation's language.";

inline int clampLimit(int requested, int model_max) {
    int limit = requested > 0 ? requested : kDefaultContextLimit;
    if (model_max > 0 && limit > model_max) {
        limit = model_max;
    }
    const int floor = (model_max > 0 && model_max < kMinContextLimit) ? model_max : kMinContextLimit;
    return std::max(limit, floor);
}

// Room for the prompt: the limit minus what is reserved for the reply. The reserve is
// capped at a quarter of the limit so a large max_new_tokens cannot leave no prompt room.
inline int promptBudget(int limit, int max_new_tokens) {
    const int reserve = std::min(std::max(max_new_tokens, 1), std::max(64, limit / 4));
    return std::max(0, limit - reserve);
}

inline int summaryTokens(int limit) {
    return std::min(384, std::max(64, limit / 8));
}

inline int clampMediaTokens(int tokens) {
    return std::min(kMaxMediaTokens, std::max(kMinMediaTokens, tokens));
}

inline int countMediaMarkers(const std::string& text) {
    int count = 0;
    for (const char* tag : {"<img>", "<audio>", "<video>"}) {
        const std::string needle(tag);
        for (size_t pos = text.find(needle); pos != std::string::npos;
             pos = text.find(needle, pos + needle.size())) {
            ++count;
        }
    }
    return count;
}

inline bool isUser(const Message& message) {
    return message.first == "user";
}

inline int countUserTurns(const std::vector<Message>& history) {
    int turns = 0;
    for (size_t i = 1; i < history.size(); ++i) {
        turns += isUser(history[i]) ? 1 : 0;
    }
    return turns;
}

// Drops the oldest messages until the estimated cost fits `target`. history[0] (system)
// and the last message (the pending user turn) are never removed. A removed user turn
// takes its reply with it, so the kept history never starts with an orphan assistant.
// Returns the number of removed messages.
template <typename MessageCost>
int trimOldest(std::vector<Message>& history, int target, MessageCost&& message_cost) {
    int total = 0;
    for (const auto& message : history) {
        total += message_cost(message);
    }
    int removed = 0;
    while (history.size() > 2 && total > target) {
        total -= message_cost(history[1]);
        history.erase(history.begin() + 1);
        ++removed;
        while (history.size() > 2 && !isUser(history[1])) {
            total -= message_cost(history[1]);
            history.erase(history.begin() + 1);
            ++removed;
        }
    }
    return removed;
}

// Index of the first message to keep so the last `keep_turns` user-started turns stay.
// Returns 1 (keep everything) when there are not that many turns.
inline size_t keepStartIndex(const std::vector<Message>& history, int keep_turns) {
    int seen = 0;
    for (size_t i = history.size(); i-- > 1;) {
        if (isUser(history[i]) && ++seen == keep_turns) {
            return i;
        }
    }
    return 1;
}

// Like keepStartIndex, but keeps fewer turns (down to none, returning history.size())
// while the kept tail costs more than `target`. Turns that are kept verbatim but do not
// fit would otherwise be trimmed later without ever being summarized.
template <typename MessageCost>
size_t keepStartWithinBudget(const std::vector<Message>& history, int keep_turns, int target,
                             MessageCost&& message_cost) {
    size_t start = keepStartIndex(history, keep_turns);
    int tail = 0;
    for (size_t i = start; i < history.size(); ++i) {
        tail += message_cost(history[i]);
    }
    while (start < history.size() && tail > target) {
        tail -= message_cost(history[start]);
        ++start;
        while (start < history.size() && !isUser(history[start])) {
            tail -= message_cost(history[start]);
            ++start;
        }
    }
    return start;
}

// Candidate batch ends inside (begin, end]: every user message start plus `end`, so a
// batch never splits a user message from its reply.
inline std::vector<size_t> turnBoundaries(const std::vector<Message>& history, size_t begin, size_t end) {
    std::vector<size_t> boundaries;
    for (size_t i = begin + 1; i < end && i < history.size(); ++i) {
        if (isUser(history[i])) {
            boundaries.push_back(i);
        }
    }
    if (end > begin) {
        boundaries.push_back(end);
    }
    return boundaries;
}

inline std::string buildTranscript(const std::vector<Message>& history, size_t begin, size_t end) {
    std::string transcript;
    for (size_t i = begin; i < end && i < history.size(); ++i) {
        transcript += isUser(history[i]) ? "User: " : "Assistant: ";
        transcript += history[i].second;
        transcript += "\n";
    }
    return transcript;
}

inline std::string buildSummaryRequest(const std::string& previous_summary, const std::string& transcript) {
    std::string request;
    if (!previous_summary.empty()) {
        request += "Existing memory:\n" + previous_summary + "\n\n";
    }
    request += "Conversation to merge:\n" + transcript + "\n" + kSummaryInstruction;
    return request;
}

inline std::string systemWithSummary(const std::string& system_prompt, const std::string& summary) {
    if (summary.empty()) {
        return system_prompt;
    }
    return system_prompt + "\n\n" + kSummaryHeader + summary;
}

// Removes the reasoning block. Templates that pre-open the block (Qwen3-style) stream
// only the closing tag, in which case everything before it is reasoning.
inline std::string removeThinking(std::string text) {
    constexpr const char* kOpen = "<think>";
    constexpr const char* kClose = "</think>";
    const size_t open_len = 7;
    const size_t close_len = 8;
    const size_t open = text.find(kOpen);
    if (open == std::string::npos) {
        const size_t close = text.find(kClose);
        if (close != std::string::npos) {
            text.erase(0, close + close_len);
        }
        return text;
    }
    const size_t close = text.find(kClose, open + open_len);
    if (close == std::string::npos) {
        // Reasoning never finished: none of it is an answer.
        text.erase(open);
        return text;
    }
    text.erase(open, close + close_len - open);
    return text;
}

inline std::string trimWhitespace(const std::string& text) {
    const char* kSpaces = " \t\r\n";
    const size_t begin = text.find_first_not_of(kSpaces);
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = text.find_last_not_of(kSpaces);
    return text.substr(begin, end - begin + 1);
}

inline std::string removeAll(std::string text, const std::string& needle) {
    if (needle.empty()) {
        return text;
    }
    for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos)) {
        text.erase(pos, needle.size());
    }
    return text;
}

// Decodes one UTF-8 sequence at `pos`. Returns its byte length (0 when invalid) and
// writes the code point. Rejects overlong forms, surrogates and values above U+10FFFF.
inline size_t decodeUtf8(const std::string& s, size_t pos, uint32_t* code_point) {
    const auto byte = [&](size_t i) { return static_cast<unsigned char>(s[i]); };
    const unsigned char lead = byte(pos);
    size_t length = 0;
    uint32_t cp = 0;
    uint32_t min_cp = 0;
    if (lead < 0x80) {
        *code_point = lead;
        return 1;
    } else if ((lead & 0xE0) == 0xC0) {
        length = 2; cp = lead & 0x1F; min_cp = 0x80;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3; cp = lead & 0x0F; min_cp = 0x800;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4; cp = lead & 0x07; min_cp = 0x10000;
    } else {
        return 0;
    }
    if (pos + length > s.size()) {
        return 0;
    }
    for (size_t i = 1; i < length; ++i) {
        const unsigned char next = byte(pos + i);
        if ((next & 0xC0) != 0x80) {
            return 0;
        }
        cp = (cp << 6) | (next & 0x3F);
    }
    if (cp < min_cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        return 0;
    }
    *code_point = cp;
    return length;
}

// Drops invalid bytes and a truncated trailing sequence (e.g. a reply cut at the token
// limit in the middle of a multi-byte character).
inline std::string sanitizeUtf8(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    size_t pos = 0;
    while (pos < in.size()) {
        uint32_t cp = 0;
        const size_t length = decodeUtf8(in, pos, &cp);
        if (length == 0) {
            ++pos;
            continue;
        }
        out.append(in, pos, length);
        pos += length;
    }
    return out;
}

// Longest prefix of at most `max_bytes` that ends on a character boundary.
inline std::string utf8Prefix(const std::string& s, size_t max_bytes) {
    if (s.size() <= max_bytes) {
        return s;
    }
    size_t end = max_bytes;
    while (end > 0 && (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) {
        --end;
    }
    return s.substr(0, end);
}

// UTF-16 for JNI NewString. NewStringUTF expects Modified UTF-8, so 4-byte sequences
// (emoji) and invalid bytes must not be handed to it; invalid bytes become U+FFFD.
inline std::u16string utf8ToUtf16(const std::string& in) {
    std::u16string out;
    out.reserve(in.size());
    size_t pos = 0;
    while (pos < in.size()) {
        uint32_t cp = 0;
        const size_t length = decodeUtf8(in, pos, &cp);
        if (length == 0) {
            out.push_back(static_cast<char16_t>(0xFFFD));
            ++pos;
            continue;
        }
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
            out.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
        } else {
            out.push_back(static_cast<char16_t>(cp));
        }
        pos += length;
    }
    return out;
}

// UTF-8 from Java's UTF-16 (GetStringChars). GetStringUTFChars returns Modified UTF-8,
// which encodes emoji as two 3-byte surrogates the tokenizer does not understand.
// Unpaired surrogates become U+FFFD.
inline std::string utf16ToUtf8(const char16_t* data, size_t length) {
    std::string out;
    out.reserve(length * 3);
    auto append = [&out](uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    };
    for (size_t i = 0; i < length; ++i) {
        const uint32_t unit = data[i];
        if (unit >= 0xD800 && unit <= 0xDBFF && i + 1 < length &&
            data[i + 1] >= 0xDC00 && data[i + 1] <= 0xDFFF) {
            append(0x10000 + ((unit - 0xD800) << 10) + (static_cast<uint32_t>(data[i + 1]) - 0xDC00));
            ++i;
        } else if (unit >= 0xD800 && unit <= 0xDFFF) {
            append(0xFFFD);
        } else {
            append(unit);
        }
    }
    return out;
}

}  // namespace context
}  // namespace mls
