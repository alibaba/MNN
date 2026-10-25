// Host unit tests for context_budget.hpp. Build and run:
//   c++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
//       -I.. context_budget_test.cpp -o /tmp/context_budget_test && /tmp/context_budget_test
#include "context_budget.hpp"

#include <cstdio>
#include <cstdlib>

using namespace mls::context;

namespace {
int g_failures = 0;

#define EXPECT_TRUE(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "%s:%d: EXPECT_TRUE(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)
#define EXPECT_EQ(a, b) EXPECT_TRUE((a) == (b))

std::vector<Message> conversation(int turns) {
    std::vector<Message> h{{"system", "sys"}};
    for (int i = 0; i < turns; ++i) {
        h.emplace_back("user", "u" + std::to_string(i));
        h.emplace_back("assistant", "a" + std::to_string(i));
    }
    return h;
}

int unitCost(const Message&) { return 10; }

void testBudgetArithmetic() {
    EXPECT_EQ(clampLimit(0, 0), kDefaultContextLimit);
    EXPECT_EQ(clampLimit(8192, 4096), 4096);
    EXPECT_EQ(clampLimit(100, 0), kMinContextLimit);
    EXPECT_EQ(clampLimit(100, 128), 128);  // a tiny model window is respected
    EXPECT_EQ(promptBudget(4096, 2048), 3072);  // reserve capped at limit / 4
    EXPECT_EQ(promptBudget(4096, 128), 3968);
    EXPECT_EQ(promptBudget(256, 0), 255);
    EXPECT_TRUE(promptBudget(256, 100000) > 0);
    EXPECT_EQ(summaryTokens(1024), 128);
    EXPECT_EQ(summaryTokens(256), 64);
    EXPECT_EQ(summaryTokens(65536), 384);
    EXPECT_EQ(clampMediaTokens(0), kMinMediaTokens);
    EXPECT_EQ(clampMediaTokens(1 << 30), kMaxMediaTokens);
}

void testMediaMarkers() {
    EXPECT_EQ(countMediaMarkers("hi"), 0);
    EXPECT_EQ(countMediaMarkers("<img>a</img><img>b</img> <audio>c</audio><video>d</video>"), 4);
}

void testTrimKeepsSystemAndPending() {
    auto h = conversation(3);
    h.emplace_back("user", "pending");
    const int removed = trimOldest(h, 0, unitCost);
    EXPECT_EQ(removed, 6);
    EXPECT_EQ(h.size(), 2u);
    EXPECT_EQ(h[0].first, "system");
    EXPECT_EQ(h[1].second, "pending");
}

void testTrimRemovesWholeTurns() {
    auto h = conversation(3);  // 7 messages
    h.emplace_back("user", "pending");  // cost 80
    trimOldest(h, 60, unitCost);
    EXPECT_EQ(h.size(), 6u);  // dropped u0 + a0 together
    EXPECT_EQ(h[1].second, "u1");
    EXPECT_TRUE(isUser(h[1]));
}

void testTrimNeverDeletesPendingWithUnbalancedHistory() {
    // A stopped turn left a user message without a reply.
    std::vector<Message> h{{"system", "s"}, {"user", "orphan"}, {"user", "pending"}};
    trimOldest(h, 0, unitCost);
    EXPECT_EQ(h.size(), 2u);
    EXPECT_EQ(h[1].second, "pending");

    // History starting with an assistant (e.g. restored from an error row).
    std::vector<Message> g{{"system", "s"}, {"assistant", "x"}, {"user", "u"}, {"assistant", "a"},
                           {"user", "pending"}};
    trimOldest(g, 30, unitCost);
    EXPECT_TRUE(isUser(g[1]));
    EXPECT_EQ(g.back().second, "pending");
}

void testTrimNoopWhenFits() {
    auto h = conversation(2);
    h.emplace_back("user", "p");
    EXPECT_EQ(trimOldest(h, 1000, unitCost), 0);
    EXPECT_EQ(h.size(), 6u);
}

void testKeepStartIndex() {
    auto h = conversation(4);  // users at 1,3,5,7
    EXPECT_EQ(keepStartIndex(h, 2), 5u);
    EXPECT_EQ(keepStartIndex(h, 4), 1u);
    EXPECT_EQ(keepStartIndex(h, 5), 1u);
    EXPECT_EQ(keepStartIndex(conversation(0), 2), 1u);
}

void testKeepStartWithinBudget() {
    auto h = conversation(4);  // users at 1,3,5,7; cost 10 each
    EXPECT_EQ(keepStartWithinBudget(h, 2, 40, unitCost), 5u);   // two turns fit
    EXPECT_EQ(keepStartWithinBudget(h, 2, 25, unitCost), 7u);   // only the last turn fits
    EXPECT_EQ(keepStartWithinBudget(h, 2, 5, unitCost), 9u);    // nothing fits: keep none
    EXPECT_EQ(keepStartWithinBudget(conversation(1), 2, 100, unitCost), 1u);
}

void testReplaceMediaTags() {
    EXPECT_EQ(replaceMediaTags("see <img>/a/b.jpg</img> and <audio>x.wav</audio><video>v</video>!"),
              std::string("see [image] and [audio][video]!"));
    EXPECT_EQ(replaceMediaTags("broken <img>/no/close"), std::string("broken [image]"));
    EXPECT_EQ(replaceMediaTags("plain"), std::string("plain"));
    std::vector<Message> h{{"system", "s"}, {"user", "<img>p</img>what"}, {"assistant", "a"}};
    EXPECT_EQ(buildTranscript(h, 1, 3), std::string("User: [image]what\nAssistant: a\n"));
}

void testSummaryWindowStart() {
    auto h = conversation(4);  // users at 1,3,5,7; each message costs 10; end = 9
    EXPECT_EQ(summaryWindowStart(h, 9, 1000, unitCost), 1u);  // everything fits
    EXPECT_EQ(summaryWindowStart(h, 9, 40, unitCost), 5u);    // last two turns
    EXPECT_EQ(summaryWindowStart(h, 9, 25, unitCost), 7u);    // last turn only
    EXPECT_EQ(summaryWindowStart(h, 9, 5, unitCost), 7u);     // even one turn is too big
    EXPECT_EQ(summaryWindowStart(h, 1, 100, unitCost), 1u);   // nothing before end
    EXPECT_TRUE(isUser(h[summaryWindowStart(h, 7, 30, unitCost)]));
}

void testTurnBoundariesNeverSplitATurn() {
    auto h = conversation(4);
    const auto b = turnBoundaries(h, 1, 7);
    EXPECT_EQ(b.size(), 3u);
    EXPECT_EQ(b[0], 3u);
    EXPECT_EQ(b[1], 5u);
    EXPECT_EQ(b[2], 7u);
    for (size_t end : b) {
        EXPECT_TRUE(end == 7u || isUser(h[end]));
    }
    EXPECT_TRUE(turnBoundaries(h, 5, 5).empty());
}

void testTranscriptAndSummaryPrompt() {
    auto h = conversation(1);
    EXPECT_EQ(buildTranscript(h, 1, 3), std::string("User: u0\nAssistant: a0\n"));
    const std::string first = buildSummaryRequest("", "T");
    EXPECT_TRUE(first.find("Existing memory") == std::string::npos);
    EXPECT_TRUE(first.find(kSummaryInstruction) != std::string::npos);
    EXPECT_TRUE(buildSummaryRequest("M", "T").find("Existing memory:\nM") != std::string::npos);
    EXPECT_EQ(systemWithSummary("sys", ""), std::string("sys"));
    EXPECT_TRUE(systemWithSummary("sys", "S").find(kSummaryHeader) != std::string::npos);
}

void testRemoveThinking() {
    EXPECT_EQ(removeThinking("<think>r</think>answer"), std::string("answer"));
    EXPECT_EQ(removeThinking("reasoning</think>\n\nanswer"), std::string("\n\nanswer"));
    EXPECT_EQ(removeThinking("plain"), std::string("plain"));
    EXPECT_EQ(removeThinking("<think>never closed"), std::string(""));
    EXPECT_EQ(trimWhitespace("  \n x y \n"), std::string("x y"));
    EXPECT_EQ(trimWhitespace(" \n "), std::string(""));
    EXPECT_EQ(removeAll("a<eop>b<eop>", "<eop>"), std::string("ab"));
}

void testUtf8() {
    const std::string han = "\xE4\xB8\xAD";        // 中
    const std::string emoji = "\xF0\x9F\x98\x80";  // 😀
    EXPECT_EQ(sanitizeUtf8("ok" + han + emoji), "ok" + han + emoji);
    EXPECT_EQ(sanitizeUtf8("ab" + han.substr(0, 2)), std::string("ab"));  // truncated tail
    EXPECT_EQ(sanitizeUtf8("a\xC0\xAF" "b"), std::string("ab"));          // overlong
    EXPECT_EQ(sanitizeUtf8("a\xED\xA0\x80" "b"), std::string("ab"));      // surrogate
    EXPECT_EQ(sanitizeUtf8("a\xFF" "b"), std::string("ab"));
    EXPECT_EQ(utf8Prefix(han + han, 4), han);
    EXPECT_EQ(utf8Prefix("abc", 10), std::string("abc"));
    const std::u16string u16 = utf8ToUtf16("A" + emoji + "\xFF");
    EXPECT_EQ(u16.size(), 4u);
    EXPECT_EQ(u16[0], u'A');
    EXPECT_EQ(u16[1], static_cast<char16_t>(0xD83D));
    EXPECT_EQ(u16[2], static_cast<char16_t>(0xDE00));
    EXPECT_EQ(u16[3], static_cast<char16_t>(0xFFFD));
}
void testUtf16ToUtf8() {
    const char16_t text[] = {u'A', 0x4E2D, 0xD83D, 0xDE00, 0xD800, u'B'};
    EXPECT_EQ(utf16ToUtf8(text, 6),
              std::string("A\xE4\xB8\xAD\xF0\x9F\x98\x80\xEF\xBF\xBD" "B"));
    EXPECT_EQ(utf16ToUtf8(text, 0), std::string());
    const std::string round = "x\xF0\x9F\x98\x80y";
    const std::u16string u16 = utf8ToUtf16(round);
    EXPECT_EQ(utf16ToUtf8(u16.data(), u16.size()), round);
}
}  // namespace

int main() {
    testBudgetArithmetic();
    testMediaMarkers();
    testTrimKeepsSystemAndPending();
    testTrimRemovesWholeTurns();
    testTrimNeverDeletesPendingWithUnbalancedHistory();
    testTrimNoopWhenFits();
    testKeepStartIndex();
    testKeepStartWithinBudget();
    testTurnBoundariesNeverSplitATurn();
    testReplaceMediaTags();
    testSummaryWindowStart();
    testTranscriptAndSummaryPrompt();
    testRemoveThinking();
    testUtf8();
    testUtf16ToUtf8();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("context_budget_test: all passed\n");
    return EXIT_SUCCESS;
}
