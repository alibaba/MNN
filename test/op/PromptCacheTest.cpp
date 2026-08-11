//
//  PromptCacheTest.cpp
//  MNN
//
//  Tests for prompt cache utilities (stripThinkBlocks).
//

#include <MNN/MNNDefine.h>
#include "MNNTestSuite.h"
#include "prompt_cache_utils.hpp"

using MNN::Transformer::stripThinkBlocks;

class PromptCacheStripThinkTest : public MNNTestCase {
public:
    virtual ~PromptCacheStripThinkTest() = default;
    virtual bool run(int precision) {
        // Test 1: basic strip
        {
            std::string text = "Hello <think>reasoning here</think> world";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "Hello  world");
        }
        // Test 2: multiple blocks
        {
            std::string text = "<think>a</think>mid<think>b</think>end";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "midend");
        }
        // Test 3: no tags
        {
            std::string text = "no tags here";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "no tags here");
        }
        // Test 4: trailing newlines after </think>
        {
            std::string text = "before<think>x</think>\n\nafter";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "beforeafter");
        }
        // Test 5: unclosed <think> (no </think>)
        {
            std::string text = "before<think>unclosed";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "before<think>unclosed");
        }
        // Test 6: empty string
        {
            std::string text = "";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "");
        }
        // Test 7: trailing \r\n mix
        {
            std::string text = "a<think>b</think>\r\nc";
            stripThinkBlocks(text);
            MNNTEST_ASSERT(text == "ac");
        }
        return true;
    }
};
MNNTestSuiteRegister(PromptCacheStripThinkTest, "op/prompt_cache");
