// sherpa-mnn/csrc/text-utils-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/text-utils.h"

#include "gtest/gtest.h"

namespace sherpa_mnn {

TEST(ToLowerCase, WideString) {
  std::string text =
      "Hallo! Übeltäter übergibt Ärzten öfters äußerst ätzende Öle 3€";
  auto t = ToLowerCase(text);
  std::cout << text << "\n";
  std::cout << t << "\n";
}

TEST(RemoveInvalidUtf8Sequences, Case1) {
  std::vector<uint8_t> v = {
      0xe4, 0xbb, 0x8a,                                  // 今
      0xe5, 0xa4, 0xa9,                                  // 天
      'i',  's',  ' ',  'M', 'o', 'd', 'a', 'y',  ',',   // is Monday,
      ' ',  'w',  'i',  'e', ' ', 'h', 'e', 'i',  0xc3,  // wie heißen Size
      0x9f, 'e',  'n',  ' ', 'S', 'i', 'e', 0xf0, 0x9d, 0x84, 0x81};

  std::vector<uint8_t> v0 = v;
  v0[1] = 0xc0;  // make the first 3 bytes an invalid utf8 character
  std::string s0{v0.begin(), v0.end()};
  EXPECT_EQ(s0.size(), v0.size());

  auto s = RemoveInvalidUtf8Sequences(s0);  // should remove 今

  v0 = v;
  // v0[23] == 0xc3
  // v0[24] == 0x9f

  v0[23] = 0xc1;

  s0 = {v0.begin(), v0.end()};
  s = RemoveInvalidUtf8Sequences(s0);  // should remove ß

  EXPECT_EQ(s.size() + 2, v.size());

  v0 = v;
  // v0[31] = 0xf0;
  // v0[32] = 0x9d;
  // v0[33] = 0x84;
  // v0[34] = 0x81;
  v0[31] = 0xf5;

  s0 = {v0.begin(), v0.end()};
  s = RemoveInvalidUtf8Sequences(s0);

  EXPECT_EQ(s.size() + 4, v.size());
}


// Tests for sanitizeUtf8
TEST(RemoveInvalidUtf8Sequences, ValidUtf8StringPassesUnchanged) {
  std::string input = "Valid UTF-8 🌍";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), input);
}

TEST(RemoveInvalidUtf8Sequences, SingleInvalidByteReplaced) {
  std::string input = "Invalid \xFF UTF-8";
  std::string expected = "Invalid  UTF-8";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, TruncatedUtf8SequenceReplaced) {
  std::string input = "Broken \xE2\x82";  // Incomplete UTF-8 sequence
  std::string expected = "Broken ";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, MultipleInvalidBytes) {
  std::string input = "Test \xC0\xC0\xF8\xA0";  // Multiple invalid sequences
  std::string expected = "Test ";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, BreakingCase_SpaceFollowedByInvalidByte) {
  std::string input = "\x20\xC4";  // Space followed by an invalid byte
  std::string expected = " ";  // 0xC4 removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, ValidUtf8WithEdgeCaseCharacters) {
  std::string input = "Edge 🏆💯";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), input);
}

TEST(RemoveInvalidUtf8Sequences, MixedValidAndInvalidBytes) {
  std::string input = "Mix \xE2\x82\xAC \xF0\x9F\x98\x81 \xFF";
  std::string expected = "Mix € 😁 ";  // Invalid bytes removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, SpaceFollowedByInvalidByte) {
  std::string input = "\x20\xC4";  // Space (0x20) followed by invalid (0xC4)
  std::string expected = " ";  // Space remains, 0xC4 is removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, RemoveTruncatedC4) {
  std::string input = "Hello \xc4 world";  // Invalid `0xC4`
  std::string expected = "Hello  world";  // `0xC4` should be removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, SpaceFollowedByInvalidByte_Breaking) {
  std::string input = "\x20\xc4";  // Space followed by invalid `0xc4`
  std::string expected = " ";  // `0xc4` should be removed, space remains
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, DebugSpaceFollowedByInvalidByte) {
  std::string input = "\x20\xc4";  // Space followed by invalid `0xc4`
  std::string output = RemoveInvalidUtf8Sequences(input);

  std::cout << "Processed string: ";
  for (unsigned char c : output) {
    printf("\\x%02x ", c);
  }
  std::cout << std::endl;

  EXPECT_EQ(output, " ");  // Expect `0xc4` to be removed, leaving only space
}

}  // namespace sherpa_mnn
