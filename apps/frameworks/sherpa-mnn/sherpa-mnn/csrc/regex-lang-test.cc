// sherpa-mnn/csrc/regex-lang-test.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <regex>  // NOLINT

#include "gtest/gtest.h"
#include "sherpa-mnn/csrc/text-utils.cc"

namespace sherpa_mnn {

static void TestLang(const std::string &expr, const std::string &text,
                     const std::vector<std::string> &expected) {
  auto ws = ToWideString(text);
  std::wstring wexpr = ToWideString(expr);
  std::wregex we(wexpr);

  auto begin = std::wsregex_iterator(ws.begin(), ws.end(), we);
  auto end = std::wsregex_iterator();
  int32_t k = 0;
  for (std::wsregex_iterator i = begin; i != end; ++i) {
    std::wsmatch match = *i;
    std::wstring match_str = match.str();
    auto ms = ToString(match_str);
    std::cout << ms << "\n";
    EXPECT_EQ(ms, expected[k]);
    k++;
  }
  EXPECT_EQ(k, expected.size());
}

TEST(German, Case1) {
  std::cout << "----------Test German----------";
  // see https://character-table.netlify.app/german/
  std::string expr =
      "([\\u0020-\\u005f\\u0061-"
      "\\u007d\\u00a0\\u00a7\\u00a9\\u00ab\\u00bb\\u00c4\\u00d6\\u00dc\\u00df\\"
      "u00e4\\u00f6\\u00fc\\u2010-\\u2011\\u2013-"
      "\\u2014\\u2018\\u201a\\u201c\\u201e\\u2026\\u2030\\u20ac]+)";

  std::string text =
      "开始Übeltäter übergibt Ärzten 中间öfters äußerst ätzende Öle结束3€";

  std::vector<std::string> expected = {"Übeltäter übergibt Ärzten ",
                                       "öfters äußerst ätzende Öle", "3€"};

  TestLang(expr, text, expected);
}

TEST(French, Case1) {
  std::string expr =
      "([\\u0020-\\u005f\\u0061-"
      "\\u007a\\u007c\\u00a0\\u00a7\\u00a9\\u00ab\\u00b2-"
      "\\u00b3\\u00bb\\u00c0\\u00c2\\u00c6-\\u00cb\\u00ce-"
      "\\u00cf\\u00d4\\u00d9\\u00db-\\u00dc\\u00e0\\u00e2\\u00e6-"
      "\\u00eb\\u00ee-\\u00ef\\u00f4\\u00f9\\u00fb-\\u00fc\\u00ff\\u0152-"
      "\\u0153\\u0178\\u02b3\\u02e2\\u1d48-\\u1d49\\u2010-\\u2011\\u2013-"
      "\\u2014\\u2019\\u201c-\\u201d\\u2020-\\u2021\\u2026\\u202f-"
      "\\u2030\\u20ac\\u2212]+)";
  std::string text =
      "L'été, 一avec son ciel bleuâtre, 二est un moment où, 三Noël, maçon";
  std::vector<std::string> expected = {
      "L'été, ",
      "avec son ciel bleuâtre, ",
      "est un moment où, ",
      "Noël, maçon",
  };
  TestLang(expr, text, expected);
}

TEST(English, Case1) {
  // https://character-table.netlify.app/english/
  std::string expr =
      "([\\u0020-\\u005f\\u0061-\\u007a\\u007c\\u00a0\\u00a7\\u00a9\\u2010-"
      "\\u2011\\u2013-\\u2014\\u2018-\\u2019\\u201c-\\u201d\\u2020-"
      "\\u2021\\u2026\\u2030\\u2032-\\u2033\\u20ac]+)";
  std::string text = "一how are you doing? 二Thank you!";

  std::vector<std::string> expected = {
      "how are you doing? ",
      "Thank you!",
  };
  TestLang(expr, text, expected);
}

}  // namespace sherpa_mnn
