// Code in this file is copied and modified from
// https://wordaligned.org/articles/cpp-streambufs

#ifndef SHERPA_ONNX_CSRC_TEE_STREAM_H_
#define SHERPA_ONNX_CSRC_TEE_STREAM_H_
#include <ostream>
#include <streambuf>
#include <string>

namespace sherpa_mnn {

template <typename char_type, typename traits = std::char_traits<char_type>>
class basic_teebuf : public std::basic_streambuf<char_type, traits> {
 public:
  using int_type = typename traits::int_type;

  basic_teebuf(std::basic_streambuf<char_type, traits> *sb1,
               std::basic_streambuf<char_type, traits> *sb2)
      : sb1(sb1), sb2(sb2) {}

 private:
  int sync() override {
    int const r1 = sb1->pubsync();
    int const r2 = sb2->pubsync();
    return r1 == 0 && r2 == 0 ? 0 : -1;
  }

  int_type overflow(int_type c) override {
    int_type const eof = traits::eof();

    if (traits::eq_int_type(c, eof)) {
      return traits::not_eof(c);
    } else {
      char_type const ch = traits::to_char_type(c);
      int_type const r1 = sb1->sputc(ch);
      int_type const r2 = sb2->sputc(ch);

      return traits::eq_int_type(r1, eof) || traits::eq_int_type(r2, eof) ? eof
                                                                          : c;
    }
  }

 private:
  std::basic_streambuf<char_type, traits> *sb1;
  std::basic_streambuf<char_type, traits> *sb2;
};

using teebuf = basic_teebuf<char>;

class TeeStream : public std::ostream {
 public:
  TeeStream(std::ostream &o1, std::ostream &o2)
      : std::ostream(&tbuf), tbuf(o1.rdbuf(), o2.rdbuf()) {}

 private:
  teebuf tbuf;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_TEE_STREAM_H_
