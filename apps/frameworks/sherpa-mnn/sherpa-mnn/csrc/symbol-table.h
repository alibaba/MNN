// sherpa-mnn/csrc/symbol-table.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SYMBOL_TABLE_H_
#define SHERPA_ONNX_CSRC_SYMBOL_TABLE_H_

#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_mnn {

// The same token can be mapped to different integer IDs, so
// we need an id2token argument here.
std::unordered_map<std::string, int32_t> ReadTokens(
    std::istream &is,
    std::unordered_map<int32_t, std::string> *id2token = nullptr);

std::vector<int32_t> ConvertTokensToIds(
    const std::unordered_map<std::string, int32_t> &token2id,
    const std::vector<std::string> &tokens);

/// It manages mapping between symbols and integer IDs.
class SymbolTable {
 public:
  SymbolTable() = default;
  /// Construct a symbol table from a file or from a buffered string.
  /// Each line in the file contains two fields:
  ///
  ///    sym ID
  ///
  /// Fields are separated by space(s).
  explicit SymbolTable(const std::string &filename, bool is_file = true);

  template <typename Manager>
  SymbolTable(Manager *mgr, const std::string &filename);

  /// Return a string representation of this symbol table
  std::string ToString() const;

  /// Return the symbol corresponding to the given ID.
  const std::string operator[](int32_t id) const;
  /// Return the ID corresponding to the given symbol.
  int32_t operator[](const std::string &sym) const;

  /// Return true if there is a symbol with the given ID.
  bool Contains(int32_t id) const;

  /// Return true if there is a given symbol in the symbol table.
  bool Contains(const std::string &sym) const;

  // for tokens.txt from Whisper
  void ApplyBase64Decode();

  int32_t NumSymbols() const { return id2sym_.size(); }

  std::string DecodeByteBpe(const std::string &text) const;

  bool IsByteBpe() const { return is_bbpe_; }

 private:
  void Init(std::istream &is);

 private:
  std::unordered_map<std::string, int32_t> sym2id_;
  std::unordered_map<int32_t, std::string> id2sym_;
  bool is_bbpe_ = false;
};

std::ostream &operator<<(std::ostream &os, const SymbolTable &symbol_table);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SYMBOL_TABLE_H_
