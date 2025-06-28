// sherpa-mnn/csrc/utils.h
//
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_UTILS_H_
#define SHERPA_ONNX_CSRC_UTILS_H_

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/symbol-table.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_mnn {

/* Encode the hotwords in an input stream to be tokens ids.
 *
 * @param is The input stream, it contains several lines, one hotword for each
 *           line. For each hotword, the tokens (cjkchar or bpe) are separated
 *           by spaces.
 * @param symbol_table  The tokens table mapping symbols to ids. All the symbols
 *                      in the stream should be in the symbol_table, if not this
 *                      function returns fasle.
 *
 * @@param hotwords  The encoded ids to be written to.
 *
 * @return  If all the symbols from ``is`` are in the symbol_table, returns true
 *          otherwise returns false.
 */
bool EncodeHotwords(std::istream &is, const std::string &modeling_unit,
                    const SymbolTable &symbol_table,
                    const ssentencepiece::Ssentencepiece *bpe_encoder_,
                    std::vector<std::vector<int32_t>> *hotwords_id,
                    std::vector<float> *boost_scores);

/* Encode the keywords in an input stream to be tokens ids.
 *
 * @param is The input stream, it contains several lines, one hotword for each
 *           line. For each hotword, the tokens (cjkchar or bpe) are separated
 *           by spaces, it might contain boosting score (starting with :),
 *           triggering threshold (starting with #) and keyword string (starting
 *           with @) too.
 * @param symbol_table  The tokens table mapping symbols to ids. All the symbols
 *                      in the stream should be in the symbol_table, if not this
 *                      function returns fasle.
 *
 * @param keywords_id The encoded ids to be written to.
 * @param keywords The original keyword string to be written to.
 * @param boost_scores  The boosting score for each keyword to be written to.
 * @param threshold  The triggering threshold for each keyword to be written to.
 *
 * @return  If all the symbols from ``is`` are in the symbol_table, returns true
 *          otherwise returns false.
 */
bool EncodeKeywords(std::istream &is, const SymbolTable &symbol_table,
                    std::vector<std::vector<int32_t>> *keywords_id,
                    std::vector<std::string> *keywords,
                    std::vector<float> *boost_scores,
                    std::vector<float> *threshold);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_UTILS_H_
