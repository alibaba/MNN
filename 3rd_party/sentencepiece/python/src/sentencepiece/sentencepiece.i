%module sentencepiece
%include exception.i

%{

#include <atomic>
#include <iostream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <thread>
#include <vector>
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>

namespace {
PyObject* kUnicodeInput = reinterpret_cast<PyObject* >(0x1);
PyObject* kByteInput = reinterpret_cast<PyObject* >(0x2);

using BytesArray = std::vector<sentencepiece::util::bytes>;

inline void ReleaseResultObject(PyObject *obj) {
  if (obj != nullptr && obj != kUnicodeInput && obj != kByteInput) {
    Py_XDECREF(obj);
  }
}

class PyInputString {
 public:
  explicit PyInputString(PyObject* obj) {
    if (PyUnicode_Check(obj)) {
      str_ = const_cast<char *>(PyUnicode_AsUTF8AndSize(obj, &size_));
      input_type_ = kUnicodeInput;
    } else if (PyBytes_Check(obj)) {
      PyBytes_AsStringAndSize(obj, &str_, &size_);
      input_type_ = kByteInput;
    } else {
      str_ = nullptr;
    }
  }
  absl::string_view str() const { return absl::string_view(data(), size()); }
  const char* data() const { return str_; }
  Py_ssize_t size() const { return size_; }
  bool IsAvalable() const { return str_ != nullptr; }
  PyObject *input_type() const { return input_type_; }

  static bool IsUnicode(PyObject *resultobj) {
    return (resultobj == nullptr || resultobj == kUnicodeInput);
  }

 private:
  PyObject* input_type_ = nullptr;
  char* str_ = nullptr;
  Py_ssize_t size_ = 0;
};

PyObject* MakePyOutputString(const std::string& output,
                             PyObject *resultobj) {
  if (PyInputString::IsUnicode(resultobj)) {
    return PyUnicode_FromStringAndSize(output.data(), output.size());
  }
  return PyBytes_FromStringAndSize(output.data(), output.size());
}

PyObject* MakePyOutputBytes(const sentencepiece::util::bytes& output) {
  return PyBytes_FromStringAndSize(output.data(), output.size());
}

int ToSwigError(sentencepiece::util::StatusCode code) {
  switch (code) {
    case sentencepiece::util::StatusCode::kNotFound:
      return SWIG_IOError;
    case sentencepiece::util::StatusCode::kOutOfRange:
      return SWIG_IndexError;
    case sentencepiece::util::StatusCode::kInvalidArgument:
      return SWIG_SyntaxError;
    default:
      return SWIG_RuntimeError;
  }
  return SWIG_RuntimeError;
}

class PySentenceIterator : public sentencepiece::SentenceIterator {
  public:
  PySentenceIterator(PyObject *iter) : iter_(iter) {
    item_ = PyIter_Next(iter_);
    CopyValue();
  }

  ~PySentenceIterator() {
   // Py_XDECREF(iter_);
  }

  bool done() const override {
    return item_ == nullptr;
  }

  void Next() override {
    item_ = PyIter_Next(iter_);
    CopyValue();
  }

  const std::string &value() const override {
    return value_;
  }

  sentencepiece::util::Status status() const override {
    return status_;
  }

  private:
   void CopyValue() {
     if (item_ == nullptr) return;
     const PyInputString ustring(item_);
     if (ustring.IsAvalable()) {
       const char *data = ustring.data();
       size_t size = ustring.size();
       while (size > 0) {
         if (data[size - 1] == '\r' || data[size - 1] == '\n')
           --size;
         else
           break;
       }
       value_.assign(data, size);
     } else {
       status_ = sentencepiece::util::Status(sentencepiece::util::StatusCode::kInternal,
                                             "Not a string.");
     }
     Py_XDECREF(item_);
   }
   PyObject *iter_ = nullptr;
   PyObject *item_ = nullptr;
   std::string value_;
   sentencepiece::util::Status status_;
};

inline void RewriteIds(const sentencepiece::SentencePieceProcessor &sp,
                       std::vector<int> *ids,
                       bool add_bos, bool add_eos, bool reverse, bool emit_unk_piece) {
  if (!add_bos && !add_eos && !reverse) return;
  if (reverse) std::reverse(ids->begin(), ids->end());
  if (add_bos) ids->insert(ids->begin(), sp.bos_id());
  if (add_eos) ids->push_back(sp.eos_id());
}

inline void RewriteIds(const sentencepiece::SentencePieceProcessor &sp,
                       std::vector<std::string> *pieces,
                       bool add_bos, bool add_eos, bool reverse, bool emit_unk_piece) {
  if (!add_bos && !add_eos && !reverse && !emit_unk_piece) return;
  if (reverse) std::reverse(pieces->begin(), pieces->end());
  if (add_bos) pieces->insert(pieces->begin(), sp.IdToPiece(sp.bos_id()));
  if (add_eos) pieces->push_back(sp.IdToPiece(sp.eos_id()));
  if (emit_unk_piece) {
    const auto &unk = sp.IdToPiece(sp.unk_id());
    for (auto &piece : *pieces) {
      const int id = sp.PieceToId(piece);
      if (id == sp.unk_id()) {
        piece = unk;
      }
    }
  }
}

inline void RewriteIds(const sentencepiece::SentencePieceProcessor &sp,
                       sentencepiece::util::bytes *proto,
                       bool add_bos, bool add_eos, bool reverse, bool emit_unk_piece) {
  if (add_bos || add_eos || reverse || emit_unk_piece) {
    throw sentencepiece::util::Status(
        sentencepiece::util::StatusCode::kUnimplemented,
        "add_bos, add_eos, reverse, and emit_unk_piece is not supported in proto API");
  }
}

inline void RewriteIds(const sentencepiece::SentencePieceProcessor &sp,
                       sentencepiece::ImmutableSentencePieceText *proto,
                       bool add_bos, bool add_eos, bool reverse, bool emit_unk_piece) {
  if (add_bos || add_eos || reverse || emit_unk_piece) {
    throw sentencepiece::util::Status(
        sentencepiece::util::StatusCode::kUnimplemented,
        "add_bos, add_eos, reverse, and emit_unk_piece is not supported in proto API");
  }
}

inline void CheckIds(const std::vector<int> &ids, int num_pieces) {
  for (int id : ids) {
    if (id < 0 || id >= num_pieces) {
      throw sentencepiece::util::Status(
          sentencepiece::util::StatusCode::kOutOfRange,
          "piece id is out of range.");
    }
  }
}

inline void CheckIds(const std::vector<absl::string_view> &ids, int num_pieces) {}

inline void CheckIdsBatch(const std::vector<std::vector<int>> &ids, int num_pieces) {
  for (const auto &v : ids) CheckIds(v, num_pieces);
}

template <typename T>
inline void ConvertToUnicodeSpans(T *proto) {}

template <>
inline void ConvertToUnicodeSpans(sentencepiece::ImmutableSentencePieceText *proto) {
  proto->ConvertToUnicodeSpans();
}

template <>
inline void ConvertToUnicodeSpans(sentencepiece::ImmutableNBestSentencePieceText *proto) {
  proto->ConvertToUnicodeSpans();
}

class ThreadPool {
 public:
  explicit ThreadPool(size_t request_size) :
    request_size_(request_size) {}

  virtual ~ThreadPool() {
    for (auto &task : tasks_) {
      task.join();
    }
  }

  void Schedule(std::function<void()> closure) {
    static constexpr size_t kMinThreadSize = 2;
    if (request_size_ < kMinThreadSize) {
      closure();
    } else {
      tasks_.emplace_back(closure);
    }
  }

 private:
  size_t request_size_ = 0;
  std::vector<std::thread> tasks_;
};

template <typename T>
inline void InitNumThreads(const std::vector<T> &ins, int *num_threads) {
  if (*num_threads < 0) {
    *num_threads = std::thread::hardware_concurrency();
  }
  *num_threads = std::max<int>(1,
                               std::min<int>({*num_threads,
                                     static_cast<int>(ins.size()), 256}));
}

#define DEFINE_ENCODE_BATCH_FUNC_IMPL(FuncName, InType, OutType)        \
  std::vector<OutType> outs(ins.size());                                \
  InitNumThreads(ins, &num_threads);                                    \
  {                                                                     \
    ThreadPool pool(ins.size());                                        \
    std::atomic<size_t> index = 0;                                      \
    for (int n = 0;  n < num_threads; ++n) {                            \
      pool.Schedule([&]() {                                             \
          size_t i = 0;                                                 \
          while ((i = std::atomic_fetch_add(&index, 1)) < outs.size()) { \
            auto out = enable_sampling ?                                \
                       self->Sample##FuncName(ins[i],                   \
                                              nbest_size, alpha) :      \
                       self->FuncName(ins[i]);                          \
            RewriteIds(*self, &out, add_bos, add_eos, reverse,          \
                       emit_unk_piece);                                 \
            ConvertToUnicodeSpans(&out);                                \
            outs[i] = std::move(out);                                   \
          }                                                             \
        });                                                             \
    }                                                                   \
  }                                                                     \
  return outs;

#define DEFINE_DECODE_BATCH_FUNC_IMPL(FuncName, InType, OutType)        \
  std::vector<OutType> outs(ins.size());                                \
  InitNumThreads(ins, &num_threads);                                    \
  {                                                                     \
    std::atomic<size_t> index = 0;                                      \
    ThreadPool pool(ins.size());                                        \
    for (int n = 0;  n < num_threads; ++n) {                            \
      pool.Schedule([&]() {                                             \
          size_t i = 0;                                                 \
          while ((i = std::atomic_fetch_add(&index, 1)) < outs.size()) { \
            auto out = self->FuncName(ins[i]);                          \
            ConvertToUnicodeSpans(&out);                                \
            outs[i] = std::move(out);                                   \
          }                                                             \
        });                                                             \
    }                                                                   \
  }                                                                     \
  return outs;

}  // namespace
%}

%init %{
#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
%}

%exception {
  try {
    $action
    ReleaseResultObject(resultobj);
  }
  catch (const sentencepiece::util::Status &status) {
    SWIG_exception(ToSwigError(status.code()), status.ToString().c_str());
  }
}

%apply unsigned int { uint32_t }

%ignore sentencepiece::util::Status;
%ignore sentencepiece::util::StatusCode;
%ignore absl::string_view;
%ignore std::string_view;
%ignore sentencepiece::SentencePieceText;
%ignore sentencepiece::NormalizerSpec;
%ignore sentencepiece::TrainerSpec;
%ignore sentencepiece::SentencePieceProcessor::status;
%ignore sentencepiece::ImmutableSentencePieceText::mutable_proto;
%ignore sentencepiece::ImmutableSentencePieceText::pieces() const;
%ignore sentencepiece::ImmutableSentencePieceText::ConvertToUnicodeSpans;
%ignore sentencepiece::ImmutableNBestSentencePieceText::mutable_proto;
%ignore sentencepiece::ImmutableNBestSentencePieceText::nbests() const;
%ignore sentencepiece::ImmutableNBestSentencePieceText::ConvertToUnicodeSpans;

%ignore sentencepiece::SentencePieceProcessor::Encode;
%ignore sentencepiece::SentencePieceProcessor::SampleEncode;
%ignore sentencepiece::SentencePieceProcessor::NBestEncode;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAndScore;
%ignore sentencepiece::SentencePieceProcessor::Decode;

%ignore sentencepiece::SentencePieceProcessor::EncodeAsPieces;
%ignore sentencepiece::SentencePieceProcessor::EncodeAsIds;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAsIds;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAsPieces;
%ignore sentencepiece::SentencePieceProcessor::NBestEncodeAsIds;
%ignore sentencepiece::SentencePieceProcessor::NBestEncodeAsPieces;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAndScoreAsIds;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAndScoreAsPieces;
%ignore sentencepiece::SentencePieceProcessor::DecodeIds;
%ignore sentencepiece::SentencePieceProcessor::DecodePieces;

%ignore sentencepiece::SentencePieceProcessor::EncodeAsSerializedProto;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAsSerializedProto;
%ignore sentencepiece::SentencePieceProcessor::NBestEncodeAsSerializedProto;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAndScoreAsSerializedProto;
%ignore sentencepiece::SentencePieceProcessor::DecodePiecesAsSerializedProto;
%ignore sentencepiece::SentencePieceProcessor::DecodeIdsAsSerializedProto;

%ignore sentencepiece::SentencePieceProcessor::EncodeAsImmutableProto;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAsImmutableProto;
%ignore sentencepiece::SentencePieceProcessor::NBestEncodeAsImmutableProto;
%ignore sentencepiece::SentencePieceProcessor::SampleEncodeAndScoreAsImmutableProto;
%ignore sentencepiece::SentencePieceProcessor::DecodePiecesAsImmutableProto;
%ignore sentencepiece::SentencePieceProcessor::DecodeIdsAsImmutableProto;

%ignore sentencepiece::SentencePieceProcessor::Normalize;
%ignore sentencepiece::SentencePieceProcessor::NormalizeWithOffsets;

%ignore sentencepiece::SentencePieceProcessor::model_proto;
%ignore sentencepiece::SentencePieceProcessor::mutable_normalizer_spec;
%ignore sentencepiece::SentencePieceProcessor::Load;
%ignore sentencepiece::SentencePieceProcessor::LoadOrDie;
%ignore sentencepiece::SentencePieceProcessor::SetModel;
%ignore sentencepiece::SentencePieceProcessor::SetNormalizer;
%ignore sentencepiece::pretokenizer::PretokenizerForTrainingInterface;
%ignore sentencepiece::SentenceIterator;
%ignore sentencepiece::ConvertToUnicodeSpans;
%ignore sentencepiece::SentencePieceTrainer::Train;
%ignore sentencepiece::SentencePieceTrainer::GetNormalizerSpec;
%ignore sentencepiece::SentencePieceTrainer::PopulateNormalizerSpec;
%ignore sentencepiece::SentencePieceTrainer::MergeSpecsFromArgs;
%ignore sentencepiece::SentencePieceTrainer::SetProtoField;
%ignore sentencepiece::SentencePieceTrainer::PopulateModelTypeFromString;
%ignore sentencepiece::SentencePieceTrainer::PieceProcecssor;
%ignore sentencepiece::SentencePieceTrainer::SetPretokenizerForTraining;
%ignore sentencepiece::SentencePieceTrainer::GetPretokenizerForTraining;
%ignore sentencepiece::SentencePieceTrainer::SetDataDir;
%ignore sentencepiece::ConvertToUnicodeAlignment;

%ignore sentencepiece::SentencePieceNormalizer::Load;
%ignore sentencepiece::SentencePieceNormalizer::Normalize;
%ignore sentencepiece::SentencePieceNormalizer::mutable_normalizer_spec;

%ignore sentencepiece::io::LoadModelProto;
%ignore sentencepiece::io::SaveModelProto;

%extend sentencepiece::SentencePieceProcessor {
  sentencepiece::util::Status LoadFromFile(absl::string_view arg) {
    return $self->Load(arg);
  }

  /////////////////////////////////////////////////////////////////////////////
  // EncodeAs* (Single request)
  std::vector<int> _EncodeAsIds(absl::string_view text,
                                bool enable_sampling,
                                int nbest_size, float alpha,
                                bool add_bos, bool add_eos, bool reverse,
                                bool emit_unk_piece) const {
    auto ids = enable_sampling ?
               $self->SampleEncodeAsIds(text, nbest_size, alpha) :
               $self->EncodeAsIds(text);
    RewriteIds(*$self, &ids, add_bos, add_eos, reverse, emit_unk_piece);
    return ids;
  }

  std::vector<std::string> _EncodeAsPieces(absl::string_view text,
                                           bool enable_sampling,
                                           int nbest_size, float alpha,
                                           bool add_bos, bool add_eos, bool reverse,
                                           bool emit_unk_piece) const {
    auto pieces = enable_sampling ?
                  $self->SampleEncodeAsPieces(text, nbest_size, alpha) :
                  $self->EncodeAsPieces(text);
    RewriteIds(*$self, &pieces, add_bos, add_eos, reverse, emit_unk_piece);
    return pieces;
  }

  sentencepiece::util::bytes _EncodeAsSerializedProto(absl::string_view text,
                                                      bool enable_sampling,
                                                      int nbest_size, float alpha,
                                                      bool add_bos, bool add_eos, bool reverse,
                                                      bool emit_unk_piece) const {
    auto proto = enable_sampling ?
                 $self->SampleEncodeAsSerializedProto(text, nbest_size, alpha) :
                 $self->EncodeAsSerializedProto(text);
    RewriteIds(*$self, &proto, add_bos, add_eos, reverse, emit_unk_piece);
    return proto;
  }

  sentencepiece::ImmutableSentencePieceText
      _EncodeAsImmutableProto(absl::string_view text,
                              bool enable_sampling,
                              int nbest_size, float alpha,
                              bool add_bos, bool add_eos, bool reverse,
                              bool emit_unk_piece) const {
    auto proto = enable_sampling ?
                 $self->SampleEncodeAsImmutableProto(text, nbest_size, alpha) :
                 $self->EncodeAsImmutableProto(text);
    proto.ConvertToUnicodeSpans();
    RewriteIds(*$self, &proto, add_bos, add_eos, reverse, emit_unk_piece);
    return proto;
  }

  /////////////////////////////////////////////////////////////////////////////
  // EncodeAs* (Batch request)
  std::vector<std::vector<int>> _EncodeAsIdsBatch(
      const std::vector<absl::string_view> &ins, int num_threads,
      bool enable_sampling, int nbest_size, float alpha,
      bool add_bos, bool add_eos, bool reverse,
      bool emit_unk_piece) const {
    DEFINE_ENCODE_BATCH_FUNC_IMPL(EncodeAsIds,
                                  absl::string_view, std::vector<int>);
  }

  std::vector<std::vector<std::string>> _EncodeAsPiecesBatch(
      const std::vector<absl::string_view> &ins, int num_threads,
      bool enable_sampling, int nbest_size, float alpha,
      bool add_bos, bool add_eos, bool reverse,
      bool emit_unk_piece) const {
    DEFINE_ENCODE_BATCH_FUNC_IMPL(EncodeAsPieces,
                                  absl::string_view, std::vector<std::string>);
  }

  BytesArray _EncodeAsSerializedProtoBatch(
      const std::vector<absl::string_view> &ins, int num_threads,
      bool enable_sampling, int nbest_size, float alpha,
      bool add_bos, bool add_eos, bool reverse,
      bool emit_unk_piece) const {
    DEFINE_ENCODE_BATCH_FUNC_IMPL(EncodeAsSerializedProto,
                                  absl::string_view,
                                  sentencepiece::util::bytes);
  }

  std::vector<sentencepiece::ImmutableSentencePieceText>
      _EncodeAsImmutableProtoBatch(
      const std::vector<absl::string_view> &ins, int num_threads,
      bool enable_sampling, int nbest_size, float alpha,
      bool add_bos, bool add_eos, bool reverse,
      bool emit_unk_piece) const {
    DEFINE_ENCODE_BATCH_FUNC_IMPL(EncodeAsImmutableProto,
                                  absl::string_view,
                                  sentencepiece::ImmutableSentencePieceText);
  }

  /////////////////////////////////////////////////////////////////////////////
  // DecodeAs* (Single request)
  std::string _DecodeIds(const std::vector<int> &ids) const {
    CheckIds(ids, $self->GetPieceSize());
    return $self->DecodeIds(ids);
  }

  sentencepiece::util::bytes _DecodeIdsAsBytes(const std::vector<int> &ids) const {
    CheckIds(ids, $self->GetPieceSize());
    return $self->DecodeIds(ids);
  }

  std::string _DecodePieces(const std::vector<absl::string_view> &pieces) const {
    return $self->DecodePieces(pieces);
  }

  sentencepiece::util::bytes _DecodeIdsAsSerializedProto(
      const std::vector<int> &ids) const {
    CheckIds(ids, $self->GetPieceSize());
    return $self->DecodeIdsAsSerializedProto(ids);
  }

  sentencepiece::util::bytes _DecodePiecesAsSerializedProto(
      const std::vector<absl::string_view> &pieces) const {
    CheckIds(pieces, $self->GetPieceSize());
    return $self->DecodePiecesAsSerializedProto(pieces);
  }

  sentencepiece::ImmutableSentencePieceText _DecodeIdsAsImmutableProto(
      const std::vector<int> &ids) const {
    CheckIds(ids, $self->GetPieceSize());
    auto proto = $self->DecodeIdsAsImmutableProto(ids);
    proto.ConvertToUnicodeSpans();
    return proto;
  }

  sentencepiece::ImmutableSentencePieceText _DecodePiecesAsImmutableProto(
      const std::vector<absl::string_view> &pieces) const {
    CheckIds(pieces, $self->GetPieceSize());
    auto proto= $self->DecodePiecesAsImmutableProto(pieces);
    proto.ConvertToUnicodeSpans();
    return proto;
  }

  /////////////////////////////////////////////////////////////////////////////
  // DecodeAs* (Batch request)
  std::vector<std::string> _DecodeIdsBatch(
      const std::vector<std::vector<int>> &ins, int num_threads) const {
    CheckIdsBatch(ins, $self->GetPieceSize());
    DEFINE_DECODE_BATCH_FUNC_IMPL(DecodeIds, int, std::string);
  }

  BytesArray _DecodeIdsAsBytesBatch(
      const std::vector<std::vector<int>> &ins, int num_threads) const {
    CheckIdsBatch(ins, $self->GetPieceSize());
    DEFINE_DECODE_BATCH_FUNC_IMPL(DecodeIds, int, std::string);
  }

  BytesArray _DecodeIdsAsSerializedProtoBatch(
      const std::vector<std::vector<int>> &ins, int num_threads) const {
    CheckIdsBatch(ins, $self->GetPieceSize());
    DEFINE_DECODE_BATCH_FUNC_IMPL(DecodeIdsAsSerializedProto, int,
                                  sentencepiece::util::bytes);
  }

  std::vector<sentencepiece::ImmutableSentencePieceText>
      _DecodeIdsAsImmutableProtoBatch(
          const std::vector<std::vector<int>> &ins, int num_threads) const {
    CheckIdsBatch(ins, $self->GetPieceSize());
    DEFINE_DECODE_BATCH_FUNC_IMPL(DecodeIdsAsImmutableProto, int,
                                  sentencepiece::ImmutableSentencePieceText);
  }

  std::vector<std::string> _DecodePiecesBatch(
      const std::vector<std::vector<absl::string_view>> &ins, int num_threads) const {
    DEFINE_DECODE_BATCH_FUNC_IMPL(DecodePieces, std::string, std::string);
  }

  BytesArray _DecodePiecesAsSerializedProtoBatch(
      const std::vector<std::vector<absl::string_view>> &ins, int num_threads) const {
    DEFINE_DECODE_BATCH_FUNC_IMPL(DecodePiecesAsSerializedProto, std::string,
                                  sentencepiece::util::bytes);
  }

  std::vector<sentencepiece::ImmutableSentencePieceText>
      _DecodePiecesAsImmutableProtoBatch(
          const std::vector<std::vector<absl::string_view>> &ins, int num_threads) const {
    DEFINE_DECODE_BATCH_FUNC_IMPL(DecodePiecesAsImmutableProto, std::string,
                                  sentencepiece::ImmutableSentencePieceText);
  }

  ////////////////////////////////////////////////////////////////////////////
  // NBestEncodeAs* (Single request)
  std::vector<std::vector<int>>
      _NBestEncodeAsIds(absl::string_view text,
                        int nbest_size,
                        bool add_bos, bool add_eos, bool reverse,
                        bool emit_unk_piece) const {
    auto idss = $self->NBestEncodeAsIds(text, nbest_size);
    for (auto &ids : idss) {
      RewriteIds(*$self, &ids, add_bos, add_eos, reverse, emit_unk_piece);
    }
    return idss;
  }

  std::vector<std::vector<std::string>>
      _NBestEncodeAsPieces(absl::string_view text,
                           int nbest_size,
                           bool add_bos, bool add_eos, bool reverse,
                           bool emit_unk_piece) const {
    auto piecess = $self->NBestEncodeAsPieces(text, nbest_size);
    for (auto &pieces : piecess) {
      RewriteIds(*$self, &pieces, add_bos, add_eos, reverse, emit_unk_piece);
    }
    return piecess;
  }

  sentencepiece::util::bytes
      _NBestEncodeAsSerializedProto(absl::string_view text,
                                    int nbest_size,
                                    bool add_bos, bool add_eos, bool reverse,
                                    bool emit_unk_piece) const {
    RewriteIds(*$self, static_cast<sentencepiece::util::bytes *>(nullptr),
               add_bos, add_eos, reverse, emit_unk_piece);
    return $self->NBestEncodeAsSerializedProto(text, nbest_size);
  }

  sentencepiece::ImmutableNBestSentencePieceText
      _NBestEncodeAsImmutableProto(absl::string_view text,
                                   int nbest_size,
                                   bool add_bos, bool add_eos, bool reverse,
                                   bool emit_unk_piece) const {
    RewriteIds(*$self, static_cast<sentencepiece::ImmutableSentencePieceText *>(nullptr),
               add_bos, add_eos, reverse, emit_unk_piece);
    auto proto = $self->NBestEncodeAsImmutableProto(text, nbest_size);
    proto.ConvertToUnicodeSpans();
    return proto;
  }


  /////////////////////////////////////////////////////////////////////////////
  // SampleEncodeAndScoreAs* (Single request)
  std::vector<std::pair<std::vector<int>, float>>
      _SampleEncodeAndScoreAsIds(absl::string_view text,
                                 int num_samples, float alpha, bool wor,
                                 bool include_best,
                                 bool add_bos, bool add_eos, bool reverse,
                                 bool emit_unk_piece) const {
    auto idss = $self->SampleEncodeAndScoreAsIds(text, num_samples,
                                                 alpha, wor, include_best);
    for (auto &ids : idss) {
      RewriteIds(*$self, &ids.first, add_bos, add_eos, reverse, emit_unk_piece);
    }
    return idss;
  }

  std::vector<std::pair<std::vector<std::string>, float>>
      _SampleEncodeAndScoreAsPieces(absl::string_view text,
                                    int num_samples, float alpha, bool wor,
                                    bool include_best,
                                    bool add_bos, bool add_eos, bool reverse,
                                    bool emit_unk_piece) const {
    auto piecess = $self->SampleEncodeAndScoreAsPieces(text, num_samples,
                                                       alpha, wor, include_best);
    for (auto &pieces : piecess) {
      RewriteIds(*$self, &pieces.first, add_bos, add_eos, reverse, emit_unk_piece);
    }
    return piecess;
  }

  sentencepiece::util::bytes
      _SampleEncodeAndScoreAsSerializedProto(absl::string_view text,
                                             int num_samples, float alpha, bool wor,
                                             bool include_best,
                                             bool add_bos, bool add_eos, bool reverse,
                                             bool emit_unk_piece) const {
    RewriteIds(*$self, static_cast<sentencepiece::util::bytes *>(nullptr),
               add_bos, add_eos, reverse, emit_unk_piece);
    return $self->SampleEncodeAndScoreAsSerializedProto(text, num_samples,
                                                        alpha, wor, include_best);
  }

  sentencepiece::ImmutableNBestSentencePieceText
      _SampleEncodeAndScoreAsImmutableProto(absl::string_view text,
                                            int num_samples, float alpha, bool wor,
                                            bool include_best,
                                            bool add_bos, bool add_eos, bool reverse,
                                            bool emit_unk_piece) const {
    RewriteIds(*$self, static_cast<sentencepiece::util::bytes *>(nullptr),
               add_bos, add_eos, reverse, emit_unk_piece);
    auto proto = $self->SampleEncodeAndScoreAsImmutableProto(text, num_samples,
                                                       alpha, wor, include_best);
    proto.ConvertToUnicodeSpans();
    return proto;
  }

  // Normalize
  std::string _Normalize(absl::string_view text) {
    return $self->Normalize(text);
  }

  std::pair<std::string, std::vector<size_t>> _NormalizeWithOffsets(absl::string_view text) {
    std::pair<std::string, std::vector<size_t>> result;
    $self->Normalize(text, &result.first, &result.second).IgnoreError();
    return result;
  }

  // Calculate Entropy
  float _CalculateEntropy(absl::string_view text, float alpha)  {
    return $self->CalculateEntropy(text, alpha);
  }

  std::vector<float> _CalculateEntropyBatch(const std::vector<absl::string_view> &ins,
                                            float alpha, int num_threads)  {
    std::vector<float> outs(ins.size());
    InitNumThreads(ins, &num_threads);
    {
      ThreadPool pool(ins.size());
      std::atomic<size_t> index = 0;
      for (int n = 0;  n < num_threads; ++n) {
        pool.Schedule([&]() {
           size_t i = 0;
           while ((i = std::atomic_fetch_add(&index, 1)) < outs.size()) {
             outs[i] = self->CalculateEntropy(ins[i], alpha);
           }
         });
      }
    }
    return outs;
  }

  // override normalizer_spec
  sentencepiece::util::Status _OverrideNormalizerSpec(
      const std::unordered_map<std::string, std::string> &args) {
    sentencepiece::util::Status status;
    for (const auto &[key, value] : args) {
      status = sentencepiece::SentencePieceTrainer::SetProtoField(
          key, value,
          $self->mutable_normalizer_spec());
      if (!status.ok()) return status;
    }
    return status;
  }

%pythoncode {
  def Init(self,
           model_file=None,
           model_proto=None,
           out_type=int,
           add_bos=False,
           add_eos=False,
           reverse=False,
           emit_unk_piece=False,
           enable_sampling=False,
           nbest_size=-1,
           alpha=0.1,
           num_threads=-1):
    """Initialzie sentencepieceProcessor.

    Args:
      model_file: The sentencepiece model file path.
      model_proto: The sentencepiece model serialized proto.
      out_type: output type. int or str.
      add_bos: Add <s> to the result (Default = false)
      add_eos: Add </s> to the result (Default = false) <s>/</s> is added after
        reversing (if enabled).
      reverse: Reverses the tokenized sequence (Default = false)
      emit_unk_piece: Emits the unk literal string (Default = false)
      nbest_size: sampling parameters for unigram. Invalid in BPE-Dropout.
                  nbest_size = {0,1}: No sampling is performed.
                  nbest_size > 1: samples from the nbest_size results.
                  nbest_size < 0: assuming that nbest_size is infinite and samples
                    from the all hypothesis (lattice) using
                    forward-filtering-and-backward-sampling algorithm.
      alpha: Soothing parameter for unigram sampling, and dropout probability of
             merge operations for BPE-dropout.
      num_threads: number of threads in batch processing (Default = -1, auto-detected)
    """

    _sentencepiece_processor_init_native(self)
    self._out_type = out_type
    self._add_bos = add_bos
    self._add_eos = add_eos
    self._reverse = reverse
    self._emit_unk_piece = emit_unk_piece
    self._enable_sampling = enable_sampling
    self._nbest_size = nbest_size
    self._alpha = alpha
    self._num_threads = num_threads
    if model_file or model_proto:
      self.Load(model_file=model_file, model_proto=model_proto)


  def Encode(self,
             input,
             out_type=None,
             add_bos=None,
             add_eos=None,
             reverse=None,
             emit_unk_piece=None,
             enable_sampling=None,
             nbest_size=None,
             alpha=None,
             num_threads=None):
    """Encode text input to segmented ids or tokens.

      Args:
      input: input string. accepsts list of string.
      out_type: output type. int or str.
      add_bos: Add <s> to the result (Default = false)
      add_eos: Add </s> to the result (Default = false) <s>/</s> is added after
               reversing (if enabled).
      reverse: Reverses the tokenized sequence (Default = false)
      emit_unk_piece: Emits the unk literal string (Default = false)
      nbest_size: sampling parameters for unigram. Invalid in BPE-Dropout.
                  nbest_size = {0,1}: No sampling is performed.
                  nbest_size > 1: samples from the nbest_size results.
                  nbest_size < 0: assuming that nbest_size is infinite and samples
                  from the all hypothesis (lattice) using
                  forward-filtering-and-backward-sampling algorithm.
      alpha: Soothing parameter for unigram sampling, and merge probability for
             BPE-dropout (probablity 'p' in BPE-dropout paper).
      num_threads: the number of threads used in the batch processing (Default = -1).
    """

    if out_type is None:
      out_type = self._out_type
    if add_bos is None:
      add_bos = self._add_bos
    if add_eos is None:
      add_eos = self._add_eos
    if reverse is None:
      reverse = self._reverse
    if emit_unk_piece is None:
      emit_unk_piece = self._emit_unk_piece
    if enable_sampling is None:
      enable_sampling = self._enable_sampling
    if nbest_size is None:
      nbest_size = self._nbest_size
    if alpha is None:
      alpha = self._alpha
    if num_threads is None:
      num_threads = self._num_threads

    if enable_sampling == True and (nbest_size is None or nbest_size == 0 or
                                    nbest_size == 1 or alpha is None):
      raise RuntimeError(
          'When enable_sampling is True, We must specify "nbest_size > 1" or "nbest_size = -1", '
          'and "alpha". "nbest_size" is enabled only on unigram mode ignored in BPE-dropout. '
          'when "nbest_size = -1" , this method samples from all candidates on the lattice '
          'instead of nbest segmentations.'
      )

    if num_threads is None or type(num_threads) is not int:
      raise RuntimeError('num_threads must be int')

    if type(input) is list:
      if out_type is int:
        return self._EncodeAsIdsBatch(input, num_threads, enable_sampling, nbest_size,
                                      alpha, add_bos, add_eos, reverse, emit_unk_piece)
      if out_type is str:
        return self._EncodeAsPiecesBatch(input, num_threads, enable_sampling, nbest_size,
                                         alpha, add_bos, add_eos, reverse, emit_unk_piece)
      if out_type == 'serialized_proto' or out_type == 'proto':
        return self._EncodeAsSerializedProtoBatch(input, num_threads, enable_sampling, nbest_size,
                                                  alpha, add_bos, add_eos, reverse, emit_unk_piece)
      if out_type == 'immutable_proto':
        return self._EncodeAsImmutableProtoBatch(input, num_threads, enable_sampling, nbest_size,
                                                 alpha, add_bos, add_eos, reverse, emit_unk_piece)

    if out_type is int:
      return self._EncodeAsIds(input, enable_sampling, nbest_size,
                               alpha, add_bos, add_eos, reverse, emit_unk_piece)
    if out_type is str:
      return self._EncodeAsPieces(input, enable_sampling, nbest_size,
                                  alpha, add_bos, add_eos, reverse, emit_unk_piece)
    if out_type == 'serialized_proto' or out_type == 'proto':
      return self._EncodeAsSerializedProto(input, enable_sampling, nbest_size,
                                           alpha, add_bos, add_eos, reverse, emit_unk_piece)
    if out_type == 'immutable_proto':
      return self._EncodeAsImmutableProto(input, enable_sampling, nbest_size,
                                          alpha, add_bos, add_eos, reverse, emit_unk_piece)

    raise RuntimeError('unknown out_type={}'.format(out_type))
    return None


  def EncodeAsPieces(self, input, **kwargs):
    return self.Encode(input=input, out_type=str, **kwargs)


  def EncodeAsIds(self, input, **kwargs):
    return self.Encode(input=input, out_type=int, **kwargs)


  def EncodeAsSerializedProto(self, input, **kwargs):
    return self.Encode(input=input, out_type='serialized_proto', **kwargs)


  def EncodeAsImmutableProto(self, input, **kwargs):
    return self.Encode(input=input, out_type='immutable_proto', **kwargs)


  def SampleEncodeAsPieces(self, input, nbest_size=None, alpha=None, **kwargs):
    return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha,
                       out_type=str, enable_sampling=True, **kwargs)


  def SampleEncodeAsIds(self, input, nbest_size=None, alpha=None,**kwargs):
    return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha,
                       out_type=int, enable_sampling=True, **kwargs)


  def SampleEncodeAsSerializedProto(self, input, nbest_size=None, alpha=None, **kwargs):
    return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha,
                       out_type='serialized_proto', enable_sampling=True, **kwargs)


  def SampleEncodeAsImmutableProto(self, input, nbest_size=None, alpha=None, **kwargs):
    return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha,
                       out_type='immutable_proto', enable_sampling=True, **kwargs)


  def NBestEncode(self,
                  input,
                  out_type=None,
                  add_bos=None,
                  add_eos=None,
                  reverse=None,
                  emit_unk_piece=None,
                  nbest_size=None):
    """NBestEncode text input to segmented ids or tokens.

      Args:
      input: input string. accepsts list of string.
      out_type: output type. int or str.
      add_bos: Add <s> to the result (Default = false)
      add_eos: Add </s> to the result (Default = false) <s>/</s> is added after reversing (if enabled).
      reverse: Reverses the tokenized sequence (Default = false)
      emit_unk_piece: Emits the unk literal string (Default = false)
      nbest_size: nbest size
    """

    if out_type is None:
      out_type = self._out_type
    if add_bos is None:
      add_bos = self._add_bos
    if add_eos is None:
      add_eos = self._add_eos
    if reverse is None:
      reverse = self._reverse
    if emit_unk_piece is None:
      emit_unk_piece = self._emit_unk_piece
    if nbest_size is None:
      nbest_size = self._nbest_size

    if nbest_size <= 0:
      nbest_size=1

    def _encode(text):
      if out_type is int:
        return self._NBestEncodeAsIds(text, nbest_size,
                                      add_bos, add_eos, reverse, emit_unk_piece)
      if out_type is str:
        return self._NBestEncodeAsPieces(text, nbest_size,
                                         add_bos, add_eos, reverse, emit_unk_piece)
      if out_type == 'serialized_proto' or out_type == 'proto':
        return self._NBestEncodeAsSerializedProto(text, nbest_size,
                                                  add_bos, add_eos, reverse, emit_unk_piece)
      if out_type == 'immutable_proto':
        return self._NBestEncodeAsImmutableProto(text, nbest_size,
                                                 add_bos, add_eos, reverse, emit_unk_piece)

      raise RuntimeError('unknown out_type')

    if type(input) is list:
      return [_encode(n) for n in input]

    return _encode(input)


  def NBestEncodeAsPieces(self, input, nbest_size=None, **kwargs):
    return self.NBestEncode(input=input, nbest_size=nbest_size,
                            out_type=str, **kwargs)


  def NBestEncodeAsIds(self, input, nbest_size=None, **kwargs):
    return self.NBestEncode(input=input, nbest_size=nbest_size,
                            out_type=int, **kwargs)


  def NBestEncodeAsSerializedProto(self, input, nbest_size=None, **kwargs):
    return self.NBestEncode(input=input, nbest_size=nbest_size,
                            out_type='serialized_proto', **kwargs)


  def NBestEncodeAsImmutableProto(self, input, nbest_size=None, **kwargs):
    return self.NBestEncode(input=input, nbest_size=nbest_size,
                            out_type='immutable_proto', **kwargs)


  def SampleEncodeAndScore(self,
                           input,
                           out_type=None,
                           add_bos=None,
                           add_eos=None,
                           reverse=None,
                           emit_unk_piece=None,
                           num_samples=None,
                           alpha=None,
                           wor=None,
                           include_best=None):
    """SampleEncodeAndScore text input to segmented ids or tokens.

      Args:
      input: input string. accepsts list of string.
      out_type: output type. int or str or 'serialized_proto' or 'immutable_proto'
      add_bos: Add <s> to the result (Default = false)
      add_eos: Add </s> to the result (Default = false) <s>/</s> is added after reversing (if enabled).
      reverse: Reverses the tokenized sequence (Default = false)
      emit_unk_piece: Emits the unk literal string (Default = false)
      num_samples: How many samples to return (Default = 1)
      alpha: inverse temperature for sampling
      wor: whether to sample without replacement (Default = false)
      include_best: whether to include the best tokenization, requires wor=True (Default = false)
    """

    if out_type is None:
      out_type = self._out_type
    if add_bos is None:
      add_bos = self._add_bos
    if add_eos is None:
      add_eos = self._add_eos
    if reverse is None:
      reverse = self._reverse
    if emit_unk_piece is None:
      emit_unk_piece = self._emit_unk_piece
    if num_samples is None:
      num_samples = 1
    if alpha is None:
      alpha = 1.
    if wor is None:
      wor = False
    if include_best is None:
      include_best = False

    if num_samples <= 0:
      raise RuntimeError('num_examples must be positive')

    if include_best and not wor:
      raise RuntimeError('When include_best is True, We must specify "wor = True".')


    def _encode(text):
      if out_type is int:
        return self._SampleEncodeAndScoreAsIds(text, num_samples, alpha, wor, include_best,
                                               add_bos, add_eos, reverse, emit_unk_piece)
      if out_type is str:
        return self._SampleEncodeAndScoreAsPieces(text, num_samples, alpha, wor, include_best,
                                                  add_bos, add_eos, reverse, emit_unk_piece)

      if out_type == 'serialized_proto' or out_type == 'proto':
        return self._SampleEncodeAndScoreAsSerializedProto(text, num_samples, alpha, wor, include_best,
                                                           add_bos, add_eos, reverse, emit_unk_piece)

      if out_type == 'immutable_proto':
        return self._SampleEncodeAndScoreAsImmutableProto(text, num_samples, alpha, wor, include_best,
                                                          add_bos, add_eos, reverse, emit_unk_piece)

      raise RuntimeError('unknown output type')


    if type(input) is list:
      return [_encode(n) for n in input]

    return _encode(input)


  def SampleEncodeAndScoreAsPieces(self, input, num_samples=None, alpha=None, **kwargs):
    return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha,
                                     out_type=str, **kwargs)


  def SampleEncodeAndScoreAsIds(self, input, num_samples=None, alpha=None, **kwargs):
    return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha,
                                     out_type=int, **kwargs)


  def SampleEncodeAndScoreAsSerializedProto(self, input, num_samples=None, alpha=None, **kwargs):
    return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha,
                                     out_type='serialized_proto', **kwargs)


  def SampleEncodeAndScoreAsImmutableProto(self, input, num_samples=None, alpha=None, **kwargs):
    return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha,
                                     out_type='immutable_proto', **kwargs)


  def Decode(self, input, out_type=str, num_threads=None):
    """Decode processed id or token sequences.

    Args:
      out_type: output type. str, bytes or 'serialized_proto' or 'immutable_proto' (Default = str)
      num_threads: the number of threads used in the batch processing (Default = -1).
    """

    if num_threads is None:
      num_threads = self._num_threads

    if num_threads is None or type(num_threads) is not int:
      raise RuntimeError('num_threads must be int')

    if not input:
      return ''

    if out_type is str:
      if type(input) is int:
        return self._DecodeIds([input])
      if type(input) is str:
        return self._DecodePieces([input])

      if type(input) is list:
        if len(input) == 0 or type(input[0]) is int:
          return self._DecodeIds(input)
        if type(input[0]) is str:
          return self._DecodePieces(input)

        if type(input[0]) is list:
          if len(input[0]) == 0 or type(input[0][0]) is int:
           return self._DecodeIdsBatch(input, num_threads)
          if type(input[0][0]) is str:
           return self._DecodePiecesBatch(input, num_threads)

    if out_type is bytes:
      if type(input) is int:
        return self._DecodeIdsAsBytes([input])
      if type(input) is str:
        return self._DecodePieces([input])

      if type(input) is list:
        if len(input) == 0 or type(input[0]) is int:
          return self._DecodeIdsAsBytes(input)
        if type(input[0]) is str:
          return self._DecodePieces(input)

        if type(input[0]) is list:
          if len(input[0]) == 0 or type(input[0][0]) is int:
           return self._DecodeIdsAsBytesBatch(input, num_threads)
          if type(input[0][0]) is str:
           return self._DecodePiecesBatch(input, num_threads)

    if out_type == 'serialized_proto':
      if type(input) is int:
        return self._DecodeIdsAsSerializedProto([input])
      if type(input) is str:
        return self._DecodePiecesAsSerializedProto([input])

      if type(input) is list:
        if len(input) == 0 or type(input[0]) is int:
          return self._DecodeIdsAsSerializedProto(input)
        if type(input[0]) is str:
          return self._DecodePiecesAsSerializedProto(input)

        if type(input[0]) is list:
          if len(input[0]) == 0 or type(input[0][0]) is int:
           return self._DecodeIdsAsSerializedProtoBatch(input, num_threads)
          if type(input[0][0]) is str:
           return self._DecodePiecesAsSerializedProtoBatch(input, num_threads)


    if out_type == 'immutable_proto':
      if type(input) is int:
        return self._DecodeIdsAsImmutableProto([input])
      if type(input) is str:
        return self._DecodePiecesAsImmutableProto([input])

      if type(input) is list:
        if len(input) == 0 or type(input[0]) is int:
          return self._DecodeIdsAsImmutableProto(input)
        if type(input[0]) is str:
          return self._DecodePiecesAsImmutableProto(input)

        if type(input[0]) is list:
          if len(input[0]) == 0 or type(input[0][0]) is int:
           return self._DecodeIdsAsImmutableProtoBatch(input, num_threads)
          if type(input[0][0]) is str:
           return self._DecodePiecesAsImmutableProtoBatch(input, num_threads)


    raise RuntimeError('unknown output or input type')
    return None


  def DecodePieces(self, input, out_type=str, **kwargs):
    return self.Decode(input=input, out_type=out_type, **kwargs)


  def DecodeIds(self, input, out_type=str, **kwargs):
    return self.Decode(input=input, out_type=out_type, **kwargs)


  def DecodePiecesAsSerializedProto(self, input, out_type='serialized_proto', **kwargs):
    return self.Decode(input=input, out_type=out_type, **kwargs)


  def DecodeIdsAsSerializedProto(self, input, out_type='serialized_proto', **kwargs):
    return self.Decode(input=input, out_type=out_type, **kwargs)


  def DecodePiecesAsImmutableProto(self, input, out_type='immutable_proto', **kwargs):
    return self.Decode(input=input, out_type=out_type, **kwargs)


  def DecodeIdsAsImmutableProto(self, input, out_type='immutable_proto', **kwargs):
    return self.Decode(input=input, out_type=out_type, **kwargs)


  def CalculateEntropy(self, input, alpha, num_threads=None):
    """Calculate sentence entropy"""
    if type(input) is list:
      if num_threads is None:
        num_threads = self._num_threads
      if num_threads is None or type(num_threads) is not int:
        raise RuntimeError('num_threads must be int')
      return self._CalculateEntropyBatch(input, alpha, num_threads)

    return self._CalculateEntropy(input, alpha)


  def Normalize(self, input, with_offsets=None):
    def _normalize(text):
      if with_offsets:
        return self._NormalizeWithOffsets(text)
      return self._Normalize(text)

    if type(input) is list:
      return [_normalize(x) for x in input]
    return _normalize(input)

  def OverrideNormalizerSpec(self, **kwargs):
    new_kwargs = {}
    for key, value in kwargs.items():
      new_kwargs[key] = str(value)
    return self._OverrideNormalizerSpec(new_kwargs)


  def piece_size(self):
    return self.GetPieceSize()


  def vocab_size(self):
    return self.GetPieceSize()


  def __getstate__(self):
    return self.serialized_model_proto()


  def __setstate__(self, serialized_model_proto):
    self.__init__()
    self.LoadFromSerializedProto(serialized_model_proto)


  def __len__(self):
    return self.GetPieceSize()


  def __getitem__(self, piece):
    return self.PieceToId(piece)


  def Load(self, model_file=None, model_proto=None):
    """Overwride SentencePieceProcessor.Load to support both model_file and model_proto.

    Args:
      model_file: The sentencepiece model file path.
      model_proto: The sentencepiece model serialized proto. Either `model_file`
        or `model_proto` must be set.
    """
    if model_file and model_proto:
      raise RuntimeError('model_file and model_proto must be exclusive.')
    if model_proto:
      return self.LoadFromSerializedProto(model_proto)
    return self.LoadFromFile(model_file)
}
}

%extend sentencepiece::SentencePieceTrainer {
  static void _TrainFromString(absl::string_view arg) {
    const auto _status = sentencepiece::SentencePieceTrainer::Train(arg);
    if (!_status.ok()) throw _status;
    return;
  }

  static void _TrainFromMap(const std::unordered_map<std::string, std::string> &args) {
    const auto _status = sentencepiece::SentencePieceTrainer::Train(args);
    if (!_status.ok()) throw _status;
    return;
  }

  static void _TrainFromMap2(const std::unordered_map<std::string, std::string> &args,
                            SentenceIterator *iter) {
    const auto _status = sentencepiece::SentencePieceTrainer::Train(args, iter);
    if (!_status.ok()) throw _status;
    return;
  }

  static sentencepiece::util::bytes _TrainFromMap3(const std::unordered_map<std::string, std::string> &args) {
    sentencepiece::util::bytes model_proto;
    const auto _status = sentencepiece::SentencePieceTrainer::Train(args, nullptr, &model_proto);
    if (!_status.ok()) throw _status;
    return model_proto;
  }

  static sentencepiece::util::bytes _TrainFromMap4(const std::unordered_map<std::string, std::string> &args,
                                                  SentenceIterator *iter) {
    sentencepiece::util::bytes model_proto;
    const auto _status = sentencepiece::SentencePieceTrainer::Train(args, iter, &model_proto);
    if (!_status.ok()) throw _status;
    return model_proto;
  }

%pythoncode {
  @staticmethod
  def _Train(arg=None, **kwargs):
    """Train Sentencepiece model. Accept both kwargs and legacy string arg."""
    if arg is not None and type(arg) is str:
      return SentencePieceTrainer._TrainFromString(arg)

    def _encode(value):
      """Encode value to CSV.."""
      if type(value) is list:
        if sys.version_info[0] == 3:
          f = StringIO()
        else:
          f = BytesIO()
        writer = csv.writer(f, lineterminator='')
        writer.writerow([str(v) for v in value])
        return f.getvalue()
      else:
        return str(value)

    sentence_iterator = None
    model_writer = None
    new_kwargs = {}
    for key, value in kwargs.items():
      if key in ['sentence_iterator', 'sentence_reader']:
        sentence_iterator = value
      elif key in ['model_writer']:
        model_writer = value
      else:
        new_kwargs[key] = _encode(value)

    if model_writer:
      if sentence_iterator:
        model_proto = SentencePieceTrainer._TrainFromMap4(new_kwargs,
                                                         sentence_iterator)
      else:
        model_proto = SentencePieceTrainer._TrainFromMap3(new_kwargs)
      model_writer.write(model_proto)
    else:
      if sentence_iterator:
        return SentencePieceTrainer._TrainFromMap2(new_kwargs, sentence_iterator)
      else:
        return SentencePieceTrainer._TrainFromMap(new_kwargs)

    return None

  @staticmethod
  def Train(arg=None, logstream=None, **kwargs):
    with _LogStream(ostream=logstream):
      SentencePieceTrainer._Train(arg=arg, **kwargs)
}
}

%extend sentencepiece::SentencePieceNormalizer {
  sentencepiece::util::Status LoadFromFile(absl::string_view arg) {
    return $self->Load(arg);
  }

  std::string _Normalize(absl::string_view text) {
    std::string result;
    const auto _status = $self->Normalize(text, &result);
    if (!_status.ok()) throw _status;
    return result;
  }

  std::pair<std::string, std::vector<size_t>> _NormalizeWithOffsets(absl::string_view text) {
    std::pair<std::string, std::vector<size_t>> result;
    const auto _status = $self->Normalize(text, &result.first, &result.second);
    if (!_status.ok()) throw _status;
    return result;
  }

  void _SetProtoField(absl::string_view name, bool value) {
    sentencepiece::SentencePieceTrainer::SetProtoField(
        name,
        value ? "1" : "0",
        $self->mutable_normalizer_spec()).IgnoreError();
  }

%pythoncode %{
  def Init(self,
           model_file=None,
           model_proto=None,
           rule_tsv=None,
           rule_name=None,
           add_dummy_prefix=False,
           escape_whitespaces=False,
           remove_extra_whitespaces=False):
    """Initialzie sentencePieceNormalizer.

    Args:
      model_file: The sentencepiece model file path.
      model_proto: The sentencepiece model serialized proto.
      rule_tsv: The normalization rule file in TSV format.
      rule_name: Pre-defined normalization name.
      add_dummy_prefix: add dummy prefix.
      escape_whitespaces: escape whitespaces.
      remove_extra_whitespaces: remove extra whitespaces.
    """

    _sentencepiece_normalizer_init_native(self)

    if model_file:
      status = self.LoadFromFile(model_file)
    elif model_proto:
      status = self.LoadFromSerializedProto(model_proto)
    elif rule_tsv:
      status = self.LoadFromRuleTSV(rule_tsv)
    elif rule_name:
      status = self.LoadFromRuleName(rule_name)
    else:
      raise RuntimeError('no model is specified')

    if status:
      self._SetProtoField('add_dummy_prefix', add_dummy_prefix)
      self._SetProtoField('escape_whitespaces', escape_whitespaces)
      self._SetProtoField('remove_extra_whitespaces', remove_extra_whitespaces)

  def Normalize(self, input, with_offsets=None):
    def _normalize(text):
      if with_offsets:
        return self._NormalizeWithOffsets(text)
      return self._Normalize(text)

    if type(input) is list:
      return [_normalize(x) for x in input]
    return _normalize(input)


  def __getstate__(self):
    return self.serialized_model_proto()


  def __setstate__(self, serialized_model_proto):
    self.__init__()
    self.LoadFromSerializedProto(serialized_model_proto)
%}
}

%extend sentencepiece::ImmutableSentencePieceText_ImmutableSentencePiece {
  const sentencepiece::util::bytes& _surface_as_bytes() const {
    return $self->surface();
  }

  const sentencepiece::util::bytes& _piece_as_bytes() const {
    return $self->piece();
  }

  %rename(_piece) piece;
  %rename(_piece_as_bytes) piece_as_bytes;
  %rename(_id) id;
  %rename(_surface) surface;
  %rename(_surface_as_bytes) surface_as_bytes;
  %rename(_begin) begin;
  %rename(_end) end;

  %pythoncode %{
    piece = property(_piece)
    piece_as_bytes = property(_piece_as_bytes)
    surface = property(_surface)
    surface_as_bytes = property(_surface_as_bytes)
    id = property(_id)
    begin = property(_begin)
    end = property(_end)

    def __str__(self):
      return ('piece: \"{}\"\n'
              'id: {}\n'
              'surface: \"{}\"\n'
              'begin: {}\n'
              'end: {}\n').format(self.piece, self.id, self.surface,
                                  self.begin, self.end)

    def __eq__(self, other):
      return self.piece == other.piece and self.id == other.id and self.surface == other.surface and self.begin == other.begin and self.end == other.end

    def __hash__(self):
      return hash(str(self))

    __repr__ = __str__
  %}
}

%extend sentencepiece::ImmutableSentencePieceText {
  const sentencepiece::util::bytes& _text_as_bytes() const {
    return $self->text();
  }

  %rename(_text) text;
  %rename(_text_as_bytes) text_as_bytes;
  %rename(_score) score;
  %rename(_pieces) pieces;
  %rename(_pieces_size) pieces_size;

  %pythoncode %{
    text = property(_text)
    text_as_bytes = property(_text_as_bytes)
    score = property(_score)

    class ImmutableSentencePieceIterator:
      def __init__(self, proto):
        self.proto = proto
        self.len = self.proto._pieces_size()

      def __len__(self):
        return self.len

      def __getitem__(self, index):
        if isinstance(index, slice):
          return [self.proto._pieces(i) for i in range(self.len)][index.start:index.stop:index.step]
        if index < 0:
          index = index + self.len
        if index < 0 or index >= self.len:
          raise IndexError('piece index is out of range')
        return self.proto._pieces(index)

      def __str__(self):
        return '\n'.join(['pieces {{\n{}}}'.format(str(x)) for x in self])

      __repr__ = __str__

    @property
    def pieces(self):
      return ImmutableSentencePieceText.ImmutableSentencePieceIterator(self)

    def __eq__(self, other):
      return self.SerializeAsString() == other.SerializeAsString()

    def __hash__(self):
      return hash(self.SerializeAsString())

    def __str__(self):
      return ('text: \"{}\"\n'
              'score: {}\n'
              '{}').format(self.text, self.score,
                           '\n'.join(['pieces {{\n{}}}'.format(str(x)) for x in self.pieces]))

    __repr__ = __str__
  %}
}

%extend sentencepiece::ImmutableNBestSentencePieceText {
  %rename(_nbests) nbests;
  %rename(_nbests_size) nbests_size;

  %pythoncode %{
    class ImmutableSentencePieceTextIterator:
      def __init__(self, proto):
        self.proto = proto
        self.len = self.proto._nbests_size()

      def __len__(self):
        return self.len

      def __getitem__(self, index):
        if isinstance(index, slice):
          return [self.proto._nbests(i) for i in range(self.len)][index.start:index.stop:index.step]
        if index < 0:
          index = index + self.len
        if index < 0 or index >= self.len:
          raise IndexError('nbests index is out of range')
        return self.proto._nbests(index)

      def __str__(self):
        return '\n'.join(['nbests {{\n{}}}'.format(str(x)) for x in self])

      __repr__ = __str__

    @property
    def nbests(self):
      return ImmutableNBestSentencePieceText.ImmutableSentencePieceTextIterator(self)

    def __eq__(self, other):
      return self.SerializeAsString() == other.SerializeAsString()

    def __hash__(self):
      return hash(self.SerializeAsString())

    def __str__(self):
      return '\n'.join(['nbests {{\n{}}}'.format(str(x)) for x in self.nbests])

    __repr__ = __str__
  %}
}

%typemap(out) std::vector<int> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyList_SET_ITEM($result, i, PyInt_FromLong(static_cast<long>($1[i])));
  }
}

%typemap(out) std::vector<float> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyList_SET_ITEM($result, i, PyFloat_FromDouble(static_cast<double>($1[i])));
  }
}

%typemap(out) std::vector<std::vector<int>> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = PyList_New($1[i].size());
    for (size_t j = 0; j < $1[i].size(); ++j) {
      PyList_SET_ITEM(obj, j, PyInt_FromLong(static_cast<long>($1[i][j])));
    }
    PyList_SET_ITEM($result, i, obj);
  }
}

%typemap(out) std::vector<std::string> {
  PyObject *input_type = resultobj;
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyList_SET_ITEM($result, i, MakePyOutputString($1[i], input_type));
  }
}

%typemap(out) BytesArray {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyList_SET_ITEM($result, i, MakePyOutputBytes($1[i]));
  }
}

%typemap(out) std::vector<std::vector<std::string>> {
  PyObject *input_type = resultobj;
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = PyList_New($1[i].size());
    for (size_t j = 0; j < $1[i].size(); ++j) {
      PyList_SET_ITEM(obj, j, MakePyOutputString($1[i][j], input_type));
    }
    PyList_SET_ITEM($result, i, obj);
  }
}

%typemap(out) sentencepiece::util::bytes {
  $result = MakePyOutputBytes($1);
}

%typemap(out) const sentencepiece::util::bytes& {
  $result = MakePyOutputBytes(*$1);
}

%typemap(out) std::string {
  PyObject *input_type = resultobj;
  $result = MakePyOutputString($1, input_type);
}

%typemap(out) const std::string& {
  PyObject *input_type = resultobj;
  $result = MakePyOutputString(*$1, input_type);
}

%typemap(out) sentencepiece::util::Status {
  if (!$1.ok()) {
    SWIG_exception(ToSwigError($1.code()), $1.ToString().c_str());
  }
  $result = SWIG_From_bool($1.ok());}


%typemap(in) const std::string & {
  const PyInputString ustring($input);
  if (!ustring.IsAvalable()) {
    PyErr_SetString(PyExc_TypeError, "not a string");
    SWIG_fail;
  }
  resultobj = ustring.input_type();
  $1 = new std::string(ustring.data(), ustring.size());
}

%typemap(typecheck) absl::string_view = char *;

%typemap(in) absl::string_view {
  const PyInputString ustring($input);
  if (!ustring.IsAvalable()) {
    PyErr_SetString(PyExc_TypeError, "not a string");
    SWIG_fail;
  }
  resultobj = ustring.input_type();
  $1 = ustring.str();
}

%typemap(in) const std::vector<absl::string_view>& {
  std::vector<absl::string_view> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<absl::string_view>(size);
    for (size_t i = 0; i < size; ++i) {
      const PyInputString ustring(PyList_GetItem($input, i));
      if (ustring.IsAvalable()) {
        (*out)[i] = ustring.str();
      } else {
        PyErr_SetString(PyExc_TypeError, "list must contain strings");
        SWIG_fail;
      }
      resultobj = ustring.input_type();
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "not a list");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(in) const std::vector<int>& {
  std::vector<int> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<int>(size);
    for (size_t i = 0; i < size; ++i) {
      PyObject *o = PyList_GetItem($input, i);
      if (PyInt_Check(o)) {
        (*out)[i] = static_cast<int>(PyInt_AsLong(o));
      } else {
        PyErr_SetString(PyExc_TypeError,"list must contain integers");
        SWIG_fail;
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(in) const std::vector<std::vector<absl::string_view>>& {
  std::vector<std::vector<absl::string_view>> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<std::vector<absl::string_view>>(size);
    for (size_t i = 0; i < size; ++i) {
      PyObject *o = PyList_GetItem($input, i);
      if (PyList_Check(o)) {
        const size_t size2 = PyList_Size(o);
        (*out)[i].resize(size2);
        for (size_t j = 0; j < size2; ++j) {
          const PyInputString ustring(PyList_GetItem(o, j));
          if (ustring.IsAvalable()) {
            (*out)[i][j] = ustring.str();
          } else {
            PyErr_SetString(PyExc_TypeError,"list must contain integers");
            SWIG_fail;
          }
          resultobj = ustring.input_type();
        }
      } else {
        PyErr_SetString(PyExc_TypeError,"not a list");
        SWIG_fail;
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(in) const std::vector<std::vector<int>>& {
  std::vector<std::vector<int>> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<std::vector<int>>(size);
    for (size_t i = 0; i < size; ++i) {
      PyObject *o = PyList_GetItem($input, i);
      if (PyList_Check(o)) {
        const size_t size2 = PyList_Size(o);
        (*out)[i].resize(size2);
        for (size_t j = 0; j < size2; ++j) {
          PyObject *o2 = PyList_GetItem(o, j);
          if (PyInt_Check(o2)) {
            (*out)[i][j] = static_cast<int>(PyInt_AsLong(o2));
          } else {
            PyErr_SetString(PyExc_TypeError, "list must contain strings");
            SWIG_fail;
          }
        }
      } else {
        PyErr_SetString(PyExc_TypeError, "not a list");
        SWIG_fail;
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(in) const std::unordered_map<std::string, std::string> & {
  std::unordered_map<std::string, std::string> *out = nullptr;
  if (PyDict_Check($input)) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    out = new std::unordered_map<std::string, std::string>;
    while (PyDict_Next($input, &pos, &key, &value)) {
      const PyInputString key_ustring(key);
      const PyInputString value_ustring(value);
      if (key_ustring.IsAvalable() && value_ustring.IsAvalable()) {
        out->emplace(std::string(key_ustring.data(), key_ustring.size()),
                     std::string(value_ustring.data(), value_ustring.size()));
      } else {
        PyErr_SetString(PyExc_TypeError, "map must contain strings.");
        SWIG_fail;
      }
      resultobj = key_ustring.input_type();
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "not a dictionary");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(out) std::vector<std::pair<std::vector<std::string>, float>> {
  PyObject *input_type = resultobj;
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = PyList_New($1[i].first.size());
    for (size_t j = 0; j < $1[i].first.size(); ++j) {
      PyList_SET_ITEM(obj, j, MakePyOutputString($1[i].first[j], input_type));
    }
    PyList_SET_ITEM($result, i, PyTuple_Pack(2, obj, PyFloat_FromDouble(static_cast<double>($1[i].second))));
  }
}

%typemap(out) std::vector<std::pair<std::vector<int>, float>> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = PyList_New($1[i].first.size());
    for (size_t j = 0; j < $1[i].first.size(); ++j) {
      PyList_SET_ITEM(obj, j, PyInt_FromLong(static_cast<long>($1[i].first[j])));
    }
    PyList_SET_ITEM($result, i, PyTuple_Pack(2, obj, PyFloat_FromDouble(static_cast<double>($1[i].second))));
  }
}

%typemap(out) std::vector<sentencepiece::ImmutableSentencePieceText> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i) {
    PyObject *obj = SWIG_NewPointerObj(new sentencepiece::ImmutableSentencePieceText($1.at(i)), SWIGTYPE_p_sentencepiece__ImmutableSentencePieceText, SWIG_POINTER_OWN | 0);
    PyList_SET_ITEM($result, i, obj);
  }
}

// Types for normalized string and offset
%typemap(out) std::pair<std::string, std::vector<size_t>> {
  PyObject *input_type = resultobj;
  if (PyInputString::IsUnicode(input_type)) {
    sentencepiece::ConvertToUnicodeAlignment(arg2, $1.first, &$1.second);
  }
  PyObject *obj = PyList_New($1.second.size());
  for (size_t i = 0; i < $1.second.size(); ++i) {
    PyList_SET_ITEM(obj, i, PyInt_FromLong(static_cast<long>($1.second[i])));
  }
  $result = PyTuple_Pack(2, MakePyOutputString($1.first, input_type), obj);
}

%typemap(in) sentencepiece::SentenceIterator * {
  sentencepiece::SentenceIterator *out = nullptr;
  if (PyIter_Check($input)) {
    out = new PySentenceIterator($input);
  } else {
    PyErr_SetString(PyExc_TypeError, "not a iterator");
    SWIG_fail;
  }
  $1 = out;
}

%typemap(freearg) const std::string& {
  delete $1;
}

%typemap(freearg) const std::vector<std::string>& {
  delete $1;
}

%typemap(freearg) const std::vector<absl::string_view>& {
  delete $1;
}

%typemap(freearg) const std::vector<std::vector<std::string>>& {
  delete $1;
}

%typemap(freearg) const std::vector<int>& {
  delete $1;
}

%typemap(freearg) const std::vector<float>& {
  delete $1;
}

%typemap(freearg) const std::vector<std::vector<int>>& {
  delete $1;
}

%typemap(freearg) const std::unordered_map<std::string, std::string> & {
  delete $1;
}

%typemap(freearg) sentencepiece::SentenceIterator * {
  delete $1;
}

%typemap(freearg) sentencepiece::ImmutableSentencePieceText_ImmutableSentencePiece {
  delete $1;
}

%typemap(freearg) sentencepiece::ImmutableSentencePieceText {
  delete $1;
}

%typemap(freearg) sentencepiece::ImmutableNBestSentencePieceText {
  delete $1;
}

%include <sentencepiece_processor.h>
%include <sentencepiece_trainer.h>

%pythoncode %{

import re
import csv
import sys
import os
import importlib.resources
from io import StringIO
from io import BytesIO


def _add_snake_case(classname):
  """Added snake_cased method from CammelCased method."""

  snake_map = {}
  for k, v in classname.__dict__.items():
    if re.match(r'^[A-Z]+', k):
      snake = re.sub(r'(?<!^)(?=[A-Z])', '_',
                     k).lower().replace('n_best', 'nbest')
      snake_map[snake] = v
  for k, v in snake_map.items():
    setattr(classname, k, v)


def _batchnize(classname, name):
  """Enables batch request for the method classname.name."""
  func = getattr(classname, name, None)
  def _func(v, n):
    if type(n) is int and (n < 0 or n >= v.piece_size()):
      raise IndexError('piece id is out of range.')
    return func(v, n)

  def _batched_func(self, arg):
    if type(arg) is list:
      return [_func(self, n) for n in arg]
    else:
      return _func(self, arg)

  setattr(classname, name, _batched_func)


_sentencepiece_processor_init_native = SentencePieceProcessor.__init__
_sentencepiece_normalizer_init_native = SentencePieceNormalizer.__init__
setattr(SentencePieceProcessor, '__init__', SentencePieceProcessor.Init)
setattr(SentencePieceNormalizer, '__init__', SentencePieceNormalizer.Init)

SentencePieceProcessor.Tokenize = SentencePieceProcessor.Encode
SentencePieceProcessor.Detokenize = SentencePieceProcessor.Decode

for m in [
    'PieceToId', 'IdToPiece', 'GetScore', 'IsUnknown', 'IsControl', 'IsUnused',
    'IsByte'
]:
  _batchnize(SentencePieceProcessor, m)

_add_snake_case(SentencePieceProcessor)
_add_snake_case(SentencePieceTrainer)
_add_snake_case(SentencePieceNormalizer)
set_random_generator_seed = SetRandomGeneratorSeed
set_min_log_level = SetMinLogLevel

from ._version import __version__

SetDataDir(os.path.join(str(importlib.resources.files('sentencepiece')), 'package_data'))

class _LogStream(object):
  def __init__(self, ostream=None):
    self.ostream = ostream
    if self.ostream is not None:
      self.orig_stream_fileno = sys.stderr.fileno()

  def __enter__(self):
    if self.ostream is not None:
      self.orig_stream_dup = os.dup(self.orig_stream_fileno)
      os.dup2(self.ostream.fileno(), self.orig_stream_fileno)

  def __exit__(self, type, value, traceback):
    if self.ostream is not None:
      os.close(self.orig_stream_fileno)
      os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
      os.close(self.orig_stream_dup)
      self.ostream.close()
%}
