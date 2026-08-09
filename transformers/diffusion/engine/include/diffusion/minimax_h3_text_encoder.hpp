//
//  minimax_h3_text_encoder.hpp
//
//  MiniMax-H3 text conditioner: the first 50 decoder layers of Qwen3-VL-32B.
//
#ifndef MNN_MINIMAX_H3_TEXT_ENCODER_HPP
#define MNN_MINIMAX_H3_TEXT_ENCODER_HPP

#include <memory>
#include <string>
#include <vector>

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>

namespace MNN {
namespace DIFFUSION {

class Tokenizer;

// MiniMax-H3 conditions on `hidden_states[50]` of its Qwen3-VL conditioner -- the output of the 50th decoder
// layer, before the model's final norm, which is why the stack cannot simply be truncated and read at its end.
// For a text-only request neither the vision tower nor the language-model head is ever touched.
//
// The conditioner is 25B parameters, so it is partitioned like the transformer, and the token embedding table
// is a host-side gather rather than a graph: only the rows a prompt actually uses are ever read.
//
// The graph has a fixed sequence length. Prompts shorter than it need no padding mask -- attention is causal,
// so a real token never attends to a padding slot -- and the padding rows are dropped from the output.
class MNN_PUBLIC MinimaxH3TextEncoder {
public:
    MinimaxH3TextEncoder(std::string resourcePath, MNNForwardType backendType);
    ~MinimaxH3TextEncoder();

    bool load();

    // Tokenizes `prompt` and returns `(numTokens, hiddenSize)` conditioning, row-major.
    bool encode(const std::string& prompt, std::vector<float>* hiddenStates, int* numTokens);

    // Same, from token ids the caller already has.
    bool encodeTokens(const std::vector<int>& tokenIds, std::vector<float>* hiddenStates);

    int hiddenSize() const;
    int maxTokens() const;

    // Caps how many layer partitions stay resident, for the same reason the transformer needs it: MNN's CUDA
    // backend materializes quantized weights, so the whole stack does not fit one device.
    void setResidentGroups(int groups);

private:
    struct Resources;

    std::shared_ptr<Express::Module> loadModule(const std::string& name);
    // Reads the rows `tokenIds` addresses out of the bfloat16 embedding table.
    bool gatherEmbeddings(const std::vector<int>& tokenIds, float* destination) const;

    std::unique_ptr<Resources> mResources;
    std::vector<std::shared_ptr<Express::Module>> mGroups;
    std::shared_ptr<Express::Executor::RuntimeManager> mRuntime;
    std::unique_ptr<Tokenizer> mTokenizer;
    Express::VARP mRopeCos;
    Express::VARP mRopeSin;
    Express::VARP mMask;
    Express::Module::Config mModuleConfig;
    std::string mResourcePath;
    MNNForwardType mBackendType;
    int mResidentGroups = 0;
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_MINIMAX_H3_TEXT_ENCODER_HPP
