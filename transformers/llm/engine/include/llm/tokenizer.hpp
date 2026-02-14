#ifndef MNN_LLM_TOKENIZER_WRAPPER_HPP
#define MNN_LLM_TOKENIZER_WRAPPER_HPP

// Thin wrapper header to expose the internal LLM tokenizer to other components
// such as the diffusion engine, without duplicating implementation code.
//
// The actual implementation lives in the engine/src directory.

#include "../../src/tokenizer.hpp"

#endif // MNN_LLM_TOKENIZER_WRAPPER_HPP
