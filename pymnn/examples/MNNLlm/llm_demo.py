#!/usr/bin/env python3
"""Demonstrate the public MNN.llm API with an exported text model."""

import argparse

import MNN.llm as mllm
import MNN.numpy as np


def demo_basic_generation(model):
    """Generate a buffered response."""
    response = model.response("What is artificial intelligence?")
    print(f"Response: {response}")


def demo_streaming(model):
    """Send generated text directly to stdout."""
    print("Response: ", end="", flush=True)
    model.response("Write a short poem about coding", stream=True)
    print()


def demo_tokenization(model):
    """Encode text and decode each token ID."""
    text = "Hello, how are you?"
    token_ids = model.tokenizer_encode(text)
    decoded = "".join(model.tokenizer_decode(token_id) for token_id in token_ids)
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")


def demo_chat_template(model):
    """Format a list of role/content messages."""
    messages = [{"role": "user", "content": "Explain quantum computing"}]
    formatted = model.apply_chat_template(messages)
    print(f"Formatted prompt: {formatted}")


def demo_low_level_generation(model):
    """Generate at most 64 tokens with manual greedy decoding."""
    prompt = model.apply_chat_template("The capital of France is")
    input_ids = model.tokenizer_encode(prompt)
    model.generate_init()
    output_ids = []
    # The manual loop owns its token budget; forward() does not apply max_new_tokens.
    for _ in range(64):
        logits = model.forward(input_ids)
        token = int(np.argmax(logits))
        model.context.current_token = token
        # Check even the first selected token before decoding or feeding it back.
        if model.stoped():
            break
        output_ids.append(token)
        input_ids = [token]
    response = "".join(model.tokenizer_decode(token_id) for token_id in output_ids)
    print(f"Response: {response}")


def demo_performance_monitoring(model):
    """Read the public Context object after generation."""
    response = model.response("Explain machine learning in one sentence")
    ctx = model.context
    print(f"Response: {response}")
    print(f"Prompt tokens: {ctx.prompt_len}")
    print(f"Generated tokens: {ctx.gen_seq_len}")
    print(f"Prefill time: {ctx.prefill_us / 1000:.2f} ms")
    print(f"Decode time: {ctx.decode_us / 1000:.2f} ms")
    if ctx.decode_us > 0:
        print(f"Generation speed: {ctx.gen_seq_len / (ctx.decode_us / 1e6):.1f} tokens/sec")


def demo_context_properties(model):
    """Inspect status and token counts."""
    model.response("Hello")
    ctx = model.context
    print(f"Status: {ctx.status}")
    print(f"Prompt length: {ctx.prompt_len}")
    print(f"Generated length: {ctx.gen_seq_len}")
    print(f"History tokens: {len(ctx.history_tokens)}")
    print(f"Generation stopped: {model.stoped()}")


def demo_conversation(model):
    """Supply the entire conversation history for each turn."""
    model.set_config({"reuse_kv": False, "use_template": False})
    model.reset()
    messages = []
    try:
        for query in ["My name is Alice. Reply with my name only.", "What is my name? Reply with my name only."]:
            messages.append({"role": "user", "content": query})
            prompt = model.apply_chat_template(messages)
            response = model.response(prompt)
            messages.append({"role": "assistant", "content": response})
            print(f"User: {query}")
            print(f"Bot: {response}")
    finally:
        model.reset()
        model.set_config({"use_template": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Path to an exported text model's config.json")
    arguments = parser.parse_args()

    model = mllm.create(arguments.config_path)
    model.set_config({"max_new_tokens": 64, "reuse_kv": False})
    model.load()
    if model.context.status == mllm.LlmStatus.NOT_LOADED:
        raise RuntimeError("Model loading failed; check the configuration and model files")

    for operation in [
        demo_basic_generation, demo_streaming, demo_tokenization,
        demo_chat_template, demo_performance_monitoring,
        demo_context_properties, demo_conversation, demo_low_level_generation,
    ]:
        model.reset()
        print(f"\n=== {operation.__name__} ===")
        operation(model)

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
