#!/usr/bin/env python3
"""
MNN LLM Module Demo

This script demonstrates Large Language Model inference using MNN.llm:
- Basic text generation
- Streaming output
- Multi-turn conversations
- Performance monitoring
- LoRA adapter usage
"""

import sys
import MNN.llm as mllm

def demo_basic_generation(model):
    """Demo: Basic text generation"""
    print("\n=== Basic Text Generation ===")

    # Simple response - stream=0 for buffered return
    response = model.response("What is artificial intelligence?", 0)
    print(f"Response: {response}\n")

def demo_streaming(model):
    """Demo: Streaming generation for better UX"""
    print("\n=== Streaming Generation ===")
    print("Response: ", end='', flush=True)

    # Note: stream=1 outputs directly to stdout and returns empty string
    # The binding does NOT support token-by-token iteration
    # stream is an int flag: 1 for streaming to stdout, 0 for buffered return
    _ = model.response("Write a short poem about coding", 1)  # stream=1
    print("\n")

def demo_conversation(model):
    """Demo: Multi-turn conversation with context"""
    print("\n=== Multi-Turn Conversation ===")

    # Turn 1: Introduce yourself
    response1 = model.response("My name is Alice and I love programming", 0)
    print(f"User: My name is Alice and I love programming")
    print(f"Bot: {response1}\n")

    # Turn 2: Model should remember context
    response2 = model.response("What is my name and what do I like?", 0)
    print(f"User: What is my name and what do I like?")
    print(f"Bot: {response2}\n")

    # Reset conversation
    model.reset()
    print("Conversation reset. Model will not remember previous context.")

def demo_chat_template(model):
    """Demo: Using chat templates"""
    print("\n=== Chat Template ===")

    # Apply chat template (model-specific formatting)
    raw_prompt = "Explain quantum computing"
    formatted = model.apply_chat_template(raw_prompt)
    print(f"Raw prompt: {raw_prompt}")
    print(f"Formatted: {formatted[:100]}...\n")

def demo_tokenization(model):
    """Demo: Tokenizer encode/decode"""
    print("\n=== Tokenization ===")

    text = "Hello, how are you?"

    # Encode text to token IDs
    token_ids = model.tokenizer_encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {token_ids}\n")

    # Decode back to text
    decoded = model.tokenizer_decode(token_ids)
    print(f"Decoded: {decoded}\n")

def demo_low_level_generation(model):
    """Demo: Low-level token generation"""
    print("\n=== Low-Level Generation ===")

    # Manual token generation (advanced usage)
    prompt = "The capital of France is"

    # Encode
    input_ids = model.tokenizer_encode(prompt)
    print(f"Prompt: {prompt}")
    print(f"Input IDs: {input_ids}")

    # Generate output tokens
    output_ids = model.generate(input_ids)
    print(f"Output IDs: {output_ids}")

    # Decode
    response = model.tokenizer_decode(output_ids)
    print(f"Response: {response}\n")

def demo_performance_monitoring(model):
    """Demo: Monitor generation performance"""
    print("\n=== Performance Monitoring ===")

    # Generate with timing
    prompt = "Explain machine learning in one sentence"
    response = model.response(prompt, 0)  # stream=0 for buffered return

    # Get performance metrics - returns dict, not object
    # llm.h:545 - get_context() method
    ctx = model.get_context()

    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
    print("Performance Metrics:")
    print(f"  Prompt tokens: {ctx['prompt_len']}")
    print(f"  Generated tokens: {ctx['gen_seq_len']}")
    print(f"  Total tokens: {ctx['all_seq_len']}")
    print(f"  Prefill time: {ctx['prefill_us'] / 1000:.2f} ms")
    print(f"  Decode time: {ctx['decode_us'] / 1000:.2f} ms")

    if ctx['decode_us'] > 0:
        tokens_per_sec = ctx['gen_seq_len'] / (ctx['decode_us'] / 1e6)
        print(f"  Generation speed: {tokens_per_sec:.1f} tokens/sec")
    print()

def demo_generation_control(model):
    """Demo: Control generation behavior"""
    print("\n=== Generation Control ===")

    # Generate with stop token
    model.generate_init(None, "\n")  # Stop at newline

    prompt_ids = model.tokenizer_encode("List three colors:")
    output_ids = model.generate(prompt_ids)
    response = model.tokenizer_decode(output_ids)

    print("Prompt: List three colors:")
    print(f"Response (stopped at newline): {response}\n")

def demo_context_properties(model):
    """Demo: Access context properties"""
    print("\n=== Context Properties ===")

    # Generate some text
    model.response("Hello", 0)

    # Get context - returns dict, not object (llm.h:545)
    ctx = model.get_context()
    print("Context State:")
    print(f"  Status: {ctx['status']}")
    print(f"  Current token: {ctx['current_token']}")
    print(f"  Prompt length: {ctx['prompt_len']}")
    print(f"  Generated length: {ctx['gen_seq_len']}")
    print(f"  History tokens: {len(ctx['history_tokens'])} tokens")

    # Check if generation finished
    if model.stoped():  # Note: real method name is stoped (llm.h:549)
        print("  Generation completed")
    print()

def demo_lora_adapters(base_model_path):
    """Demo: LoRA adapter usage (requires LoRA models)"""
    print("\n=== LoRA Adapters ===")
    print("Note: This demo requires pre-trained LoRA adapters")
    print("Example workflow:")
    print("""
    # Load base model
    base = mllm.create('./base-model/config.json')
    base.load()

    # Create split LoRA instance (llm.h:593-611)
    # This is the only LoRA method available in the binding
    math_llm = base.create_lora('./math_lora.mnn')
    math_llm.load()

    # Use the LoRA model
    response = math_llm.response('Solve x^2 - 4x + 4 = 0', 0)
    print(response)

    # Create another LoRA adapter
    code_llm = base.create_lora('./code_lora.mnn')
    code_llm.load()
    response = code_llm.response('Write a quicksort in Python', 0)

    # Note: apply_lora(), select_module(), release_module()
    # are NOT available in the Python binding
    """)

def demo_backend_configuration(model_path):
    """Demo: Configure inference backend"""
    print("\n=== Backend Configuration ===")
    print("Note: Backend configuration is typically done in the model's config.json")
    print("""
    # The config.json file controls:
    # - backend_type: cpu, metal, opencl, vulkan
    # - thread_num: number of threads for CPU inference
    # - precision: normal, low, high
    # - memory_mode: normal, low

    # set_config() method exists but requires a specific string format
    # Format is undocumented - prefer editing config.json directly
    """)

def main():
    print("MNN LLM Module Demo\n")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("Usage: python llm_demo.py <path_to_model_config.json>")
        print("\nExample:")
        print("  python llm_demo.py ./qwen-1.8b-int4/config.json")
        print("\nNote: Download models from MNN Model Zoo:")
        print("  https://www.yuque.com/mnn/en/model_zoo")
        sys.exit(1)

    config_path = sys.argv[1]

    print(f"Loading model: {config_path}")
    print("This may take a few seconds...\n")

    # Create and load model
    model = mllm.create(config_path)
    model.load()

    print("Model loaded successfully!")
    print("=" * 60)

    # Run demos (comment out demos if model is slow)
    try:
        demo_basic_generation(model)
        demo_streaming(model)
        demo_tokenization(model)
        demo_chat_template(model)
        demo_performance_monitoring(model)
        demo_context_properties(model)
        demo_conversation(model)
        demo_generation_control(model)
        demo_low_level_generation(model)

        # Informational demos (no model required)
        demo_lora_adapters(config_path)
        demo_backend_configuration(config_path)

    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Some features may not be supported by this model.")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nFor more information, see:")
    print("  - docs/pymnn/llm.md")
    print("  - https://www.yuque.com/mnn/en/llm")

if __name__ == "__main__":
    main()
