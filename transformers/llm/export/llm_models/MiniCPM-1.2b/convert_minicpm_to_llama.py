from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
#torch.manual_seed(0)

path = "path-to-MiniCPM-1B-sft-bf16"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True)

responds, history = model.chat(tokenizer, "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？", temperature=0.3, top_p=0.5)
print(responds)


state_dict = model.state_dict()
print(state_dict.keys())

scale_emb = 12
dim_model_base = 256
scale_depth = 1.4
num_layers = 52
hidden_size = 1536

new_emb = state_dict["model.embed_tokens.weight"] * scale_emb
state_dict["model.embed_tokens.weight"] = new_emb

new_emb = state_dict["lm_head.weight"] / (hidden_size / dim_model_base)
state_dict["lm_head.weight"] = new_emb

for i in range(num_layers):
    attn_out_name = f"model.layers.{i}.self_attn.o_proj.weight"
    new_weight = state_dict[attn_out_name] * (scale_depth / math.sqrt(num_layers))
    state_dict[attn_out_name] = new_weight

    ffn_down_proj_name = f"model.layers.{i}.mlp.down_proj.weight"
    new_weight = state_dict[ffn_down_proj_name] * (scale_depth / math.sqrt(num_layers))
    state_dict[ffn_down_proj_name] = new_weight

torch.save(state_dict, "pytorch_model_llama.bin")
