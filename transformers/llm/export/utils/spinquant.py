

import functools
import math

import torch
import tqdm
import typing
from .hadamard_utils import apply_exact_had_to_linear

def cleanup_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    # elif mode == "hadamard":
    #     return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(
    #     W, had_dim=-1, output=False
    # )  # apply exact (inverse) hadamard on the weights of mlp output (R4^{-1}), R4 not supported yet!
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)

@torch.inference_mode()
def rotate_model(model, rot_path, rot_mode="random"):
    R1 = get_orthogonal_matrix(model.config.hidden_size, rot_mode)
    if rot_path is not None:
        R_cpk = rot_path
        R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if rot_path is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
        else:
            R2 = get_orthogonal_matrix(head_dim, rot_mode)
        rotate_attention_inputs(layers[idx], R1)
        rotate_attention_output(layers[idx], R1)
        rotate_mlp_input(layers[idx], R1)
        rotate_mlp_output(layers[idx], R1)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)
    return R1, R2


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
    kwargs = {"model": model}

    # Embedding fusion
    for W in [model.model.embed_tokens]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = [layer for layer in model.model.layers]

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )

        W_norm = layer.post_attention_layernorm.weight.data
        layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm)
        
    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )
    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)


def fuse_spinquant_rot(model, rot_path):
    if model.lm_head.weight.equal(model.model.embed_tokens.weight):
        print("detach the connection between lm_head weight and embedd_tokens weight")
        model.lm_head.weight = torch.nn.Parameter(
            model.model.embed_tokens.weight.detach().clone()
        )
    model.eval()
    fuse_layer_norms(model)
    R1, R2 = rotate_model(model, rot_path)
    cleanup_memory()
    return model, R1, R2