import argparse
import importlib
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


WAN_CONFIG_CANDIDATES = (
    "t2v-1.3B",
    "t2v-1.3b",
    "wan2.1-t2v-1.3b",
    "Wan2.1-T2V-1.3B",
)


@dataclass
class WanComponents:
    text_encoder: object
    transformer: object
    vae_decoder: object
    tokenizer: object = None
    source: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Wan2.1-T2V-1.3B components to ONNX for the MNN Wan diffusion runtime."
    )
    parser.add_argument("--model_path", required=True, help="Local Wan checkpoint or official Wan source/checkpoint root.")
    parser.add_argument("--output_path", required=True, help="Directory for ONNX models and tokenizer assets.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--dtype",
        choices=("fp32", "float32", "fp16", "float16"),
        default="fp32",
        help="Export dtype. fp16 requires CUDA for most Wan checkpoints.",
    )
    parser.add_argument("--fp16", action="store_true", help="Alias for --dtype fp16.")
    parser.add_argument("--width", type=int, default=256, help="Export smoke width. Prefer small values for bring-up.")
    parser.add_argument("--height", type=int, default=256, help="Export smoke height. Prefer small values for bring-up.")
    parser.add_argument("--frames", type=int, default=9, help="Export smoke frame count.")
    parser.add_argument("--text_len", type=int, default=512, help="Text token length for the text encoder input.")
    parser.add_argument("--device", default=None, help="Export device. Defaults to cuda for fp16 when available, else cpu.")
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        choices=("text_encoder", "transformer", "vae_decoder", "all"),
        help="Subset of modules to export. Repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--align_manifest",
        help="Optional alignment manifest JSON. When provided, export shapes are resolved from the manifest case.",
    )
    return parser.parse_args()


def normalize_dtype_name(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "float": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "torch.float32": "fp32",
        "half": "fp16",
        "float16": "fp16",
        "fp16": "fp16",
        "torch.float16": "fp16",
    }
    return mapping.get(text, None)


def torch_dtype_from_args(args, torch):
    dtype_name = normalize_dtype_name("fp16" if args.fp16 else args.dtype) or "fp32"
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def add_import_roots(model_path):
    root = Path(model_path).resolve()
    candidates = [root, root.parent, root / "Wan2.1", root / "wan", root / "src"]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def save_tokenizer(tokenizer, output_path):
    if tokenizer is None or not hasattr(tokenizer, "save_pretrained"):
        return
    tokenizer_dir = Path(output_path) / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir.as_posix())


def get_attr_any(obj, names):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def pick_transformer(pipe):
    return get_attr_any(pipe, ("transformer", "model", "dit", "unet"))


def pick_vae_decoder(pipe):
    vae = get_attr_any(pipe, ("vae", "vae_decoder", "autoencoder"))
    if vae is None:
        return None
    return vae


def load_diffusers_pipeline(model_path, dtype, device, errors):
    try:
        diffusers = importlib.import_module("diffusers")
    except Exception as e:
        errors.append(f"diffusers import failed: {e}")
        return None

    pipeline_classes = []
    for name in ("WanPipeline", "AutoPipelineForText2Video"):
        cls = getattr(diffusers, name, None)
        if cls is not None:
            pipeline_classes.append((name, cls))
    if not pipeline_classes:
        errors.append("diffusers is installed but WanPipeline/AutoPipelineForText2Video is not available")
        return None

    for name, cls in pipeline_classes:
        try:
            pipe = cls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
                trust_remote_code=True,
            )
            if hasattr(pipe, "to"):
                pipe = pipe.to(device)
            text_encoder = get_attr_any(pipe, ("text_encoder", "text_encoder_2"))
            transformer = pick_transformer(pipe)
            vae_decoder = pick_vae_decoder(pipe)
            if text_encoder is None or transformer is None or vae_decoder is None:
                errors.append(
                    f"{name} loaded but missing components: "
                    f"text_encoder={text_encoder is not None}, "
                    f"transformer={transformer is not None}, vae={vae_decoder is not None}"
                )
                continue
            tokenizer = get_attr_any(pipe, ("tokenizer", "tokenizer_2"))
            return WanComponents(text_encoder, transformer, vae_decoder, tokenizer, f"diffusers.{name}")
        except Exception as e:
            errors.append(f"diffusers.{name}.from_pretrained failed: {e}")
    return None


def pick_wan_config(configs):
    for key in WAN_CONFIG_CANDIDATES:
        if key in configs:
            return configs[key], key
    if len(configs) == 1:
        key = next(iter(configs.keys()))
        return configs[key], key
    return None, None


def patch_t5_for_export(text_encoder, text_len=512):
    """Patch T5 encoder to pre-compute pos_bias as a buffer for ONNX export."""
    if text_encoder is None:
        return

    try:
        import torch

        # Find the T5Encoder model
        encoder_model = None
        if hasattr(text_encoder, 'model'):
            encoder_model = text_encoder.model
        elif hasattr(text_encoder, 'encoder'):
            encoder_model = text_encoder.encoder

        if encoder_model is None or not hasattr(encoder_model, 'pos_embedding'):
            return

        # Pre-compute pos_bias and register as buffer
        pos_embedding = encoder_model.pos_embedding
        if pos_embedding is not None and hasattr(pos_embedding, 'forward'):
            # Compute pos_bias for the max sequence length (512)
            with torch.no_grad():
                pos_bias = pos_embedding(text_len, text_len)
                # Register as buffer so it becomes a constant in ONNX
                encoder_model.register_buffer('_export_pos_bias', pos_bias)

            # Patch pos_embedding.forward to return the buffer
            original_forward = pos_embedding.forward

            def export_friendly_forward(lq, lk):
                if hasattr(encoder_model, '_export_pos_bias'):
                    return encoder_model._export_pos_bias
                return original_forward(lq, lk)

            pos_embedding.forward = export_friendly_forward

        # Also patch T5Attention to use batch_size=1 for attn_bias
        for module in encoder_model.modules():
            if module.__class__.__name__ == 'T5Attention':
                original_attn_forward = module.forward

                def make_export_friendly_attn_forward(attn_module):
                    def export_friendly_attn_forward(x, context=None, mask=None, pos_bias=None):
                        import torch.nn.functional as F
                        context = x if context is None else context
                        b, n, c = x.size(0), attn_module.num_heads, attn_module.head_dim

                        q = attn_module.q(x).view(b, -1, n, c)
                        k = attn_module.k(context).view(b, -1, n, c)
                        v = attn_module.v(context).view(b, -1, n, c)

                        # Use batch_size=1 for attn_bias
                        attn_bias = x.new_zeros(1, n, q.size(1), k.size(1))
                        if pos_bias is not None:
                            attn_bias = attn_bias + pos_bias
                        if mask is not None:
                            assert mask.ndim in [2, 3]
                            mask_view = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
                            attn_bias = attn_bias.expand(b, -1, -1, -1).clone()
                            attn_bias.masked_fill_(mask_view == 0, torch.finfo(x.dtype).min)
                        else:
                            attn_bias = attn_bias.expand(b, -1, -1, -1)

                        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
                        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
                        x_out = torch.einsum('bnij,bjnc->binc', attn, v)

                        x_out = x_out.reshape(b, -1, n * c)
                        x_out = attn_module.o(x_out)
                        x_out = attn_module.dropout(x_out)
                        return x_out

                    return export_friendly_attn_forward

                module.forward = make_export_friendly_attn_forward(module)

    except Exception as e:
        import traceback
        traceback.print_exc()


def patch_official_wan_for_export(torch, model_mod, transformer, use_cpu_attention=False, attention_mod=None):
    if use_cpu_attention and attention_mod is not None:
        def cpu_attention(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["dtype"] = torch.float32
            kwargs["fa_version"] = None
            return attention_mod.attention(*args, **kwargs)

        model_mod.flash_attention = cpu_attention

    original_rope_apply = getattr(model_mod, "rope_apply", None)
    original_sinusoidal_embedding_1d = getattr(model_mod, "sinusoidal_embedding_1d", None)

    def export_friendly_sinusoidal_embedding_1d(dim, position):
        if original_sinusoidal_embedding_1d is None:
            raise RuntimeError("official Wan sinusoidal_embedding_1d is unavailable")
        assert dim % 2 == 0
        half = dim // 2
        position = position.to(dtype=torch.float32)
        base = torch.arange(half, device=position.device, dtype=torch.float32).div(float(half))
        sinusoid = torch.outer(position, torch.pow(position.new_tensor(10000.0), -base))
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    def export_friendly_rope_apply(x, grid_sizes, freqs):
        if not (isinstance(freqs, torch.Tensor) and freqs.ndim >= 3 and freqs.size(-1) == 2):
            return original_rope_apply(x, grid_sizes, freqs)
        n = x.size(2)
        c = x.size(3) // 2
        split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
        freq_chunks = freqs.split(split_sizes, dim=1)
        outputs = []
        for index, (frames, height, width) in enumerate(grid_sizes.tolist()):
            seq_len = frames * height * width
            x_pairs = x[index, :seq_len].reshape(seq_len, n, -1, 2)
            freqs_i = torch.cat(
                [
                    freq_chunks[0][:frames].view(frames, 1, 1, -1, 2).expand(frames, height, width, -1, 2),
                    freq_chunks[1][:height].view(1, height, 1, -1, 2).expand(frames, height, width, -1, 2),
                    freq_chunks[2][:width].view(1, 1, width, -1, 2).expand(frames, height, width, -1, 2),
                ],
                dim=-2,
            ).reshape(seq_len, 1, -1, 2)
            real = x_pairs[..., 0] * freqs_i[..., 0] - x_pairs[..., 1] * freqs_i[..., 1]
            imag = x_pairs[..., 0] * freqs_i[..., 1] + x_pairs[..., 1] * freqs_i[..., 0]
            rotated = torch.stack((real, imag), dim=-1).flatten(2)
            outputs.append(torch.cat([rotated, x[index, seq_len:]], dim=0))
        return torch.stack(outputs).to(dtype=x.dtype)

    model_mod.sinusoidal_embedding_1d = export_friendly_sinusoidal_embedding_1d
    model_mod.rope_apply = export_friendly_rope_apply
    if hasattr(transformer, "freqs") and isinstance(transformer.freqs, torch.Tensor) and torch.is_complex(transformer.freqs):
        transformer.freqs = torch.view_as_real(transformer.freqs).to(dtype=torch.float32)


def patch_official_vae_for_export(torch, vae_module):
    if vae_module is None:
        return

    functional = importlib.import_module("torch.nn.functional")
    root_module = vae_module
    if not hasattr(root_module, "modules") and hasattr(root_module, "model"):
        root_module = vae_module.model
    if not hasattr(root_module, "modules"):
        return

    for module in root_module.modules():
        mode = getattr(module, "mode", None)
        if mode != "nearest-exact":
            continue

        module.mode = "nearest"

        def export_friendly_forward(x, _module=module):
            return functional.interpolate(
                x.float(),
                size=_module.size,
                scale_factor=_module.scale_factor,
                mode="nearest",
                align_corners=_module.align_corners,
                recompute_scale_factor=_module.recompute_scale_factor,
            ).type_as(x)

        module.forward = export_friendly_forward


def load_official_wan(model_path, dtype, device, errors):
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            torch.cuda.current_device = lambda: 0
    except Exception:
        torch = None
    try:
        wan = importlib.import_module("wan")
    except Exception as e:
        errors.append(f"official wan import failed: {e}")
        return None

    try:
        configs_mod = importlib.import_module("wan.configs")
        configs = getattr(configs_mod, "WAN_CONFIGS", {})
    except Exception as e:
        errors.append(f"wan.configs import failed: {e}")
        configs = {}

    config, config_key = pick_wan_config(configs) if configs else (None, None)
    wan_t2v = getattr(wan, "WanT2V", None)
    if wan_t2v is None:
        try:
            wan_t2v = getattr(importlib.import_module("wan.text2video"), "WanT2V", None)
        except Exception:
            wan_t2v = None
    if wan_t2v is None:
        errors.append("official wan package does not expose WanT2V")
        return None
    if config is None:
        errors.append("official wan package loaded, but no Wan2.1-T2V-1.3B config was found in WAN_CONFIGS")
        return None
    if device == "cpu":
        try:
            t5_mod = importlib.import_module("wan.modules.t5")
            model_mod = importlib.import_module("wan.modules.model")
            attention_mod = importlib.import_module("wan.modules.attention")
            vae_mod = importlib.import_module("wan.modules.vae")
            text_encoder = t5_mod.T5EncoderModel(
                text_len=getattr(config, "text_len", 512),
                dtype=dtype,
                device=torch.device("cpu"),
                checkpoint_path=(Path(model_path) / config.t5_checkpoint).as_posix(),
                tokenizer_path=(Path(model_path) / config.t5_tokenizer).as_posix(),
            )
            patch_t5_for_export(text_encoder, getattr(config, "text_len", 512))
            transformer = model_mod.WanModel.from_pretrained(model_path)
            transformer.eval().requires_grad_(False)
            patch_official_wan_for_export(
                torch,
                model_mod,
                transformer,
                use_cpu_attention=True,
                attention_mod=attention_mod,
            )
            vae_decoder = vae_mod.WanVAE(
                z_dim=getattr(config, "in_dim", 16),
                vae_pth=(Path(model_path) / config.vae_checkpoint).as_posix(),
                dtype=dtype,
                device="cpu",
            )
            patch_official_vae_for_export(torch, vae_decoder)
            tokenizer = get_attr_any(getattr(text_encoder, "tokenizer", None), ("tokenizer",))
            return WanComponents(
                text_encoder,
                transformer,
                vae_decoder,
                tokenizer,
                f"official wan components({config_key})",
            )
        except Exception as e:
            errors.append(f"official Wan component load failed on cpu: {e}")
            return None

    kwargs = {
        "config": config,
        "checkpoint_dir": model_path,
        "rank": 0,
        "t5_fsdp": False,
        "dit_fsdp": False,
        "use_usp": False,
        "t5_cpu": device == "cpu",
    }
    if device.startswith("cuda"):
        try:
            kwargs["device_id"] = int(device.split(":", 1)[1]) if ":" in device else 0
        except ValueError:
            kwargs["device_id"] = 0
    else:
        kwargs["device_id"] = -1

    try:
        signature = inspect.signature(wan_t2v)
        call_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        pipe = wan_t2v(**call_kwargs)
        text_encoder = get_attr_any(pipe, ("text_encoder", "t5", "text_model"))
        transformer = pick_transformer(pipe)
        vae_decoder = pick_vae_decoder(pipe)
        # WanT2V keeps T5/VAE on CPU until called and may load them in bf16; force them to
        # the target device and dtype so downstream wrappers can run consistently.
        inner_t5 = get_attr_any(text_encoder, ("model",))
        if inner_t5 is not None and hasattr(inner_t5, "to"):
            inner_t5.to(device=device, dtype=dtype)
        inner_vae = get_attr_any(vae_decoder, ("model",))
        if inner_vae is not None and hasattr(inner_vae, "to"):
            inner_vae.to(device=device, dtype=dtype)
        if transformer is not None:
            try:
                model_mod = importlib.import_module("wan.modules.model")
                attention_mod = importlib.import_module("wan.modules.attention")
                patch_official_wan_for_export(
                    torch,
                    model_mod,
                    transformer,
                    use_cpu_attention=True,
                    attention_mod=attention_mod,
                )
            except Exception:
                pass
        if text_encoder is None or transformer is None or vae_decoder is None:
            errors.append(
                f"official WanT2V({config_key}) loaded but missing components: "
                f"text_encoder={text_encoder is not None}, "
                f"transformer={transformer is not None}, vae={vae_decoder is not None}"
            )
            return None
        patch_official_vae_for_export(torch, vae_decoder)
        tokenizer = get_attr_any(text_encoder, ("tokenizer",)) or get_attr_any(pipe, ("tokenizer",))
        # Unwrap Wan's HuggingfaceTokenizer wrapper to get the underlying HF tokenizer
        # so build_prompt_batch can use the standard HuggingFace API.
        inner = get_attr_any(tokenizer, ("tokenizer",))
        if inner is not None:
            tokenizer = inner
        return WanComponents(text_encoder, transformer, vae_decoder, tokenizer, f"official wan.WanT2V({config_key})")
    except Exception as e:
        errors.append(f"official WanT2V construction failed: {e}")
        return None


def load_wan_components(model_path, dtype, device):
    add_import_roots(model_path)
    errors = []
    components = load_diffusers_pipeline(model_path, dtype, device, errors)
    if components is not None:
        return components
    components = load_official_wan(model_path, dtype, device, errors)
    if components is not None:
        return components
    detail = "\n  - ".join(errors) if errors else "no loader attempts were made"
    raise RuntimeError(
        "Failed to load Wan2.1-T2V-1.3B from the local environment.\n"
        "Expected either a local diffusers Wan checkpoint or the official Wan code importable from --model_path.\n"
        "Loader errors:\n  - "
        + detail
    )


def maybe_eval(module):
    if hasattr(module, "eval"):
        module.eval()
    return module


def move_to(module, device=None, dtype=None):
    if hasattr(module, "to"):
        try:
            module.to(device=device, dtype=dtype)
        except TypeError:
            try:
                module.to(device)
            except Exception:
                pass
        except Exception:
            pass
    return module


def first_tensor_output(output, preferred_name=None):
    if preferred_name and hasattr(output, preferred_name):
        return getattr(output, preferred_name)
    if isinstance(output, dict):
        if preferred_name and preferred_name in output:
            return output[preferred_name]
        for value in output.values():
            if hasattr(value, "shape"):
                return value
    if isinstance(output, (list, tuple)):
        for value in output:
            if hasattr(value, "shape"):
                return value
    if hasattr(output, "shape"):
        return output
    raise RuntimeError(f"Model output does not contain a tensor: {type(output)}")


def call_with_supported_kwargs(module, positional_args, kwargs):
    try:
        signature = inspect.signature(module.forward if hasattr(module, "forward") else module)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        if accepts_kwargs:
            return module(*positional_args, **kwargs)
        supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return module(*positional_args, **supported)
    except (TypeError, ValueError):
        return module(*positional_args, **kwargs)


class TextEncoderWrapper:
    def __init__(self, torch, text_encoder):
        self.torch = torch
        self.module = maybe_eval(text_encoder)
        self.inner = get_attr_any(text_encoder, ("model", "text_model", "encoder")) or text_encoder
        maybe_eval(self.inner)

    def __call__(self, input_ids):
        attention_mask = self.torch.ones_like(input_ids, dtype=input_ids.dtype)
        attempts = (
            lambda: call_with_supported_kwargs(
                self.inner,
                (),
                {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": False},
            ),
            lambda: call_with_supported_kwargs(
                self.module,
                (),
                {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": False},
            ),
            lambda: self.inner(input_ids, attention_mask),
            lambda: self.inner(input_ids),
        )
        errors = []
        for attempt in attempts:
            try:
                return first_tensor_output(attempt(), "last_hidden_state")
            except Exception as e:
                errors.append(str(e))
        raise RuntimeError("text_encoder export call failed: " + " | ".join(errors))


class TransformerWrapper:
    def __init__(self, torch, transformer):
        self.torch = torch
        self.module = maybe_eval(transformer)

    def _is_official_wan_model(self):
        if getattr(self.module, "model_type", None) is None:
            return False
        if not hasattr(self.module, "patch_size") or not hasattr(self.module, "text_len"):
            return False
        try:
            signature = inspect.signature(self.module.forward)
        except (TypeError, ValueError):
            return False
        return "seq_len" in signature.parameters and "context" in signature.parameters

    def _sequence_length(self, hidden_states):
        patch_size = getattr(self.module, "patch_size", None)
        if isinstance(patch_size, (list, tuple)) and len(patch_size) == 3:
            patch_t = max(1, int(patch_size[0]))
            patch_h = max(1, int(patch_size[1]))
            patch_w = max(1, int(patch_size[2]))
            frames = int(math.ceil(hidden_states.shape[2] / float(patch_t)))
            height = int(math.ceil(hidden_states.shape[3] / float(patch_h)))
            width = int(math.ceil(hidden_states.shape[4] / float(patch_w)))
            return frames * height * width
        return hidden_states.shape[2] * hidden_states.shape[3] * hidden_states.shape[4]

    def _forward_official_wan(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask):
        if not hasattr(self.module, "forward"):
            raise RuntimeError("official Wan forward is unavailable")
        signature = inspect.signature(self.module.forward)
        if "seq_len" not in signature.parameters or "context" not in signature.parameters:
            raise RuntimeError("module forward signature does not match official WanModel")
        batch = hidden_states.shape[0]
        x_list = [hidden_states[index] for index in range(batch)]
        context_list = []
        for index in range(batch):
            valid_tokens = int(encoder_attention_mask[index].sum().item())
            if valid_tokens <= 0:
                valid_tokens = encoder_hidden_states.shape[1]
            context_list.append(encoder_hidden_states[index, :valid_tokens])
        outputs = self.module(
            x_list,
            t=timestep,
            context=context_list,
            seq_len=self._sequence_length(hidden_states),
        )
        if not isinstance(outputs, (list, tuple)):
            raise RuntimeError("official WanModel forward did not return a list")
        return self.torch.stack(list(outputs), dim=0)

    def __call__(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask):
        mask = encoder_attention_mask.to(dtype=encoder_hidden_states.dtype).unsqueeze(-1)
        masked_encoder_hidden_states = encoder_hidden_states * mask
        if self._is_official_wan_model():
            return self._forward_official_wan(
                hidden_states,
                timestep,
                masked_encoder_hidden_states,
                encoder_attention_mask,
            )
        attempts = (
            lambda: call_with_supported_kwargs(
                self.module,
                (),
                {
                    "hidden_states": hidden_states,
                    "sample": hidden_states,
                    "x": hidden_states,
                    "timestep": timestep,
                    "t": timestep,
                    "encoder_hidden_states": masked_encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                    "context": masked_encoder_hidden_states,
                    "seq_len": self._sequence_length(hidden_states),
                    "return_dict": False,
                },
            ),
            lambda: self.module(hidden_states, timestep, masked_encoder_hidden_states, encoder_attention_mask),
            lambda: self.module(hidden_states, timestep, masked_encoder_hidden_states),
        )
        errors = []
        for attempt in attempts:
            try:
                return first_tensor_output(attempt(), "sample")
            except Exception as e:
                errors.append(str(e))
        raise RuntimeError("transformer export call failed: " + " | ".join(errors))


class VaeDecoderWrapper:
    def __init__(self, vae_decoder):
        self.module = maybe_eval(vae_decoder)
        self.config = getattr(vae_decoder, "config", None)

    def _normalize_latents(self, latent_sample):
        if self.config is None:
            return latent_sample
        mean = getattr(self.config, "latents_mean", None)
        std = getattr(self.config, "latents_std", None)
        z_dim = getattr(self.config, "z_dim", None)
        if mean is None or std is None or z_dim is None:
            return latent_sample
        mean = latent_sample.new_tensor(mean).view(1, z_dim, 1, 1, 1)
        inv_std = 1.0 / latent_sample.new_tensor(std).view(1, z_dim, 1, 1, 1)
        # Wan2.1 official de-normalization: z = latent / std + mean
        # Equivalently: z = latent * inv_std + mean
        return latent_sample * inv_std + mean

    @staticmethod
    def _normalize_output(output):
        video = first_tensor_output(output, "sample")
        if hasattr(video, "dim") and video.dim() == 4:
            # Official Wan VAE may return a single [C, T, H, W] tensor. Normalize
            # the exported contract to [B, C, T, H, W] for the C++ frame writer.
            return video.unsqueeze(0)
        return video

    def __call__(self, latent_sample):
        latent_sample = self._normalize_latents(latent_sample)
        attempts = []
        if hasattr(self.module, "decode"):
            attempts.append(lambda: self.module.decode([latent_sample.squeeze(0)])[0].unsqueeze(0))
            attempts.append(lambda: self.module.decode(latent_sample, return_dict=False))
            attempts.append(lambda: self.module.decode(latent_sample))
        attempts.extend(
            (
                lambda: call_with_supported_kwargs(
                    self.module,
                    (),
                    {"latent_sample": latent_sample, "sample": latent_sample, "return_dict": False},
                ),
                lambda: self.module(latent_sample),
            )
        )
        errors = []
        for attempt in attempts:
            try:
                return self._normalize_output(attempt())
            except Exception as e:
                errors.append(str(e))
        raise RuntimeError("vae_decoder export call failed: " + " | ".join(errors))


def make_torch_module(torch, wrapper):
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapper = wrapper
            export_module = None
            if hasattr(wrapper, "module") and isinstance(wrapper.module, torch.nn.Module):
                export_module = wrapper.module
            if export_module is None and hasattr(wrapper, "inner") and isinstance(wrapper.inner, torch.nn.Module):
                export_module = wrapper.inner
            if export_module is not None:
                self.bound_module = export_module

        def forward(self, *args):
            return self.wrapper(*args)

    return Module().eval()


def infer_int_attr(obj, names, default):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if isinstance(value, int):
                return value
        config = getattr(obj, "config", None)
        if config is not None and hasattr(config, name):
            value = getattr(config, name)
            if isinstance(value, int):
                return value
    return default


def infer_text_hidden_size(components, default=4096):
    return infer_int_attr(
        components.text_encoder,
        ("hidden_size", "d_model", "dim", "text_dim", "context_dim", "cross_attention_dim"),
        infer_int_attr(
            components.transformer,
            ("cross_attention_dim", "caption_channels", "text_dim", "context_dim", "encoder_hidden_size"),
            default,
        ),
    )


def infer_latent_channels(components, default=16):
    return infer_int_attr(
        components.transformer,
        ("in_channels", "num_channels", "z_dim", "latent_channels", "in_dim"),
        infer_int_attr(components.vae_decoder, ("z_dim", "latent_channels", "in_channels"), default),
    )


def latent_frame_count(frames):
    return max(1, (frames - 1) // 4 + 1)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def pick_case_dict(data):
    candidates = []
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("alignment_case"),
                data.get("resolved_case"),
                data.get("export_case"),
                data.get("case"),
                data.get("capture"),
                data.get("runtime"),
                data.get("args"),
                data,
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def int_from_case(case_dict, key, default):
    value = case_dict.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def str_from_case(case_dict, key, default=None):
    value = case_dict.get(key, default)
    if value is None:
        return default
    return str(value)


def resolve_alignment_case(args):
    resolved = {
        "cfg_batch": 2,
        "dtype": normalize_dtype_name("fp16" if args.fp16 else args.dtype) or "fp32",
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "text_len": args.text_len,
        "source": "cli",
    }
    if not args.align_manifest:
        return resolved, None
    manifest_path = Path(args.align_manifest).resolve()
    manifest = read_json(manifest_path)
    case_dict = pick_case_dict(manifest)
    cfg_batch = int_from_case(
        case_dict,
        "cfg_batch",
        int_from_case(
            case_dict,
            "text_batch",
            int_from_case(
                case_dict,
                "transformer_batch",
                resolved["cfg_batch"],
            ),
        ),
    )
    resolved.update(
        {
            "cfg_batch": cfg_batch,
            "dtype": normalize_dtype_name(str_from_case(case_dict, "dtype", resolved["dtype"])) or resolved["dtype"],
            "width": int_from_case(case_dict, "width", resolved["width"]),
            "height": int_from_case(case_dict, "height", resolved["height"]),
            "frames": int_from_case(case_dict, "frames", resolved["frames"]),
            "text_len": int_from_case(case_dict, "text_len", resolved["text_len"]),
            "source": manifest_path.as_posix(),
        }
    )
    return resolved, manifest_path


def write_export_report(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def onnx_export(torch, model, args, output_path, input_names, output_names, opset):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args,
        output_path.as_posix(),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def export_models(args):
    import torch

    export_case, manifest_path = resolve_alignment_case(args)
    dtype_name = export_case["dtype"]
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    if args.device:
        device = args.device
    elif dtype == torch.float16 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if dtype == torch.float16 and device == "cpu":
        raise ValueError("fp16 export requires --device cuda or an available CUDA device")

    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output_path).resolve()
    components = load_wan_components(model_path.as_posix(), dtype, device)
    print(f"Loaded Wan components from {components.source}")

    for module in (components.text_encoder, components.transformer, components.vae_decoder):
        move_to(module, device=device, dtype=dtype)
    save_tokenizer(components.tokenizer, output_path)

    latent_h = max(1, export_case["height"] // 8)
    latent_w = max(1, export_case["width"] // 8)
    latent_t = latent_frame_count(export_case["frames"])
    latent_channels = infer_latent_channels(components)
    text_hidden = infer_text_hidden_size(components)

    text_encoder = make_torch_module(torch, TextEncoderWrapper(torch, components.text_encoder))
    transformer = make_torch_module(torch, TransformerWrapper(torch, components.transformer))
    vae_decoder = make_torch_module(torch, VaeDecoderWrapper(components.vae_decoder))
    text_batch = max(1, export_case["cfg_batch"])
    transformer_batch = max(1, export_case["cfg_batch"])
    export_plan = {
        "text_encoder": {
            "module": "text_encoder",
            "path": (output_path / "text_encoder" / "model.onnx").as_posix(),
            "input_names": ["input_ids"],
            "output_names": ["last_hidden_state"],
            "input_shapes": [[text_batch, export_case["text_len"]]],
        },
        "transformer": {
            "module": "transformer",
            "path": (output_path / "transformer" / "model.onnx").as_posix(),
            "input_names": ["hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask"],
            "output_names": ["noise_pred"],
            "input_shapes": [
                [transformer_batch, latent_channels, latent_t, latent_h, latent_w],
                [transformer_batch],
                [transformer_batch, export_case["text_len"], text_hidden],
                [transformer_batch, export_case["text_len"]],
            ],
        },
        "vae_decoder": {
            "module": "vae_decoder",
            "path": (output_path / "vae_decoder" / "model.onnx").as_posix(),
            "input_names": ["latent_sample"],
            "output_names": ["sample"],
            "input_shapes": [[1, latent_channels, latent_t, latent_h, latent_w]],
        },
    }

    selected = set(args.module) if args.module else {"all"}
    if "all" in selected:
        selected = {"text_encoder", "transformer", "vae_decoder"}

    with torch.no_grad():
        if "text_encoder" in selected:
            onnx_export(
                torch,
                text_encoder,
                (torch.ones(text_batch, export_case["text_len"], device=device, dtype=torch.int32),),
                output_path / "text_encoder" / "model.onnx",
                export_plan["text_encoder"]["input_names"],
                export_plan["text_encoder"]["output_names"],
                args.opset,
            )
        if "transformer" in selected:
            onnx_export(
                torch,
                transformer,
                (
                    torch.randn(transformer_batch, latent_channels, latent_t, latent_h, latent_w, device=device,
                                dtype=dtype),
                    torch.zeros(transformer_batch, device=device, dtype=dtype),
                    torch.randn(transformer_batch, export_case["text_len"], text_hidden, device=device, dtype=dtype),
                    torch.ones(transformer_batch, export_case["text_len"], device=device, dtype=torch.int32),
                ),
                output_path / "transformer" / "model.onnx",
                export_plan["transformer"]["input_names"],
                export_plan["transformer"]["output_names"],
                args.opset,
            )
        if "vae_decoder" in selected:
            onnx_export(
                torch,
                vae_decoder,
                (torch.randn(1, latent_channels, latent_t, latent_h, latent_w, device=device, dtype=dtype),),
                output_path / "vae_decoder" / "model.onnx",
                export_plan["vae_decoder"]["input_names"],
                export_plan["vae_decoder"]["output_names"],
                args.opset,
            )

    tokenizer_dir = output_path / "tokenizer"
    tokenizer_saved = tokenizer_dir.exists()
    report = {
        "align_manifest": manifest_path.as_posix() if manifest_path is not None else None,
        "device": device,
        "dtype": dtype_name,
        "frames": export_case["frames"],
        "height": export_case["height"],
        "latent_shape": [transformer_batch, latent_channels, latent_t, latent_h, latent_w],
        "model_path": model_path.as_posix(),
        "opset": args.opset,
        "output_path": output_path.as_posix(),
        "save_tokenizer": tokenizer_saved,
        "resolved_case": {
            "cfg_batch": export_case["cfg_batch"],
            "frames": export_case["frames"],
            "height": export_case["height"],
            "latent_channels": latent_channels,
            "latent_h": latent_h,
            "latent_t": latent_t,
            "latent_w": latent_w,
            "source": export_case["source"],
            "text_batch": text_batch,
            "text_hidden": text_hidden,
            "text_len": export_case["text_len"],
            "transformer_batch": transformer_batch,
            "vae_batch": 1,
            "width": export_case["width"],
        },
        "source": components.source,
        "text_batch": text_batch,
        "text_len": export_case["text_len"],
        "tokenizer_path": tokenizer_dir.as_posix() if tokenizer_saved else None,
        "transformer_batch": transformer_batch,
        "vae_latent_shape": [1, latent_channels, latent_t, latent_h, latent_w],
        "width": export_case["width"],
        "modules": export_plan,
    }
    write_export_report(output_path / "export_report.json", report)
    print(f"Exported Wan ONNX models to {output_path}")


if __name__ == "__main__":
    export_models(parse_args())
