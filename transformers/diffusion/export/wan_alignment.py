import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


MODULE_ORDER = ("vae", "t5", "dit", "e2e")
MODULE_ALIASES = {
    "text_encoder": "t5",
    "transformer": "dit",
    "vae_decoder": "vae",
}
DEFAULT_PROMPT = "A calm ocean wave rolling onto a black sand beach at sunrise."
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_E2E_WIDTH = 256
DEFAULT_E2E_HEIGHT = 256
DEFAULT_E2E_FRAMES = 9
DEFAULT_E2E_STEPS = 2
DEFAULT_CFG_SCALE = 5.0
DEFAULT_TEXT_LEN = 512
MANIFEST_VERSION = 1


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_module_name(name):
    key = name.lower()
    return MODULE_ALIASES.get(key, key)


def normalize_modules(values):
    modules = []
    for value in values or MODULE_ORDER:
        name = normalize_module_name(value)
        if name == "all":
            return list(MODULE_ORDER)
        if name not in MODULE_ORDER:
            raise ValueError(
                "Unsupported module '{}'. Expected one of: all, {}.".format(
                    value, ", ".join(MODULE_ORDER)
                )
            )
        if name not in modules:
            modules.append(name)
    return modules or list(MODULE_ORDER)


def add_common_case_args(parser, include_model_path):
    if include_model_path:
        parser.add_argument("--model_path", required=True, help="Local Wan checkpoint or official Wan source root.")
    parser.add_argument("--output_dir", required=True, help="Directory for golden artifacts and reports.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt for tokenizer/T5 and e2e.")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for CFG.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic latent and tensor generation.")
    parser.add_argument("--width", type=int, default=DEFAULT_E2E_WIDTH, help="Video width.")
    parser.add_argument("--height", type=int, default=DEFAULT_E2E_HEIGHT, help="Video height.")
    parser.add_argument("--frames", type=int, default=DEFAULT_E2E_FRAMES, help="Video frame count.")
    parser.add_argument("--steps", type=int, default=DEFAULT_E2E_STEPS, help="Low-spec e2e denoise steps.")
    parser.add_argument("--cfg_scale", type=float, default=DEFAULT_CFG_SCALE, help="CFG guidance scale.")
    parser.add_argument("--text_len", type=int, default=DEFAULT_TEXT_LEN, help="Tokenizer/T5 max token length.")
    parser.add_argument(
        "--dtype",
        choices=("fp32", "float32", "fp16", "float16"),
        default="fp32",
        help="Reference dtype for torch capture or backend execution metadata.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for reference capture/e2e. Defaults to cuda for fp16 when available, else cpu.",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Subset to run: vae, t5, dit, e2e, or all. Repeatable.",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Wan numerical alignment helper for reference, ONNX, and MNN module/e2e comparisons."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture",
        help="Run the local Wan reference path and save golden inputs/outputs plus manifest.json.",
    )
    add_common_case_args(capture, include_model_path=True)

    compare_onnx = subparsers.add_parser(
        "compare-onnx",
        help="Run exported ONNX models with onnxruntime and compare against an existing manifest.",
    )
    compare_onnx.add_argument("--artifact_dir", required=True, help="Directory containing manifest.json and golden npy files.")
    compare_onnx.add_argument("--onnx_root", required=True, help="Root produced by wan_onnx_export.py.")
    compare_onnx.add_argument("--module", action="append", default=[], help="Subset: vae, t5, dit, e2e, or all.")
    compare_onnx.add_argument("--atol", type=float, default=None, help="Override absolute tolerance.")
    compare_onnx.add_argument("--rtol", type=float, default=None, help="Override relative tolerance.")

    compare_mnn = subparsers.add_parser(
        "compare-mnn",
        help="Run converted MNN modules with PyMNN and compare against an existing manifest.",
    )
    compare_mnn.add_argument("--artifact_dir", required=True, help="Directory containing manifest.json and golden npy files.")
    compare_mnn.add_argument("--mnn_root", required=True, help="Root produced by wan_convert_mnn.py.")
    compare_mnn.add_argument("--module", action="append", default=[], help="Subset: vae, t5, dit, e2e, or all.")
    compare_mnn.add_argument("--atol", type=float, default=None, help="Override absolute tolerance.")
    compare_mnn.add_argument("--rtol", type=float, default=None, help="Override relative tolerance.")
    compare_mnn.add_argument("--thread_num", type=int, default=1, help="PyMNN runtime thread count.")
    compare_mnn.add_argument(
        "--mnn_backend",
        choices=("CPU", "CUDA"),
        default="CPU",
        help="MNN runtime backend (default CPU).",
    )

    e2e = subparsers.add_parser(
        "e2e",
        help="Run the fixed small e2e case on reference, ONNX, or MNN, dump intermediates, and optionally compare.",
    )
    e2e.add_argument("--artifact_dir", required=True, help="Directory containing manifest.json and output reports.")
    e2e.add_argument(
        "--backend",
        required=True,
        choices=("reference", "onnx", "mnn"),
        help="Execution backend for the low-spec e2e rerun.",
    )
    e2e.add_argument("--model_path", help="Local Wan checkpoint root for backend=reference.")
    e2e.add_argument("--onnx_root", help="ONNX root for backend=onnx.")
    e2e.add_argument("--mnn_root", help="MNN root for backend=mnn.")
    e2e.add_argument("--compare", action="store_true", help="Compare the backend e2e dump against the reference manifest.")
    e2e.add_argument("--atol", type=float, default=None, help="Override absolute tolerance.")
    e2e.add_argument("--rtol", type=float, default=None, help="Override relative tolerance.")
    e2e.add_argument("--thread_num", type=int, default=1, help="PyMNN runtime thread count for backend=mnn.")
    e2e.add_argument(
        "--mnn_backend",
        choices=("CPU", "CUDA"),
        default="CPU",
        help="MNN runtime backend for backend=mnn (default CPU).",
    )
    e2e.add_argument(
        "--dtype",
        choices=("fp32", "float32", "fp16", "float16"),
        default="fp32",
        help="Reference dtype for backend=reference.",
    )
    e2e.add_argument("--device", default=None, help="Torch device for backend=reference.")
    return parser


def require_file(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("{} not found: {}".format(description, path))
    return path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=False)
        fp.write("\n")


def write_npy(root, relative_path, array):
    root = Path(root)
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.as_posix(), np.asarray(array))
    array = np.asarray(array)
    return {
        "path": relative_path,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def load_npy(root, entry):
    return np.load((Path(root) / entry["path"]).as_posix(), allow_pickle=False)


def cosine_similarity(lhs, rhs):
    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lhs_norm = np.linalg.norm(lhs)
    rhs_norm = np.linalg.norm(rhs)
    if lhs_norm == 0.0 and rhs_norm == 0.0:
        return 1.0
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))


def compare_arrays(reference, candidate, atol, rtol):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("Shape mismatch: reference={} candidate={}".format(reference.shape, candidate.shape))
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "shape": list(reference.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_candidate": str(candidate.dtype),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "cosine": cosine_similarity(reference, candidate),
        "allclose": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def module_tolerance(module_name, atol_override=None, rtol_override=None):
    base = {
        "vae": (1e-3, 1e-3),
        "t5": (1e-3, 1e-3),
        "dit": (1e-2, 1e-3),
        "e2e": (1e-2, 1e-3),
    }
    atol, rtol = base[module_name]
    if atol_override is not None:
        atol = atol_override
    if rtol_override is not None:
        rtol = rtol_override
    return atol, rtol


def latent_frame_count(frames):
    return max(1, (frames - 1) // 4 + 1)


def resolve_torch_dtype(dtype_name, torch):
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def pick_torch_device(dtype_name, explicit_device, torch):
    if explicit_device:
        return explicit_device
    if dtype_name in ("fp16", "float16") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_wan_export_module():
    module_path = Path(__file__).resolve().parent / "wan_onnx_export.py"
    spec = importlib.util.spec_from_file_location("wan_onnx_export_local", module_path.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load helper module: {}".format(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prompt_batch(tokenizer, prompt, negative_prompt, text_len, torch, device):
    if tokenizer is None:
        raise RuntimeError("Wan loader did not expose a tokenizer; capture requires a local tokenizer.")
    prompts = [negative_prompt, prompt]
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=text_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device=device)
    attention_mask = encoded["attention_mask"].to(device=device)
    if input_ids.shape[0] != 2:
        raise RuntimeError("Expected CFG prompt batch=2, got {}".format(tuple(input_ids.shape)))
    return input_ids, attention_mask


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def deterministic_latent(shape, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32).astype(dtype)


def deterministic_timesteps(steps, shift=3.0):
    """Build the FlowMatch timestep schedule used by Wan2.1 inference.

    This must mirror WanDiffusion::runVideo in C++ exactly so the alignment
    harness reflects real on-device behaviour: linear t in [1.0, 0.001]
    re-mapped through the shifted scheduler then scaled to [0, 1000].
    """
    if steps <= 0:
        raise ValueError("--steps must be > 0")
    if steps == 1:
        return np.array([1000.0], dtype=np.float32)
    values = np.empty(steps, dtype=np.float32)
    for i in range(steps):
        t_linear = 1.0 + i * (0.001 - 1.0) / (steps - 1)
        t_shifted = (shift * t_linear) / (1.0 + (shift - 1.0) * t_linear)
        values[i] = t_shifted * 1000.0
    return values


@dataclass
class ReferenceRunner:
    torch: object
    device: str
    dtype: object
    latent_channels: int
    tokenizer: object
    source: str
    text_encoder: object
    transformer: object
    vae_decoder: object

    def run_t5(self, input_ids):
        tensor = self.torch.from_numpy(np.asarray(input_ids)).to(device=self.device, dtype=self.torch.int32)
        # Free GPU memory: T5 (~25GB fp32) competes with DiT/VAE on a single 32GB card.
        # Move to device just-in-time and offload back to CPU after use.
        is_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
        inner = getattr(self.text_encoder, "inner", None)
        module = getattr(self.text_encoder, "module", None)
        if is_cuda:
            if inner is not None and hasattr(inner, "to"):
                inner.to(device=self.device, dtype=self.dtype)
            if module is not None and module is not inner and hasattr(module, "to"):
                module.to(device=self.device, dtype=self.dtype)
        with self.torch.no_grad():
            result = torch_to_numpy(self.text_encoder(tensor))
        if is_cuda:
            if inner is not None and hasattr(inner, "to"):
                inner.to("cpu")
            if module is not None and module is not inner and hasattr(module, "to"):
                module.to("cpu")
            self.torch.cuda.empty_cache()
        return result

    def run_dit(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask):
        hidden_states = self.torch.from_numpy(np.asarray(hidden_states)).to(device=self.device, dtype=self.dtype)
        timestep = self.torch.from_numpy(np.asarray(timestep)).to(device=self.device, dtype=self.dtype)
        encoder_hidden_states = self.torch.from_numpy(np.asarray(encoder_hidden_states)).to(
            device=self.device, dtype=self.dtype
        )
        encoder_attention_mask = self.torch.from_numpy(np.asarray(encoder_attention_mask)).to(
            device=self.device, dtype=self.torch.int32
        )
        with self.torch.no_grad():
            return torch_to_numpy(
                self.transformer(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask)
            )

    def run_vae(self, latent_sample):
        latent_sample = self.torch.from_numpy(np.asarray(latent_sample)).to(device=self.device, dtype=self.dtype)
        with self.torch.no_grad():
            return torch_to_numpy(self.vae_decoder(latent_sample))


@dataclass
class OnnxRunner:
    ort: object
    text_session: object = None
    dit_session: object = None
    vae_session: object = None

    @classmethod
    def from_root(cls, onnx_root, modules=None):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("compare-onnx requires onnxruntime: {}".format(exc))
        onnx_root = Path(onnx_root)
        modules = set(modules or ("t5", "dit", "vae"))
        providers = ["CPUExecutionProvider"]
        return cls(
            ort=ort,
            text_session=ort.InferenceSession(
                require_file(onnx_root / "text_encoder" / "model.onnx", "text_encoder ONNX").as_posix(),
                providers=providers,
            )
            if "t5" in modules
            else None,
            dit_session=ort.InferenceSession(
                require_file(onnx_root / "transformer" / "model.onnx", "transformer ONNX").as_posix(),
                providers=providers,
            )
            if "dit" in modules
            else None,
            vae_session=ort.InferenceSession(
                require_file(onnx_root / "vae_decoder" / "model.onnx", "vae_decoder ONNX").as_posix(),
                providers=providers,
            )
            if "vae" in modules
            else None,
        )

    @staticmethod
    def _run_single(session, feeds):
        outputs = session.run(None, feeds)
        if not outputs:
            raise RuntimeError("ONNX session produced no outputs for {}".format(session))
        return np.asarray(outputs[0])

    def run_t5(self, input_ids):
        return self._run_single(self.text_session, {"input_ids": np.asarray(input_ids, dtype=np.int32)})

    def run_dit(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask):
        feeds = {
            "hidden_states": np.asarray(hidden_states),
            "timestep": np.asarray(timestep),
            "encoder_hidden_states": np.asarray(encoder_hidden_states),
            "encoder_attention_mask": np.asarray(encoder_attention_mask, dtype=np.int32),
        }
        return self._run_single(self.dit_session, feeds)

    def run_vae(self, latent_sample):
        return self._run_single(self.vae_session, {"latent_sample": np.asarray(latent_sample)})


@dataclass
class MnnRunner:
    MNN: object
    F: object
    nn: object
    text_module: object = None
    dit_module: object = None
    vae_module: object = None

    @classmethod
    def from_root(cls, mnn_root, thread_num=1, modules=None, backend="CPU"):
        try:
            import MNN
            import MNN.expr as F
            import MNN.nn as nn
        except Exception as exc:
            raise RuntimeError("compare-mnn requires PyMNN in PYTHONPATH: {}".format(exc))
        mnn_root = Path(mnn_root)
        modules = set(modules or ("t5", "dit", "vae"))
        runtime_manager = None
        if backend != "CPU":
            runtime_manager = nn.create_runtime_manager([
                {
                    "backend": backend,
                    "precision": "high",
                    "numThread": thread_num,
                }
            ])
        kwargs = {
            "dynamic": False,
            "shape_mutable": False,
            "thread_num": thread_num,
        }
        if runtime_manager is not None:
            kwargs["runtime_manager"] = runtime_manager
        return cls(
            MNN=MNN,
            F=F,
            nn=nn,
            text_module=nn.load_module_from_file(
                require_file(mnn_root / "text_encoder.mnn", "text_encoder MNN").as_posix(),
                ["input_ids"],
                ["last_hidden_state"],
                **kwargs
            )
            if "t5" in modules
            else None,
            dit_module=nn.load_module_from_file(
                require_file(mnn_root / "transformer.mnn", "transformer MNN").as_posix(),
                ["hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask"],
                ["noise_pred"],
                **kwargs
            )
            if "dit" in modules
            else None,
            vae_module=nn.load_module_from_file(
                require_file(mnn_root / "vae_decoder.mnn", "vae_decoder MNN").as_posix(),
                ["latent_sample"],
                ["sample"],
                **kwargs
            )
            if "vae" in modules
            else None,
        )

    def _dtype_to_expr(self, array):
        dtype = np.asarray(array).dtype
        if dtype == np.float64:
            return self.F.double
        if dtype == np.float32 or dtype == np.float16:
            return self.F.float
        if dtype == np.int64:
            return self.F.int64
        if dtype == np.uint8:
            return self.F.uint8
        if dtype == np.int32 or dtype == np.int16 or dtype == np.int8:
            return self.F.int
        raise TypeError("Unsupported numpy dtype for PyMNN input: {}".format(dtype))

    def _make_input_var(self, array, expected_var):
        array = np.ascontiguousarray(array)
        var = self.F.const(array, list(array.shape), self.F.NCHW, self._dtype_to_expr(array))
        if getattr(expected_var, "data_format", None) == self.F.NC4HW4:
            var = self.F.convert(var, self.F.NC4HW4)
        return var

    def _run_single(self, module, arrays):
        info = module.get_info()
        input_vars = info["inputs"]
        if len(input_vars) != len(arrays):
            raise RuntimeError(
                "Input arity mismatch for MNN module. expected {} got {}".format(len(input_vars), len(arrays))
            )
        vars_in = [self._make_input_var(arrays[i], input_vars[i]) for i in range(len(arrays))]
        outputs = module.forward(vars_in if len(vars_in) != 1 else vars_in[0])
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        output = outputs[0]
        if getattr(output, "data_format", None) == self.F.NC4HW4:
            output = self.F.convert(output, self.F.NCHW)
        data = np.array(output.read(), copy=True)
        return data.reshape(output.shape)

    def run_t5(self, input_ids):
        return self._run_single(self.text_module, [np.asarray(input_ids, dtype=np.int32)])

    def run_dit(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask):
        return self._run_single(
            self.dit_module,
            [
                np.asarray(hidden_states),
                np.asarray(timestep),
                np.asarray(encoder_hidden_states),
                np.asarray(encoder_attention_mask, dtype=np.int32),
            ],
        )

    def run_vae(self, latent_sample):
        return self._run_single(self.vae_module, [np.asarray(latent_sample)])


def create_reference_runner(args):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("capture/reference e2e requires torch: {}".format(exc))
    wan_export = load_wan_export_module()
    torch_dtype = resolve_torch_dtype(args.dtype, torch)
    device = pick_torch_device(args.dtype, getattr(args, "device", None), torch)
    if torch_dtype == torch.float16 and device == "cpu":
        raise ValueError("fp16 reference execution requires --device cuda or an available CUDA device")
    components = wan_export.load_wan_components(Path(args.model_path).resolve().as_posix(), torch_dtype, device)
    for module in (components.text_encoder, components.transformer, components.vae_decoder):
        wan_export.move_to(module, device=device, dtype=torch_dtype)
    latent_channels = 16
    if hasattr(wan_export, "infer_latent_channels"):
        latent_channels = int(wan_export.infer_latent_channels(components))
    elif getattr(components.vae_decoder, "config", None) is not None and hasattr(components.vae_decoder.config, "z_dim"):
        latent_channels = int(components.vae_decoder.config.z_dim)
    return ReferenceRunner(
        torch=torch,
        device=device,
        dtype=torch_dtype,
        latent_channels=latent_channels,
        tokenizer=components.tokenizer,
        source=components.source,
        text_encoder=wan_export.TextEncoderWrapper(torch, components.text_encoder),
        transformer=wan_export.TransformerWrapper(torch, components.transformer),
        vae_decoder=wan_export.VaeDecoderWrapper(components.vae_decoder),
    )


def build_case_metadata(args, latent_channels):
    latent_h = max(1, args.height // 8)
    latent_w = max(1, args.width // 8)
    latent_t = latent_frame_count(args.frames)
    return {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": int(args.seed),
        "width": int(args.width),
        "height": int(args.height),
        "frames": int(args.frames),
        "steps": int(args.steps),
        "cfg_scale": float(args.cfg_scale),
        "text_len": int(args.text_len),
        "dtype": args.dtype,
        "latent_channels": int(latent_channels),
        "latent_shape": [1, int(latent_channels), int(latent_t), int(latent_h), int(latent_w)],
        "cfg_batch_order": ["negative", "positive"],
    }


def run_e2e_case(runner, metadata, input_ids, attention_mask):
    text = runner.run_t5(input_ids)
    latent_shape = metadata["latent_shape"]
    sample = deterministic_latent(latent_shape, metadata["seed"])
    timesteps = deterministic_timesteps(metadata["steps"])
    steps = []
    current = sample.astype(np.float32, copy=True)
    for index, timestep_value in enumerate(timesteps):
        sample_before = current.copy()
        hidden_states = np.repeat(sample_before, 2, axis=0)
        timestep = np.full((2,), timestep_value, dtype=np.float32)
        noise_pred = runner.run_dit(hidden_states, timestep, text, attention_mask)
        if noise_pred.shape[0] != 2:
            raise RuntimeError("Expected DiT noise prediction batch=2 for CFG, got {}".format(noise_pred.shape))
        noise_pred_uncond = noise_pred[0:1].astype(np.float32, copy=True)
        noise_pred_cond = noise_pred[1:2].astype(np.float32, copy=True)
        guided_noise = noise_pred_uncond + np.float32(metadata["cfg_scale"]) * (noise_pred_cond - noise_pred_uncond)
        # Match WanDiffusion::stepFlowMatch in C++:
        #   sample + modelOutput * (nextT - t) / 1000.0
        next_timestep_value = float(timesteps[index + 1]) if index + 1 < len(timesteps) else 0.0
        delta = np.float32((next_timestep_value - float(timestep_value)) / 1000.0)
        sample_after = sample_before + delta * guided_noise
        steps.append(
            {
                "index": index,
                "timestep": float(timestep_value),
                "sample_before": sample_before,
                "noise_pred": noise_pred.astype(np.float32, copy=True),
                "noise_pred_uncond": noise_pred_uncond,
                "noise_pred_cond": noise_pred_cond,
                "guided_noise": guided_noise.astype(np.float32, copy=True),
                "sample_after": sample_after.astype(np.float32, copy=True),
            }
        )
        current = sample_after.astype(np.float32, copy=True)
    final_video = runner.run_vae(current)
    return {
        "text_embeddings": text.astype(np.float32, copy=True),
        "initial_latent": sample.astype(np.float32, copy=True),
        "steps": steps,
        "final_latent": current.astype(np.float32, copy=True),
        "final_video": np.asarray(final_video),
    }


def capture_command(args):
    modules = normalize_modules(args.module)
    output_dir = Path(args.output_dir).resolve()
    runner = create_reference_runner(args)
    metadata = build_case_metadata(args, latent_channels=runner.latent_channels)
    input_ids_torch, attention_mask_torch = build_prompt_batch(
        runner.tokenizer, args.prompt, args.negative_prompt, args.text_len, runner.torch, runner.device
    )
    input_ids = torch_to_numpy(input_ids_torch).astype(np.int32, copy=False)
    attention_mask = torch_to_numpy(attention_mask_torch).astype(np.int32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "reference": {
            "source": runner.source,
            "model_path": Path(args.model_path).resolve().as_posix(),
            "device": runner.device,
            "dtype": args.dtype,
        },
        "case": metadata,
        "modules": {},
        "reports": {},
    }

    manifest["modules"]["t5"] = {
        "inputs": {
            "input_ids": write_npy(output_dir, "t5/input_ids.npy", input_ids),
            "attention_mask": write_npy(output_dir, "t5/attention_mask.npy", attention_mask),
        }
    }

    if "t5" in modules or "dit" in modules or "e2e" in modules:
        last_hidden_state = runner.run_t5(input_ids).astype(np.float32, copy=True)
        manifest["modules"]["t5"]["outputs"] = {
            "last_hidden_state": write_npy(output_dir, "t5/last_hidden_state.npy", last_hidden_state)
        }

    if "dit" in modules:
        dit_hidden_states = deterministic_latent(metadata["latent_shape"], args.seed + 101)
        dit_hidden_states = np.repeat(dit_hidden_states, 2, axis=0)
        dit_timestep = np.full((2,), np.float32(0.75), dtype=np.float32)
        dit_output = runner.run_dit(
            dit_hidden_states,
            dit_timestep,
            last_hidden_state,
            attention_mask,
        ).astype(np.float32, copy=True)
        manifest["modules"]["dit"] = {
            "inputs": {
                "hidden_states": write_npy(output_dir, "dit/hidden_states.npy", dit_hidden_states),
                "timestep": write_npy(output_dir, "dit/timestep.npy", dit_timestep),
                "encoder_hidden_states": write_npy(output_dir, "dit/encoder_hidden_states.npy", last_hidden_state),
                "encoder_attention_mask": write_npy(output_dir, "dit/encoder_attention_mask.npy", attention_mask),
            },
            "outputs": {
                "noise_pred": write_npy(output_dir, "dit/noise_pred.npy", dit_output),
            },
        }

    if "vae" in modules:
        vae_latent = deterministic_latent(metadata["latent_shape"], args.seed + 202)
        vae_output = runner.run_vae(vae_latent)
        manifest["modules"]["vae"] = {
            "inputs": {
                "latent_sample": write_npy(output_dir, "vae/latent_sample.npy", vae_latent),
            },
            "outputs": {
                "sample": write_npy(output_dir, "vae/sample.npy", vae_output),
            },
        }

    if "e2e" in modules:
        e2e = run_e2e_case(runner, metadata, input_ids, attention_mask)
        e2e_manifest = {
            "inputs": {
                "input_ids": write_npy(output_dir, "e2e/input_ids.npy", input_ids),
                "attention_mask": write_npy(output_dir, "e2e/attention_mask.npy", attention_mask),
                "initial_latent": write_npy(output_dir, "e2e/initial_latent.npy", e2e["initial_latent"]),
                "text_embeddings": write_npy(output_dir, "e2e/text_embeddings.npy", e2e["text_embeddings"]),
            },
            "steps": [],
            "outputs": {
                "final_latent": write_npy(output_dir, "e2e/final_latent.npy", e2e["final_latent"]),
                "final_video": write_npy(output_dir, "e2e/final_video.npy", e2e["final_video"]),
            },
        }
        for step in e2e["steps"]:
            step_dir = "e2e/steps/step_{:02d}".format(step["index"])
            e2e_manifest["steps"].append(
                {
                    "index": step["index"],
                    "timestep": step["timestep"],
                    "sample_before": write_npy(output_dir, step_dir + "/sample_before.npy", step["sample_before"]),
                    "noise_pred": write_npy(output_dir, step_dir + "/noise_pred.npy", step["noise_pred"]),
                    "noise_pred_uncond": write_npy(
                        output_dir, step_dir + "/noise_pred_uncond.npy", step["noise_pred_uncond"]
                    ),
                    "noise_pred_cond": write_npy(
                        output_dir, step_dir + "/noise_pred_cond.npy", step["noise_pred_cond"]
                    ),
                    "guided_noise": write_npy(output_dir, step_dir + "/guided_noise.npy", step["guided_noise"]),
                    "sample_after": write_npy(output_dir, step_dir + "/sample_after.npy", step["sample_after"]),
                }
            )
        manifest["modules"]["e2e"] = e2e_manifest

    save_json(output_dir / "manifest.json", manifest)
    save_json(
        output_dir / "reports" / "capture_report.json",
        {
            "created_at": utc_now(),
            "output_dir": output_dir.as_posix(),
            "modules": modules,
            "reference": manifest["reference"],
            "case": manifest["case"],
        },
    )
    print("Captured Wan alignment artifacts to {}".format(output_dir))


def load_manifest(artifact_dir):
    artifact_dir = Path(artifact_dir).resolve()
    manifest = load_json(require_file(artifact_dir / "manifest.json", "manifest.json"))
    return artifact_dir, manifest


def manifest_modules(manifest, include_e2e):
    modules = [name for name in MODULE_ORDER if name in manifest.get("modules", {})]
    if not include_e2e:
        modules = [name for name in modules if name != "e2e"]
    return modules


def compare_module_output(artifact_dir, manifest, module_name, candidate_outputs, atol_override=None, rtol_override=None):
    atol, rtol = module_tolerance(module_name, atol_override, rtol_override)
    module_manifest = manifest["modules"].get(module_name)
    if module_manifest is None:
        raise RuntimeError("Manifest does not contain module '{}'".format(module_name))
    outputs = module_manifest["outputs"]
    report = {
        "module": module_name,
        "tensors": {},
        "pass": True,
    }
    for output_name, entry in outputs.items():
        reference = load_npy(artifact_dir, entry)
        candidate = candidate_outputs[output_name]
        metrics = compare_arrays(reference, candidate, atol=atol, rtol=rtol)
        report["tensors"][output_name] = metrics
        report["pass"] = report["pass"] and metrics["allclose"]
    return report


def run_backend_module(artifact_dir, manifest, runner, module_name):
    module = manifest["modules"][module_name]
    if module_name == "t5":
        input_ids = load_npy(artifact_dir, module["inputs"]["input_ids"])
        return {"last_hidden_state": runner.run_t5(input_ids)}
    if module_name == "dit":
        return {
            "noise_pred": runner.run_dit(
                load_npy(artifact_dir, module["inputs"]["hidden_states"]),
                load_npy(artifact_dir, module["inputs"]["timestep"]),
                load_npy(artifact_dir, module["inputs"]["encoder_hidden_states"]),
                load_npy(artifact_dir, module["inputs"]["encoder_attention_mask"]),
            )
        }
    if module_name == "vae":
        return {"sample": runner.run_vae(load_npy(artifact_dir, module["inputs"]["latent_sample"]))}
    raise ValueError("Unsupported module backend run: {}".format(module_name))


def build_runner_from_compare_args(command, args, modules):
    if command == "compare-onnx":
        return OnnxRunner.from_root(args.onnx_root, modules=modules)
    if command == "compare-mnn":
        return MnnRunner.from_root(
            args.mnn_root,
            thread_num=args.thread_num,
            modules=modules,
            backend=getattr(args, "mnn_backend", "CPU"),
        )
    raise ValueError("Unsupported compare command: {}".format(command))


def compare_command(command, args):
    artifact_dir, manifest = load_manifest(args.artifact_dir)
    requested_modules = normalize_modules(args.module) if args.module else manifest_modules(manifest, include_e2e=True)
    modules = [m for m in requested_modules if m != "e2e"]
    runner = build_runner_from_compare_args(command, args, requested_modules)
    report = {
        "created_at": utc_now(),
        "command": command,
        "artifact_dir": artifact_dir.as_posix(),
        "modules": {},
        "pass": True,
    }
    for module_name in modules:
        start = time.time()
        outputs = run_backend_module(artifact_dir, manifest, runner, module_name)
        module_report = compare_module_output(
            artifact_dir, manifest, module_name, outputs, atol_override=args.atol, rtol_override=args.rtol
        )
        module_report["elapsed_ms"] = round((time.time() - start) * 1000.0, 3)
        report["modules"][module_name] = module_report
        report["pass"] = report["pass"] and module_report["pass"]

    if "e2e" in requested_modules:
        e2e_report = e2e_compare(
            artifact_dir,
            manifest,
            runner,
            backend_name="onnx" if command == "compare-onnx" else "mnn",
            atol_override=args.atol,
            rtol_override=args.rtol,
            write_outputs=True,
        )
        report["modules"]["e2e"] = e2e_report
        report["pass"] = report["pass"] and e2e_report["pass"]

    report_path = artifact_dir / "reports" / "{}.json".format(command.replace("-", "_"))
    save_json(report_path, report)
    print("Wrote {}".format(report_path))
    if not report["pass"]:
        raise SystemExit(1)


def materialize_e2e_outputs(artifact_dir, backend_name, e2e_result):
    output_root = Path(artifact_dir) / "reports" / "e2e" / backend_name
    output_root.mkdir(parents=True, exist_ok=True)
    dumped = {
        "text_embeddings": write_npy(output_root, "text_embeddings.npy", e2e_result["text_embeddings"]),
        "initial_latent": write_npy(output_root, "initial_latent.npy", e2e_result["initial_latent"]),
        "final_latent": write_npy(output_root, "final_latent.npy", e2e_result["final_latent"]),
        "final_video": write_npy(output_root, "final_video.npy", e2e_result["final_video"]),
        "steps": [],
    }
    for step in e2e_result["steps"]:
        step_dir = "steps/step_{:02d}".format(step["index"])
        dumped["steps"].append(
            {
                "index": step["index"],
                "timestep": step["timestep"],
                "sample_before": write_npy(output_root, step_dir + "/sample_before.npy", step["sample_before"]),
                "noise_pred": write_npy(output_root, step_dir + "/noise_pred.npy", step["noise_pred"]),
                "noise_pred_uncond": write_npy(output_root, step_dir + "/noise_pred_uncond.npy", step["noise_pred_uncond"]),
                "noise_pred_cond": write_npy(output_root, step_dir + "/noise_pred_cond.npy", step["noise_pred_cond"]),
                "guided_noise": write_npy(output_root, step_dir + "/guided_noise.npy", step["guided_noise"]),
                "sample_after": write_npy(output_root, step_dir + "/sample_after.npy", step["sample_after"]),
            }
        )
    return dumped


def e2e_compare(artifact_dir, manifest, runner, backend_name, atol_override=None, rtol_override=None, write_outputs=False):
    module = manifest["modules"].get("e2e")
    if module is None:
        raise RuntimeError("Manifest does not contain e2e artifacts. Re-run capture with --module e2e or --module all.")
    metadata = manifest["case"]
    input_ids = load_npy(artifact_dir, module["inputs"]["input_ids"])
    attention_mask = load_npy(artifact_dir, module["inputs"]["attention_mask"])
    result = run_e2e_case(runner, metadata, input_ids, attention_mask)
    dumped = materialize_e2e_outputs(artifact_dir, backend_name, result) if write_outputs else None
    atol, rtol = module_tolerance("e2e", atol_override, rtol_override)
    report = {
        "module": "e2e",
        "backend": backend_name,
        "pass": True,
        "tensors": {
            "text_embeddings": compare_arrays(
                load_npy(artifact_dir, module["inputs"]["text_embeddings"]), result["text_embeddings"], atol, rtol
            ),
            "initial_latent": compare_arrays(
                load_npy(artifact_dir, module["inputs"]["initial_latent"]), result["initial_latent"], atol, rtol
            ),
            "final_latent": compare_arrays(
                load_npy(artifact_dir, module["outputs"]["final_latent"]), result["final_latent"], atol, rtol
            ),
            "final_video": compare_arrays(
                load_npy(artifact_dir, module["outputs"]["final_video"]), result["final_video"], atol, rtol
            ),
        },
        "steps": [],
    }
    for name in ("text_embeddings", "initial_latent", "final_latent", "final_video"):
        report["pass"] = report["pass"] and report["tensors"][name]["allclose"]
    if len(module["steps"]) != len(result["steps"]):
        raise RuntimeError(
            "E2E step count mismatch: manifest={} backend={}".format(len(module["steps"]), len(result["steps"]))
        )
    for step_ref, step_result in zip(module["steps"], result["steps"]):
        step_report = {
            "index": step_ref["index"],
            "timestep": step_ref["timestep"],
            "tensors": {},
            "pass": True,
        }
        for tensor_name in (
            "sample_before",
            "noise_pred",
            "noise_pred_uncond",
            "noise_pred_cond",
            "guided_noise",
            "sample_after",
        ):
            metrics = compare_arrays(
                load_npy(artifact_dir, step_ref[tensor_name]),
                step_result[tensor_name],
                atol=atol,
                rtol=rtol,
            )
            step_report["tensors"][tensor_name] = metrics
            step_report["pass"] = step_report["pass"] and metrics["allclose"]
        report["steps"].append(step_report)
        report["pass"] = report["pass"] and step_report["pass"]
    if dumped is not None:
        report["dumped_outputs"] = dumped
    return report


def e2e_command(args):
    artifact_dir, manifest = load_manifest(args.artifact_dir)
    if args.backend == "reference":
        if not args.model_path:
            raise ValueError("--model_path is required for e2e --backend reference")
        runner = create_reference_runner(args)
    elif args.backend == "onnx":
        if not args.onnx_root:
            raise ValueError("--onnx_root is required for e2e --backend onnx")
        runner = OnnxRunner.from_root(args.onnx_root)
    else:
        if not args.mnn_root:
            raise ValueError("--mnn_root is required for e2e --backend mnn")
        runner = MnnRunner.from_root(
            args.mnn_root,
            thread_num=args.thread_num,
            backend=getattr(args, "mnn_backend", "CPU"),
        )
    report = e2e_compare(
        artifact_dir,
        manifest,
        runner,
        backend_name=args.backend,
        atol_override=args.atol,
        rtol_override=args.rtol,
        write_outputs=True,
    )
    out_path = artifact_dir / "reports" / "e2e_{}.json".format(args.backend)
    save_json(
        out_path,
        {
            "created_at": utc_now(),
            "artifact_dir": artifact_dir.as_posix(),
            "report": report,
        },
    )
    print("Wrote {}".format(out_path))
    if args.compare and not report["pass"]:
        raise SystemExit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "capture":
            capture_command(args)
        elif args.command == "compare-onnx":
            compare_command("compare-onnx", args)
        elif args.command == "compare-mnn":
            compare_command("compare-mnn", args)
        elif args.command == "e2e":
            e2e_command(args)
        else:
            parser.error("Unknown command: {}".format(args.command))
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
