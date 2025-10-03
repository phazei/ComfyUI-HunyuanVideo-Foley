# utils.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from diffusers.utils.torch_utils import randn_tensor
from comfy.utils import load_torch_file, ProgressBar

# --- Optional imports from the original HunyuanVideo-Foley package ---
try:
    from hunyuanvideo_foley.models.dac_vae.model.dac import DAC
    from hunyuanvideo_foley.utils.schedulers import FlowMatchDiscreteScheduler
    from hunyuanvideo_foley.utils.feature_utils import (
        encode_video_with_siglip2,
        encode_video_with_sync,
        encode_text_feat,
    )
except Exception:
    # Defer ImportError until the calling site actually uses these helpers.
    DAC = None
    pass

# -----------------------------------------------------------------------------------
# HELPER FUNCTIONS - ADAPTED FOR COMFYUI WORKFLOW
# These are modified versions of the original library's functions to make them
# compatible with ComfyUI's data flow (e.g., accepting a torch.Generator).
# -----------------------------------------------------------------------------------

# DAC kwargs + explicit latent_dim (must be 128 or the decoder mismatches)
# extracted from original pth
_DAC_KWARGS = dict(
    encoder_dim=128,
    encoder_rates=[2, 3, 4, 5, 8],
    latent_dim=128,
    decoder_dim=2048,
    decoder_rates=[8, 5, 4, 3, 2],
    n_codebooks=9,
    codebook_size=1024,
    codebook_dim=8,
    quantizer_dropout=False,
    sample_rate=48000,
    continuous=True,
)

def _tdev(d):  # accept "cpu", "cuda:0", torch.device
    return d if isinstance(d, torch.device) else torch.device(str(d))

def _extract_state(obj):
    # Accept: nn.Module, {"state_dict":..., "metadata":...}, or a flat dict of tensors
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # plain dict of tensors (e.g., safetensors via comfy)
        # keep only tensor entries
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    raise RuntimeError(f"Unsupported checkpoint payload: {type(obj)}")

def load_dac_any(path: str, device="cpu", strict: bool = True):
    """
    Single loader for .pth and .safetensors using the KNOWN, FIXED kwargs.
    No header reads, no inference. We set model.metadata ourselves.
    """
    if DAC is None:
        raise RuntimeError("DAC class import failed")

    dev = _tdev(device)

    # Load payload to CPU (Comfy expects a real torch.device here)
    obj = load_torch_file(path, device=torch.device("cpu"))
    sd = _extract_state(obj)

    # Build exactly the architecture you specified
    model = DAC(**_DAC_KWARGS)
    model.load_state_dict(sd, strict=strict)

    # Put the meta where it goes.
    model.metadata = {
        "kwargs": {**_DAC_KWARGS},
        "converted_from": "vae_128d_48k.pth",
        "format": "pth_or_safetensors",
        "source_path": os.path.basename(path),
    }

    return model.to(dev).eval()

def get_module_size_in_mb(module: nn.Module) -> float:
    """Calculates the total size of a module's parameters in megabytes."""
    total_bytes = 0
    for param in module.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 * 1024)


def _caps(model_dict, cfg):
    tokmax = int(getattr(getattr(model_dict, "clap_tokenizer", None), "model_max_length", 10**9) or 10**9)
    posmax = int(getattr(getattr(getattr(model_dict, "clap_model", None), "config", None), "max_position_embeddings", 10**9) or 10**9)
    cfgmax = int(getattr(getattr(cfg, "model_config", None), "model_kwargs", {}).get("text_length", 10**9))
    return min(tokmax, posmax, cfgmax)


def _pad_or_trim_time(x, T_fixed: int):
    # x: [B, T_cur, D] -> [B, T_fixed, D]
    B, T_cur, D = x.shape
    if T_cur == T_fixed:
        return x
    if T_cur > T_fixed:
        return x[:, :T_fixed, :]
    return F.pad(x, (0, 0, 0, T_fixed - T_cur))


def prepare_latents_with_generator(scheduler, batch_size, num_channels_latents, length, dtype, device, generator=None):
    """Creates the initial random noise tensor using a specified torch.Generator for reproducibility."""
    shape = (batch_size, num_channels_latents, int(length))
    # Use the passed generator for reproducible random noise, compatible with 64-bit seeds.
    latents = randn_tensor(shape, device=device, dtype=dtype, generator=generator)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents


# Denoise keeps fast CFG path; we optimize memory elsewhere (ping-pong + precision + no extra repeats)
def denoise_process_with_generator(
    visual_feats,
    text_feats,
    audio_len_in_s,
    model_dict,
    cfg,
    guidance_scale,
    num_inference_steps,
    batch_size,
    sampler,
    generator=None,
):
    """
    An adaptation of the original denoise_process that accepts a torch.Generator for seeding,
    a sampler/solver name, and uses a ComfyUI progress bar.
    """
    target_dtype = model_dict.foley_model.dtype
    device = model_dict.device

    scheduler = FlowMatchDiscreteScheduler(
        shift=cfg.diffusion_config.sample_flow_shift,
        solver=sampler
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    latents = prepare_latents_with_generator(
        scheduler, batch_size=batch_size,
        num_channels_latents=cfg.model_config.model_kwargs.audio_vae_latent_dim,
        length=audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate,
        dtype=target_dtype, device=device, generator=generator
    )

    # Precompute CFG-invariant feature tensors once outside the loop to reduce allocator churn
    siglip2_feat_rep = visual_feats['siglip2_feat'].repeat(batch_size, 1, 1)
    syncformer_feat_rep = visual_feats['syncformer_feat'].repeat(batch_size, 1, 1)
    text_feat_rep = text_feats['text_feat'].repeat(batch_size, 1, 1)
    uncond_text_rep = text_feats['uncond_text_feat'].repeat(batch_size, 1, 1)

    # --- PAD EMBEDDINGS TOKENZIER ---

    T_cur_len = int(text_feat_rep.shape[1])
    cap   = _caps(model_dict, cfg)

    # Two-bucket policy: 77 normally, 128 if prompt exceeds 77 (respect hard caps)
    if T_cur_len <= 77:
        T_fixed = min(77, cap)
    else:
        T_fixed = min(128, cap)

    # Cache once per session to avoid flapping if prompts bounce around
    if not hasattr(model_dict.foley_model, "_text_len_fixed"):
        model_dict.foley_model._text_len_fixed = T_fixed
    # If you prefer “sticky first bucket,” comment the next line.
    else:
        # stick to bigger bucket if it's triggered
        model_dict.foley_model._text_len_fixed = max(model_dict.foley_model._text_len_fixed, T_fixed)

    T_fixed = model_dict.foley_model._text_len_fixed
    logger.info(f"Using T_FIXED bucket: {T_fixed} (prompt had {T_cur_len} tokens; cap {cap})")

    # Normalize shapes for compile reuse
    text_feat_rep   = _pad_or_trim_time(text_feat_rep,   T_fixed)
    uncond_text_rep = _pad_or_trim_time(uncond_text_rep, T_fixed)

    uncond_siglip2_feat = model_dict.foley_model.get_empty_clip_sequence(bs=batch_size, len=siglip2_feat_rep.shape[1]).to(device)
    uncond_syncformer_feat = model_dict.foley_model.get_empty_sync_sequence(bs=batch_size, len=syncformer_feat_rep.shape[1]).to(device)
    if guidance_scale > 1.0:
        pre_siglip2_input = torch.cat([uncond_siglip2_feat, siglip2_feat_rep])
        pre_sync_input = torch.cat([uncond_syncformer_feat, syncformer_feat_rep])
        pre_text_input = torch.cat([uncond_text_rep, text_feat_rep])
    else:
        pre_siglip2_input = siglip2_feat_rep
        pre_sync_input = syncformer_feat_rep
        pre_text_input = text_feat_rep

    pbar = ProgressBar(len(timesteps))
    with torch.inference_mode():
        for i, t in enumerate(timesteps):
            # Prepare inputs for classifier-free guidance
            latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

            # ---- ensure timestep lives on the SAME device as latents (avoid CPU in graph) ----
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=latent_input.device)
            else:
                t = t.to(device=latent_input.device)
            # expand to batch without materializing CPU intermediates
            t_expand = t.expand(latent_input.shape[0]).contiguous()
            # -----------------------------------------------------------------------------

            # Use precomputed conditional/unconditional features (no per-step rebuild)
            siglip2_feat_input = pre_siglip2_input
            syncformer_feat_input = pre_sync_input
            text_feat_input = pre_text_input

            # Match inputs to the model's actual compute dtype to avoid matmul dtype mismatches
            compute_dtype = next(model_dict.foley_model.parameters()).dtype
            latent_input = latent_input.to(dtype=compute_dtype)
            siglip2_feat_input = siglip2_feat_input.to(dtype=compute_dtype)
            syncformer_feat_input = syncformer_feat_input.to(dtype=compute_dtype)
            text_feat_input = text_feat_input.to(dtype=compute_dtype)

            # Predict the noise residual
            if compute_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type=latent_input.device.type, dtype=compute_dtype):
                    noise_pred = model_dict.foley_model(
                        x=latent_input, t=t_expand, cond=text_feat_input,
                        clip_feat=siglip2_feat_input, sync_feat=syncformer_feat_input
                    )["x"]
            else:
                noise_pred = model_dict.foley_model(
                    x=latent_input, t=t_expand, cond=text_feat_input,
                    clip_feat=siglip2_feat_input, sync_feat=syncformer_feat_input
                )["x"]

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents)[0]
            pbar.update(1)

    # Decode latents to audio waveform
    # Ensure dtype/device match DAC weights to avoid mismatches
    with torch.inference_mode():
        dac_weight = next(model_dict.dac_model.parameters())
        latents_dec = latents.to(device=dac_weight.device, dtype=dac_weight.dtype)
        audio = model_dict.dac_model.decode(latents_dec)

    # Trim to exact length
    audio = audio[:, :int(audio_len_in_s * model_dict.dac_model.sample_rate)]
    return audio, model_dict.dac_model.sample_rate


# Keep preprocessing on CPU; move to device just-in-time inside encode functions
def feature_process_from_tensors(frames_8fps, frames_25fps, prompt, neg_prompt, deps, cfg):
    """
    Helper function takes pre-sampled frame tensors and extracts all necessary features.
    """
    visual_features = {}

    # Process SigLIP2 features (Content analysis) at 8 FPS
    processed_8fps = torch.stack([deps.siglip2_preprocess(frame) for frame in frames_8fps])  # CPU tensors
    # Process Synchformer features (Timing/Sync analysis) at 25 FPS
    processed_25fps = torch.stack([deps.syncformer_preprocess(frame) for frame in frames_25fps])  # CPU tensors

    # Move just-in-time to device for encoding to minimize residency
    processed_8fps_dev = processed_8fps.unsqueeze(0).to(deps.device, non_blocking=True)
    visual_features['siglip2_feat'] = encode_video_with_siglip2(processed_8fps_dev, deps)

    processed_25fps_dev = processed_25fps.unsqueeze(0).to(deps.device, non_blocking=True)
    visual_features['syncformer_feat'] = encode_video_with_sync(processed_25fps_dev, deps)

    # Audio length is determined by the duration of the sync stream (25 FPS)
    audio_len_in_s = frames_25fps.shape[0] / 25.0

    # Process Text features for both positive and negative prompts
    prompts = [neg_prompt, prompt]
    text_feat_res, _ = encode_text_feat(prompts, deps)

    text_feats = {'text_feat': text_feat_res[1:], 'uncond_text_feat': text_feat_res[:1]}

    # Free CPU preprocessing tensors proactively (they can be large)
    del processed_8fps, processed_25fps, processed_8fps_dev, processed_25fps_dev

    return visual_features, text_feats, audio_len_in_s


# -----------------------------------------------------------------------------------
# FP8 WEIGHT-ONLY QUANTIZATION HELPERS (storage in fp8, compute in fp16/bf16)
# -----------------------------------------------------------------------------------
_DENY_SUBSTRINGS = (
    ".bias",            # never quantize biases; they’re tiny and can be precision-sensitive
    ".norm",            # covers LayerNorm/RMSNorm params (e.g., ".norm.weight")
    "q_norm.",          # explicit Q-norms
    "k_norm.",          # explicit K-norms
    "final_layer.",     # keep model output projection high precision
    "visual_proj.",     # keep early visual projection high precision
                        # exclude cross-attn query/proj (both audio & v_cond)
    "audio_cross_q.",
    "v_cond_cross_q.",
    "audio_cross_proj.",
    "v_cond_cross_proj.",
)

# FP8 storage dtypes we support (PyTorch exposes these two).
_FP8_DTYPES = (torch.float8_e5m2, torch.float8_e4m3fn)


class FP8WeightWrapper(nn.Module):
    """
    Minimal unified FP8 storage wrapper for Linear / Conv1d / Conv2d.

    - Stores weights in FP8 (qdtype) as buffers (so they serialize with state_dict).
    - On forward, upcasts weights (and bias if present) to the incoming tensor dtype
      (fp16/bf16/float32) before calling the functional op, so compute stays high precision.
    """
    def __init__(self, mod: nn.Module, qdtype: torch.dtype):
        super().__init__()
        # Identify which op we’re wrapping; needed to pick the correct functional call.
        self.kind = (
            "linear" if isinstance(mod, nn.Linear)
            else "conv1d" if isinstance(mod, nn.Conv1d)
            else "conv2d"
        )
        self.qdtype = qdtype  # target FP8 storage dtype (e5m2 or e4m3fn)

        # Convolution parameters are required to replay the exact conv op at inference.
        if self.kind != "linear":
            self.stride   = mod.stride
            self.padding  = mod.padding
            self.dilation = mod.dilation
            self.groups   = mod.groups

        # Allocate FP8 weight storage (on the same device), then copy from the original module.
        # Using a buffer (not a Parameter) avoids FP8 params flowing through optimizers.
        self.register_buffer(
            "weight",
            mod.weight.detach().to(device=mod.weight.device, dtype=qdtype),
            persistent=True,
        )

        # Keep bias in higher precision (float32) to avoid tiny-scale loss; store as buffer too.
        if mod.bias is None:
            self.bias = None
        else:
            self.register_buffer(
                "bias",
                mod.bias.detach().to(device=mod.bias.device, dtype=torch.float32),
                persistent=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast FP8 storage to the activation's compute dtype (fp16/bf16/fp32)
        w = self.weight.to(dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(dtype=x.dtype)

        if self.kind == "linear":
            return F.linear(x, w, b)

        if self.kind == "conv1d":
            # weight shape: [Cout, Cin_per_group, K], so expected Cin = Cin_per_group * groups
            if x.ndim != 3:
                raise RuntimeError(f"conv1d expects 3D input, got {tuple(x.shape)}")
            expected_Cin = w.shape[1] * self.groups

            # channels-first (N, C, L)
            if x.shape[1] == expected_Cin:
                return F.conv1d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

            # channels-last (N, L, C) → transpose to (N, C, L), conv, then transpose back
            if x.shape[2] == expected_Cin:
                x_t = x.transpose(1, 2)
                y_t = F.conv1d(x_t, w, b, self.stride, self.padding, self.dilation, self.groups)
                return y_t.transpose(1, 2)

            raise RuntimeError(
                f"conv1d channel mismatch: input {tuple(x.shape)}, expected Cin {expected_Cin}"
            )

        # self.kind == "conv2d"
        # weight shape: [Cout, Cin_per_group, kH, kW] → expected Cin = Cin_per_group * groups
        if x.ndim != 4:
            raise RuntimeError(f"conv2d expects 4D input, got {tuple(x.shape)}")
        expected_Cin = w.shape[1] * self.groups

        # channels-first (N, C, H, W)
        if x.shape[1] == expected_Cin:
            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        # channels-last (N, H, W, C) → permute to (N, C, H, W), conv, permute back
        if x.shape[3] == expected_Cin:
            x_t = x.permute(0, 3, 1, 2)
            y_t = F.conv2d(x_t, w, b, self.stride, self.padding, self.dilation, self.groups)
            return y_t.permute(0, 2, 3, 1)

        raise RuntimeError(
            f"conv2d channel mismatch: input {tuple(x.shape)}, expected Cin {expected_Cin}"
        )


def _wrap_fp8_inplace(module: nn.Module, quantization: str = "fp8_e4m3fn", state_dict: dict | None = None):
    """
    Walk the module tree and replace Linear/Conv1d/Conv2d with FP8WeightWrapper.

    - Skips any submodule whose qualified name contains a deny substring.
    - If the checkpoint (state_dict) already has FP8 for <name>.weight, those bytes are copied
      verbatim into the wrapper (no re-encoding). Otherwise, the weight is downcast once to FP8.
    - Compute remains in the activation dtype at runtime (the wrapper upcasts on forward).
    - Returns (counts_per_type, saved_bytes).

    Args:
        module:      root nn.Module to transform in place.
        quantization:"fp8_e5m2" or "fp8_e4m3fn" — the FP8 storage dtype to use when downcasting.
        state_dict:  optional checkpoint tensors to source FP8 bytes from (for exact retention).

    Example:
        counts, saved = _wrap_fp8_inplace(foley_model, "fp8_e5m2", state_dict)
    """
    # Choose FP8 storage dtype based on the string; default path is e4m3fn.
    qdtype = torch.float8_e5m2 if quantization == "fp8_e5m2" else torch.float8_e4m3fn

    # Per-type replacement counters; useful for logging coverage.
    counts = {"linear": 0, "conv1d": 0, "conv2d": 0}

    # Total bytes saved (approx) = sum(original_bytes - fp8_bytes) for each replaced weight.
    saved_bytes = 0

    def _recurse(parent: nn.Module, prefix: str = ""):
        nonlocal saved_bytes
        # Iterate over immediate children so we can replace them in place.
        for name, child in list(parent.named_children()):
            # Qualified name (e.g., "triple_blocks.2.audio_mlp.fc1")
            full = f"{prefix}{name}" if prefix else name

            # Respect deny list: skip wrapping and keep descending into its children.
            if any(tok in full for tok in _DENY_SUBSTRINGS):
                _recurse(child, full)
                continue

            # Decide if this child is one of the supported types we wrap.
            kind = (
                "linear" if isinstance(child, nn.Linear)
                else "conv1d" if isinstance(child, nn.Conv1d)
                else "conv2d" if isinstance(child, nn.Conv2d)
                else None
            )

            if kind is None:
                # Not a target type; recurse to search deeper.
                _recurse(child, full)
                continue

            # Compute original weight footprint in bytes for reporting.
            before = child.weight.numel() * child.weight.element_size()

            # Build a wrapper with FP8 storage, seeded from the current module.
            wrapped = FP8WeightWrapper(child, qdtype)

            # Fast path: if the checkpoint already had FP8 for this exact tensor name,
            # copy those bytes (no re-quantization drift); cast only if FP8 variant differs.
            if state_dict is not None:
                w_src = state_dict.get(f"{full}.weight")
                if isinstance(w_src, torch.Tensor) and w_src.dtype in _FP8_DTYPES:
                    with torch.no_grad():
                        wrapped.weight.copy_(w_src if w_src.dtype == qdtype else w_src.to(qdtype))

            # Replace the child with our FP8 wrapper in the parent module.
            setattr(parent, name, wrapped)

            # Update counters and saved-bytes estimate (FP8 is 1 byte per element).
            counts[kind] += 1
            saved_bytes += max(0, before - wrapped.weight.numel() * 1)

    # Kick off the in-place transformation from the provided root.
    _recurse(module)

    # Return how many modules we wrapped per type and the approximate memory saved.
    return counts, saved_bytes


# -----------------------------------------------------------------------------------
# DTYPE / QUANT DETECTION HELPERS
# -----------------------------------------------------------------------------------

def _detect_ckpt_fp8(state_dict):
    """Return 'fp8_e5m2' / 'fp8_e4m3fn' if any tensor in the checkpoint uses that dtype; else None."""
    detected = None
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float8_e5m2:
                detected = "fp8_e5m2"
                break
            if v.dtype == torch.float8_e4m3fn:
                detected = "fp8_e4m3fn"
                break
    return detected


def _detect_ckpt_major_precision(state_dict):
    """Return torch dtype among {bf16, fp16, fp32} that dominates parameter sizes in the checkpoint."""
    counts = {torch.bfloat16: 0, torch.float16: 0, torch.float32: 0}
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            if v.dtype in counts:
                counts[v.dtype] += v.numel()
    if all(c == 0 for c in counts.values()):
        return torch.bfloat16
    return max(counts, key=counts.get)


# --- HY-FOLEY: during Inductor compile, default tensor factories -> CUDA if unspecified ---
class _CudaFactoriesDuringCompile:
    """
    Scope-limited patch: while active, torch factory calls with no explicit device
    will default to CUDA (if available). This targets Inductor's tiny compile-time
    scratch tensors so it never kicks the CPU codegen path on Windows.
    """
    _NAMES = ("empty", "zeros", "full", "arange", "linspace", "tensor")

    def __enter__(self):
        self.torch = torch
        self.saved = {n: getattr(torch, n) for n in self._NAMES}

        def _wrap(fn):
            def inner(*args, **kwargs):
                # Only add device if missing; no change if caller already set it.
                if "device" not in kwargs and torch.cuda.is_available():
                    kwargs["device"] = "cuda"
                return fn(*args, **kwargs)
            return inner

        for n, fn in self.saved.items():
            setattr(torch, n, _wrap(fn))
        return self

    def __exit__(self, exc_type, exc, tb):
        for n, fn in self.saved.items():
            setattr(self.torch, n, fn)
