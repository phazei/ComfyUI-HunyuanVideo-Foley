import sys
import os
import time
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from loguru import logger
from torchvision.transforms import v2
from transformers import AutoTokenizer, AutoModel, ClapTextModelWithProjection
from accelerate import init_empty_weights
from diffusers.utils.torch_utils import randn_tensor

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file
import comfy.utils

logger.remove()
logger.add(sys.stdout, level="INFO", format="HunyuanVideo-Foley: {message}")

# --- Add 'foley' models directory to ComfyUI's search paths ---
# This ensures ComfyUI can find models placed in 'ComfyUI/models/foley/'
foley_models_dir = os.path.join(folder_paths.models_dir, "foley")
if "foley" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["foley"] = ([foley_models_dir], folder_paths.supported_pt_extensions)

# --- Import the original, unmodified HunyuanVideo-Foley modules ---
# We treat the original code as an external library to keep our node clean.
try:
    from hunyuanvideo_foley.utils.config_utils import load_yaml, AttributeDict
    from hunyuanvideo_foley.utils.schedulers import FlowMatchDiscreteScheduler
    from hunyuanvideo_foley.models.dac_vae.model.dac import DAC
    from hunyuanvideo_foley.models.synchformer import Synchformer
    from hunyuanvideo_foley.models.hifi_foley import HunyuanVideoFoley
    from hunyuanvideo_foley.utils.feature_utils import encode_video_with_siglip2, encode_video_with_sync, encode_text_feat
except ImportError as e:
    logger.error(f"Failed to import HunyuanVideo-Foley modules: {e}")
    logger.error("Please ensure the ComfyUI_HunyuanVideoFoley custom node is installed correctly.")
    raise

# -----------------------------------------------------------------------------------
# HELPER FUNCTIONS - ADAPTED FOR COMFYUI WORKFLOW
# These are modified versions of the original library's functions to make them
# compatible with ComfyUI's data flow (e.g., accepting a torch.Generator).
# -----------------------------------------------------------------------------------

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
def denoise_process_with_generator(visual_feats, text_feats, audio_len_in_s, model_dict, cfg, guidance_scale, num_inference_steps, batch_size, sampler, generator=None):
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

    pbar = comfy.utils.ProgressBar(len(timesteps))
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
class LinearFP8Wrapper(nn.Module):
    """Wraps a Linear layer with FP8 weight *storage* and safe upcast at forward time.
    - Weight is stored as float8 (e4m3fn/e5m2) to save VRAM.
    - Forward upcasts weight (and bias) to the input dtype for matmul; compute stays in fp16/bf16.
    This avoids unsupported Float8 promotion errors on Ampere while keeping memory wins.
    """
    def __init__(self, in_features:int, out_features:int, bias:torch.Tensor|None, quant_dtype:torch.dtype, device:torch.device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_dtype = quant_dtype
        # Store weight as FP8 on the same device as the original module
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=quant_dtype, device=device), requires_grad=False)
        if bias is not None:
            # Keep bias in higher precision (original dtype) for stability
            self.bias = nn.Parameter(bias.detach().to(device=device), requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, lin:nn.Linear, quant_dtype:torch.dtype):
        dev = lin.weight.device
        mod = cls(lin.in_features, lin.out_features, lin.bias, quant_dtype, dev)
        with torch.no_grad():
            mod.weight.copy_(lin.weight.detach().to(dtype=quant_dtype))
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast weight to input dtype to ensure supported matmul kernels
        w = self.weight.to(dtype=x.dtype)
        b = self.bias
        if b is not None and b.dtype != x.dtype:
            b = b.to(dtype=x.dtype)
        return F.linear(x, w, b)


def _quantize_linears_to_fp8_inplace(module: nn.Module, quantization: str = "fp8_e4m3fn", min_features: int = 0):
    """Recursively replace nn.Linear with LinearFP8Wrapper using the chosen FP8 storage dtype.
    Only weight storage changes; compute remains in the runtime dtype by upcasting in forward.
    """
    if quantization == "fp8_e5m2":
        qdtype = torch.float8_e5m2
    else:
        # default to e4m3fn (balanced dynamic range/precision for activations/weights in practice)
        qdtype = torch.float8_e4m3fn

    count = 0
    saved_bytes = 0

    def _replace(parent: nn.Module):
        nonlocal count, saved_bytes
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                # Skip extremely small linears if desired
                if max(child.in_features, child.out_features) < min_features:
                    continue
                # Estimate memory before/after (weights only)
                orig = child.weight
                before = orig.numel() * orig.element_size()  # bytes
                wrapped = LinearFP8Wrapper.from_linear(child, qdtype)
                setattr(parent, name, wrapped)
                after = wrapped.weight.numel() * 1  # float8 is 1 byte/elt
                count += 1
                saved_bytes += max(0, before - after)
            else:
                _replace(child)

    _replace(module)
    return count, saved_bytes

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

# -----------------------------------------------------------------------------------
# NODE 1: Hunyuan Model Loader (refactored: pure load + wrapper that compiles)
# -----------------------------------------------------------------------------------
class HunyuanModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("foley"),),
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Compute dtype for non-quantized params and autocast (auto = detect from checkpoint)"}),
                "quantization": (["none", "fp8_e4m3fn", "fp8_e5m2", "auto"], {"default": "none", "tooltip": "FP8 weight-only storage for Linear layers, saves a few GB VRAM (compute still fp16/bf16)"}),
            },
            "optional": {
                # Optional input: configuration produced by the Torch Compile node.
                "torch_compile_cfg": ("TORCH_COMPILE_CFG", {"tooltip": "If provided, compile the model using this configuration."}),
                # Optional input for BlockSwap settings.
                "block_swap_args": ("BLOCKSWAPARGS", {"tooltip": "If provided, enables BlockSwap VRAM optimization during sampling."}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "build_model"
    CATEGORY = "audio/HunyuanFoley"

    def load_model(self, model_name, precision, quantization):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        # dtype resolved after checkpoint is loaded if precision == 'auto'
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(precision, torch.bfloat16)

        model_path = folder_paths.get_full_path("foley", model_name)
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "hunyuanvideo-foley-xxl.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Hunyuan config file not found at {config_path}")
        cfg = load_yaml(config_path)

        # Load weights onto the offload device first to save VRAM
        state_dict = load_torch_file(model_path, device=offload_device)

        # Auto-detect quantization and precision from checkpoint
        detected_fp8 = _detect_ckpt_fp8(state_dict)
        if precision == "auto":
            dtype = _detect_ckpt_major_precision(state_dict)
            logger.info(f"Auto precision selected from checkpoint: {str(dtype)}")

        # Initialize the model structure on the 'meta' device (no memory allocated yet)
        with init_empty_weights():
            foley_model = HunyuanVideoFoley(cfg, dtype=dtype)

        # Materialize the model on the offload device (CPU) to avoid VRAM spikes in the loader
        foley_model.to_empty(device=offload_device)

        # Load the state dict into the properly materialized model
        foley_model.load_state_dict(state_dict, strict=False)
        # Ensure the runtime parameter dtype matches the requested precision
        foley_model.to(dtype=dtype)
        foley_model.eval()

        # Optional FP8 weight-only quantization for Linear layers
        if quantization != "none":
            # Choose quantization mode (auto = honor fp8 tensors if present, else default to e4m3fn)
            if quantization == "auto":
                qmode = detected_fp8 if detected_fp8 is not None else "fp8_e4m3fn"
            else:
                qmode = quantization
            count, saved = _quantize_linears_to_fp8_inplace(foley_model, qmode, min_features=0)
            gb = saved / (1024**3)
            logger.info(f"Applied {qmode} weight-only quantization to {count} Linear layers; ~{gb:.2f} GB saved on weights.")

        logger.info(f"Loaded HunyuanVideoFoley main model: {model_name}")
        return foley_model

    def _apply_torch_compile(self, model: nn.Module, compile_cfg: dict):
        """
        Applies torch.compile to the computationally heavy blocks of the model
        instead of the entire model. This improves compilation reliability and
        enables dynamic operations like BlockSwap in the main forward pass.
        
        This method also wraps each compiled block's forward pass to provide
        real-time progress feedback in the console during execution.
        """
        if hasattr(model, "_blocks_are_compiled") and model._blocks_are_compiled:
            logger.info("Model blocks are already compiled. Skipping setup.")
            return model

        try:
            torch._dynamo.config.cache_size_limit = int(compile_cfg.get("dynamo_cache_limit", 64))
        except Exception:
            pass

        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build.")

        # --- Signature Generation Helper Functions (defined inside to be captured by closure) ---
        def _sig(args, kwargs, block_name=None):
            def _meta(o):
                if torch.is_tensor(o):
                    dev = (o.device.type, o.device.index or 0)
                    return (
                        "T",
                        tuple(o.shape),
                        str(o.dtype),
                        dev,
                        tuple(o.stride()),
                        bool(o.requires_grad),
                        bool(o.is_contiguous()),
                    )
                if isinstance(o, (list, tuple)):
                    return tuple(_meta(x) for x in o)
                if isinstance(o, dict):
                    # sort for determinism
                    return tuple(sorted((k, _meta(v)) for k, v in o.items()))
                return (type(o).__name__,)

            meta = (block_name, _meta(args), _meta(kwargs))
            blob = repr(meta).encode()
            # 64-bit stable hash for set membership
            return int.from_bytes(hashlib.blake2s(blob, digest_size=8).digest(), "little")

        # --- JIT Compilation Progress Tracking Setup ---
        model._compilation_progress_counter = [0]
        model._total_blocks_to_compile = len(model.triple_blocks) + len(model.single_blocks)
        model._seen_compile_signatures = set() # This will store the signatures of compiled functions.

        def _create_logged_forward(original_forward_method, block_name, model_ref):
            """
            A wrapper function that intercepts every call to a block's forward method
            to update a progress bar in the console for each denoising step.
            """

            def logged_forward(*args, **kwargs):
                # Calculate the signature of the current inputs.
                current_signature = _sig(args, kwargs, block_name)
                # Check if this specific signature has been compiled before.
                if current_signature not in model_ref._seen_compile_signatures:
                    model_ref._seen_compile_signatures.add(current_signature) # Mark as seen at every end
                    # It's a new signature, so a compilation will happen. Update the progress bar.
                    
                    counter_list = model_ref._compilation_progress_counter
                    total_blocks = model_ref._total_blocks_to_compile

                    # Only show the progress bar for the initial set of compilations.
                    if counter_list[0] < total_blocks:
                    # Increment the global counter every time a block is executed.
                        counter_list[0] += 1

                        # --- ASCII Progress Bar Logic ---
                        # Calculate progress for the current denoising step.
                        progress = counter_list[0] / total_blocks
                        bar_length = 40
                        filled_length = int(bar_length * progress)
                        bar = '█' * filled_length + '─' * (bar_length - filled_length)
                        print(f"\rHunyuanVideo-Foley: JIT Compiling {block_name}... [{bar}] {counter_list[0]}/{total_blocks} ({progress:.0%})", end="", flush=True)
                        if counter_list[0] >= total_blocks:
                            print("\n" + "HunyuanVideo-Foley: JIT Compilation finished.", flush=True)

                with _CudaFactoriesDuringCompile():
                    return original_forward_method(*args, **kwargs)

            return logged_forward

        # --- Main Compilation Logic ---
        backend   = compile_cfg.get("backend", "inductor")
        mode      = compile_cfg.get("mode", "default")
        dynamic   = compile_cfg.get("dynamic", False)  # may be True/False/None
        fullgraph = compile_cfg.get("fullgraph", False)

        logger.info(f"torch.compile transformer blocks with backend='{backend}', mode='{mode}'...")

        # --- Compile and Wrap Triple-Stream Blocks ---
        logger.info(f"{len(model.triple_blocks)} TwoStreamCABlocks...")
        logger.info(f"{len(model.single_blocks)} SingleStreamBlocks...")

        # --- Compile and Wrap Triple-Stream Blocks ---
        for i, block in enumerate(model.triple_blocks):
            original_block = block._orig_mod if hasattr(block, "_orig_mod") else block
            block_name = f"Triple-Stream Block {i+1}"
            try:
                compiled_block = torch.compile(original_block, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
                compiled_block.forward = _create_logged_forward(compiled_block.forward, block_name, model)
                model.triple_blocks[i] = compiled_block
            except Exception as e:
                logger.error(f"Failed to compile {block_name}. Continuing without compiling. Error: {e}")

        # --- Compile and Wrap Single-Stream Blocks ---
        for i, block in enumerate(model.single_blocks):
            original_block = block._orig_mod if hasattr(block, "_orig_mod") else block
            block_name = f"Single-Stream Block {i+1}"
            try:
                compiled_block = torch.compile(original_block, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
                compiled_block.forward = _create_logged_forward(compiled_block.forward, block_name, model)
                model.single_blocks[i] = compiled_block
            except Exception as e:
                logger.error(f"Failed to compile {block_name}. Continuing without compiling. Error: {e}")
        
        model._blocks_are_compiled = True
        return model

    def build_model(self, model_name, precision, quantization, torch_compile_cfg=None, block_swap_args=None):
        # NOTE: block_swap_args are received here but will be used in the sampler node.
        # This is the first step in making the system aware of them.
        foley_model = self.load_model(model_name, precision, quantization)
        
        # total_model_size_mb = get_module_size_in_mb(foley_model)
        # triple_blocks_size_mb = get_module_size_in_mb(foley_model.triple_blocks)
        # single_blocks_size_mb = get_module_size_in_mb(foley_model.single_blocks)
        # total_blocks_size_mb = triple_blocks_size_mb + single_blocks_size_mb
        
        # logger.info(f"--- Model Size Report ---")
        # logger.info(f"Total Model Size: {total_model_size_mb:.2f} MB")
        # logger.info(f"  - Triple-Stream Blocks (19x): {triple_blocks_size_mb:.2f} MB")
        # logger.info(f"  - Single-Stream Blocks (38x): {single_blocks_size_mb:.2f} MB")
        # logger.info(f"  - Total Swappable Block Size: {total_blocks_size_mb:.2f} MB")
        # logger.info(f"  - Non-Block Parameters (Embedders, etc.): {total_model_size_mb - total_blocks_size_mb:.2f} MB")
        # logger.info(f"-------------------------")

        # Add block_swap_args to the model object so the sampler can access it later.
        foley_model.block_swap_args = block_swap_args

        if torch_compile_cfg is not None:
            try:
                foley_model = self._apply_torch_compile(foley_model, torch_compile_cfg)
                logger.info("HunyuanVideoFoley model blocks compiled successfully via TorchCompile configuration.")
            except Exception as e:
                logger.error(f"TorchCompile setup failed; continuing with eager model. Error: {e}")

        return (foley_model,)

# -----------------------------------------------------------------------------------
# NODE 2: Hunyuan Dependencies Loader
# -----------------------------------------------------------------------------------
class HunyuanDependenciesLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": ([f for f in folder_paths.get_filename_list("foley") if "vae" in f],), "synchformer_name": ([f for f in folder_paths.get_filename_list("foley") if "synch" in f],)}}

    RETURN_TYPES = ("HUNYUAN_DEPS",)
    FUNCTION = "load_dependencies"
    CATEGORY = "audio/HunyuanFoley"

    def load_dependencies(self, vae_name, synchformer_name):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        deps = {}

        # Load local model files (VAE, Synchformer)
        deps['dac_model'] = DAC.load(folder_paths.get_full_path("foley", vae_name)).to(offload_device).eval()
        synchformer_sd = load_torch_file(folder_paths.get_full_path("foley", synchformer_name), device=offload_device)
        syncformer_model = Synchformer()
        syncformer_model.load_state_dict(synchformer_sd, strict=False)
        deps['syncformer_model'] = syncformer_model.to(offload_device).eval()

        # Define pure tensor-based v2 preprocessing pipelines
        # SigLIP2 pipeline: The input is a (C,H,W) uint8 tensor.
        deps['siglip2_preprocess'] = v2.Compose([
            v2.Resize((512, 512), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True), # Converts uint8 [0,255] to float [0,1]
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Synchformer pipeline: The input is a (C,H,W) uint8 tensor.
        deps['syncformer_preprocess'] = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True), # Converts uint8 [0,255] to float [0,1]
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Load models from Hugging Face
        deps['siglip2_model'] = AutoModel.from_pretrained("google/siglip2-base-patch16-512").to(offload_device).eval()
        deps['clap_tokenizer'] = AutoTokenizer.from_pretrained("laion/larger_clap_general")
        deps['clap_model'] = ClapTextModelWithProjection.from_pretrained("laion/larger_clap_general").to(offload_device).eval()

        deps['device'] = device

        logger.info("Loaded all HunyuanVideoFoley dependencies.")
        return (AttributeDict(deps),)

# -----------------------------------------------------------------------------------
# NODE 3: Hunyuan Foley Sampler
# -----------------------------------------------------------------------------------
class HunyuanFoleySampler:
    # Define the list of available samplers for the dropdown
    SAMPLER_NAMES = ["euler", "heun-2", "midpoint-2", "kutta-4"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "frame_rate": ("FLOAT", {"default": 16, "min": 1, "max": 120, "step": 0.1, "tooltip": "The framerate of the input image sequence"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 1, "max": 60.0, "step": 0.1, "tooltip": "Duration of the audio to generate in seconds"}),
                "prompt": ("STRING", {"multiline": True, "default": "A person walks on frozen ice"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "noisy, harsh"}),
                "cfg_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Classifier-Free Guidance scale"}),
                "steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1, "tooltip": "Number of denoising steps"}),
                "sampler": (cls.SAMPLER_NAMES, {"default": "euler", "tooltip": "These were included with the official repo, but only Euler seems decent..."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1, "tooltip": "Number of audio variations to generate at once"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload models from VRAM after generation"}),
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio_first", "audio_batch")
    FUNCTION = "generate_audio"
    CATEGORY = "audio/HunyuanFoley"

    def generate_audio(self, hunyuan_model, hunyuan_deps, frame_rate, duration, prompt, negative_prompt, cfg_scale, steps, sampler, batch_size, seed, force_offload, image=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Reset the compilation progress counter at the start of every run.
        if hasattr(hunyuan_model, "_compilation_progress_counter"):
            hunyuan_model._compilation_progress_counter[0] = 0

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "hunyuanvideo-foley-xxl.yaml")
        if not os.path.exists(config_path): raise FileNotFoundError(f"Hunyuan config file not found at {config_path}")
        hunyuan_cfg = load_yaml(config_path)

        rng = torch.Generator(device="cpu").manual_seed(seed)

        # Determine target dtype from main model, and keep it for deps
        target_dtype = hunyuan_model.dtype

        # \- PHASE 1 -------------------------------------------------------
        # Feature extraction on GPU with only extractor models resident
        logger.info("Phase 1: Extracting features (ping-pong on)")

        # Move extractors to GPU in target dtype. Keep tokenizer on CPU.
        for key in ['siglip2_model', 'syncformer_model', 'clap_model']:
            hunyuan_deps[key].to(device=device, dtype=target_dtype)

        visual_feats = {}
        audio_len_in_s = duration

        # Handle optional image input
        if image is not None:
            # --- Frame Preparation and Resampling for Video-to-Audio ---
            logger.info("Image input provided. Running in Video-to-Audio mode.")
            total_input_frames = image.shape[0]
            num_frames_to_process = int(duration * frame_rate)

            # Pad frames by repeating the last frame if the input is shorter than the requested duration
            if num_frames_to_process > total_input_frames:
                logger.warning(f"Requested duration needs {num_frames_to_process} frames, but only {total_input_frames} are available. Padding by holding the last frame.")
                padding_needed = num_frames_to_process - total_input_frames
                last_frame = image[-1:].repeat(padding_needed, 1, 1, 1)
                image_slice_base = image
                image_slice = torch.cat((image_slice_base, last_frame), dim=0)
            else:
                image_slice = image[:num_frames_to_process]

            # Convert ComfyUI's IMAGE tensor (B, H, W, C, float 0-1) to model's expected (T, C, H, W, byte 0-255)
            image_slice = (image_slice * 255.0).byte().permute(0, 3, 1, 2)

            # Resample to 8 FPS for SigLIP2 (content analysis)
            indices_8fps = torch.linspace(0, num_frames_to_process - 1, int(duration * 8)).long()
            indices_8fps = indices_8fps.to(device=image_slice.device, non_blocking=True)
            frames_8fps = image_slice.index_select(0, indices_8fps)

            # Resample to 25 FPS for Synchformer (sync analysis)
            indices_25fps = torch.linspace(0, num_frames_to_process - 1, int(duration * 25)).long()
            indices_25fps = indices_25fps.to(device=image_slice.device, non_blocking=True)
            frames_25fps = image_slice.index_select(0, indices_25fps)

            # Process features from the prepared frames
            visual_feats, text_feats, audio_len_in_s = feature_process_from_tensors(frames_8fps, frames_25fps, prompt, negative_prompt, hunyuan_deps, hunyuan_cfg)

        else:
            # --- Feature Preparation for Text-to-Audio ---
            logger.info("No image input provided. Running in Text-to-Audio mode.")
            # Create empty (zero) tensors for visual features
            clip_seq_len = int(duration * 8)
            num_sync_frames = int(duration * 25)
            num_sync_segments = (num_sync_frames - 16) // 8 + 1
            sync_seq_len = int(num_sync_segments * 8)

            # Keep empty features on CPU for now; move to GPU later just-in-time
            visual_feats['siglip2_feat'] = hunyuan_model.get_empty_clip_sequence(bs=1, len=clip_seq_len).to('cpu', dtype=target_dtype)
            visual_feats['syncformer_feat'] = hunyuan_model.get_empty_sync_sequence(bs=1, len=sync_seq_len).to('cpu', dtype=target_dtype)

            # Process text features normally
            prompts = [negative_prompt, prompt]
            text_feat_res, _ = encode_text_feat(prompts, hunyuan_deps)
            text_feats = {'text_feat': text_feat_res[1:], 'uncond_text_feat': text_feat_res[:1]}

        # Immediately offload extractor models and free cache (ping-pong step)
        for key in ['siglip2_model', 'syncformer_model', 'clap_model']:
            hunyuan_deps[key].to(offload_device)
        mm.soft_empty_cache()
        logger.info("Phase 1 Complete. Extractors offloaded.")

        # Move features to CPU (pinned) to minimize residency between phases
        # Ensure features are in target dtype and pinned for fast H2D copy later
        for k in ['siglip2_feat', 'syncformer_feat']:
            if visual_feats.get(k) is not None:
                visual_feats[k] = visual_feats[k].to('cpu', dtype=target_dtype, copy=True).pin_memory()
        for k in ['text_feat', 'uncond_text_feat']:
            text_feats[k] = text_feats[k].to('cpu', dtype=target_dtype, copy=True).pin_memory()

        # \- PHASE 2 -------------------------------------------------------
        # Denoising with only the main model resident; delay DAC until decode
        logger.info("Phase 2: Denoising with main model")

        # Check for and apply BlockSwap settings before sampling
        if hasattr(hunyuan_model, 'block_swap_args') and hunyuan_model.block_swap_args is not None:
            # The block_swap method will handle device placement itself.
            args = hunyuan_model.block_swap_args
            hunyuan_model.block_swap(
                blocks_to_swap=args.get("blocks_to_swap", 0),
                use_non_blocking=args.get("use_non_blocking", False),
                prefetch_blocks=args.get("prefetch_blocks", 0),
                block_swap_debug=args.get("block_swap_debug", False),
            )
        else:
            # If not used, we must explicitly move the model to the main device.
            hunyuan_model.to(device)

        # Just-in-time copy features to GPU
        visual_feats_gpu = {
            'siglip2_feat': visual_feats['siglip2_feat'].to(device, non_blocking=True),
            'syncformer_feat': visual_feats['syncformer_feat'].to(device, non_blocking=True),
        }
        text_feats_gpu = {
            'text_feat': text_feats['text_feat'].to(device, non_blocking=True),
            'uncond_text_feat': text_feats['uncond_text_feat'].to(device, non_blocking=True),
        }

        # Combine all necessary model components into one dictionary for the denoiser
        # Avoid mutating shared deps; shallow-copy into a fresh AttributeDict for this call
        model_dict_for_process = AttributeDict(dict(hunyuan_deps))
        model_dict_for_process['foley_model'] = hunyuan_model
        model_dict_for_process['device'] = device

        logger.info(f"Generating {audio_len_in_s:.2f}s of audio...")
        logger.debug(f"Visual features keys ready for denoiser: {list(visual_feats_gpu.keys())}")  # Added for debugging

        # Ensure DAC is on GPU (and in a safe dtype) **before** denoise; decode happens inside the denoiser
        hunyuan_deps['dac_model'].to(device=device, dtype=torch.float32)
        # Run the denoising process on the GPU
        decoded_waveform, sample_rate = denoise_process_with_generator(
            visual_feats_gpu, text_feats_gpu, audio_len_in_s,
            model_dict_for_process, hunyuan_cfg,
            guidance_scale=cfg_scale, num_inference_steps=steps,
            batch_size=batch_size, sampler=sampler, generator=rng
        )

        waveform_batch = decoded_waveform.float().cpu()

        # --- Model Offloading for VRAM Management ---
        if force_offload:
            logger.info("Offloading models to save VRAM...")
            hunyuan_model.to(offload_device)
            for key in ['dac_model']:
                hunyuan_deps[key].to(offload_device)
            mm.soft_empty_cache()

        # --- Prepare the two separate outputs ---
        # Output 1: A standard AUDIO dict with only the first waveform.
        # This is for convenience and direct connection to simple nodes like Preview Audio.
        first_waveform = waveform_batch[0].unsqueeze(0)
        audio_output_first = {"waveform": first_waveform, "sample_rate": sample_rate}

        # Output 2: An AUDIO dict containing the entire batch of waveforms.
        # This is for advanced workflows and compatibility with batch-aware nodes.
        audio_output_batch = {"waveform": waveform_batch, "sample_rate": sample_rate}

        return (audio_output_first, audio_output_batch)
    
# -----------------------------------------------------------------------------------
# NODE: Hunyuan Foley Torch Compile (optional accelerator)
# -----------------------------------------------------------------------------------

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


class HunyuanFoleyTorchCompile:
    """Torch Compile.
    
    If you change anything like duration, or batch, it'll compile again and takes about 2 minutes on a 3090.
    Saves about 30% of the time.
    """
    DESCRIPTION = cleandoc(__doc__ or "")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["inductor"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Capture entire graph (stricter); usually keep off"}),
                "mode": (["default", "reduce-overhead", "max-autotune"], {"default": "default"}),
                "dynamic": (["true", "false", "None"], {"default": "false", "tooltip": "Allow shape dynamism; safer when duration/batch vary"}),
                "dynamo_cache_limit": ("INT", {"default": 64, "min": 64, "max": 8192, "step": 64,
                                               "tooltip": "TorchDynamo graph cache size to limit graph explosion"}),
            }
        }

    # Emits a config object to be consumed by the Model Loader
    RETURN_TYPES = ("TORCH_COMPILE_CFG",)
    FUNCTION = "make_config"
    CATEGORY = "audio/HunyuanFoley"

    def make_config(self, backend, mode, dynamic, fullgraph, dynamo_cache_limit):
        # Map tri-state string to Python value
        dyn_map = {"true": True, "false": False, "None": None}
        dynamic_val = dyn_map.get(str(dynamic), False)
        cfg = {
            "backend": backend,
            "mode": mode,
            "dynamic": dynamic_val,   # may be True/False/None
            "fullgraph": fullgraph,
            "dynamo_cache_limit": int(dynamo_cache_limit),
        }
        # returning a plain dict is fine for custom types in Comfy
        return (cfg,)

class HunyuanBlockSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks_to_swap": ("INT", {"default": 30, "min": 0, "max": 57, "step": 1, "tooltip": "Number of transformer blocks to offload to CPU. The model has 57 blocks in total (19 triple-stream + 38 single-stream)."}),
            },
            "optional": {
                # These are added for future compatibility, mirroring WanVideo's options.
                "use_non_blocking": ("BOOLEAN", {"default": False, "tooltip": "Use non-blocking memory transfer for offloading. Can be faster but reserves more RAM."}),
                "prefetch_blocks": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1, "tooltip": "Number of blocks to prefetch to GPU ahead of time. Hides data transfer latency."}),
                "block_swap_debug": ("BOOLEAN", {"default": False, "tooltip": "Enable debug logging for block swapping performance."}),
            },
        }
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "set_args"
    CATEGORY = "audio/HunyuanFoley"
    DESCRIPTION = "Settings for block swapping to reduce VRAM by offloading transformer blocks to CPU."

    def set_args(self, **kwargs):
        # This node simply bundles its arguments into a dictionary.
        return (kwargs,)

# -----------------------------------------------------------------------------------
# HELPER NODE: Select Audio From Batch
# -----------------------------------------------------------------------------------
class SelectAudioFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_batch": ("AUDIO", {"tooltip": "An audio object containing a batch of waveforms."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 63, "tooltip": "The 0-based index of the audio to select from the batch."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "select_audio"
    CATEGORY = "audio/utils"

    def select_audio(self, audio_batch, index):
        waveform_batch = audio_batch['waveform']
        sample_rate = audio_batch['sample_rate']

        # Check if the index is valid
        if index >= waveform_batch.shape[0]:
            logger.warning(f"Index {index} is out of bounds for audio batch of size {waveform_batch.shape[0]}. Clamping to last item.")
            index = waveform_batch.shape[0] - 1

        # Select the waveform at the specified index and keep a batch dimension of 1
        selected_waveform = waveform_batch[index].unsqueeze(0)

        # Package it into the standard AUDIO dictionary format for other nodes
        audio_output = {"waveform": selected_waveform, "sample_rate": sample_rate}
        return (audio_output,)

# -----------------------------------------------------------------------------------
# NODE MAPPINGS - This is how ComfyUI discovers the nodes.
# -----------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "HunyuanModelLoader": HunyuanModelLoader,
    "HunyuanDependenciesLoader": HunyuanDependenciesLoader,
    "HunyuanFoleySampler": HunyuanFoleySampler,
    "HunyuanFoleyTorchCompile": HunyuanFoleyTorchCompile,
    "HunyuanBlockSwap": HunyuanBlockSwap,
    "SelectAudioFromBatch": SelectAudioFromBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanModelLoader": "Hunyuan-Foley Model Loader",
    "HunyuanDependenciesLoader": "Hunyuan-Foley Dependencies Loader",
    "HunyuanFoleySampler": "Hunyuan-Foley Sampler",
    "HunyuanFoleyTorchCompile": "Hunyuan-Foley Torch Compile",
    "HunyuanBlockSwap": "Hunyuan-Foley BlockSwap Settings",
    "SelectAudioFromBatch": "Select Audio From Batch",
}
