import sys
import os
import torch
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

def prepare_latents_with_generator(scheduler, batch_size, num_channels_latents, length, dtype, device, generator=None):
    """Creates the initial random noise tensor using a specified torch.Generator for reproducibility."""
    shape = (batch_size, num_channels_latents, int(length))
    # Use the passed generator for reproducible random noise, compatible with 64-bit seeds.
    latents = randn_tensor(shape, device=device, dtype=dtype, generator=generator)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents

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

    pbar = comfy.utils.ProgressBar(len(timesteps))
    for i, t in enumerate(timesteps):
        # Prepare inputs for classifier-free guidance
        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        t_expand = t.repeat(latent_input.shape[0])

        # Prepare conditional and unconditional features
        siglip2_feat = visual_feats['siglip2_feat'].repeat(batch_size, 1, 1)
        uncond_siglip2_feat = model_dict.foley_model.get_empty_clip_sequence(bs=batch_size, len=siglip2_feat.shape[1]).to(device)
        siglip2_feat_input = torch.cat([uncond_siglip2_feat, siglip2_feat]) if guidance_scale > 1.0 else siglip2_feat

        syncformer_feat = visual_feats['syncformer_feat'].repeat(batch_size, 1, 1)
        uncond_syncformer_feat = model_dict.foley_model.get_empty_sync_sequence(bs=batch_size, len=syncformer_feat.shape[1]).to(device)
        syncformer_feat_input = torch.cat([uncond_syncformer_feat, syncformer_feat]) if guidance_scale > 1.0 else syncformer_feat

        text_feat = text_feats['text_feat'].repeat(batch_size, 1, 1)
        uncond_text_feat = text_feats['uncond_text_feat'].repeat(batch_size, 1, 1)
        text_feat_input = torch.cat([uncond_text_feat, text_feat]) if guidance_scale > 1.0 else text_feat

        # Predict the noise residual
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=target_dtype):
            noise_pred = model_dict.foley_model(x=latent_input, t=t_expand, cond=text_feat_input, clip_feat=siglip2_feat_input, sync_feat=syncformer_feat_input)["x"]

        # Perform guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents)[0]
        pbar.update(1)

    # Decode latents to audio waveform
    with torch.no_grad():
        audio = model_dict.dac_model.decode(latents)
    
    # Trim to exact length
    audio = audio[:, :int(audio_len_in_s * model_dict.dac_model.sample_rate)]
    return audio, model_dict.dac_model.sample_rate

def feature_process_from_tensors(frames_8fps, frames_25fps, prompt, neg_prompt, deps, cfg):
    """
    New helper function to replace the old file-based `feature_process`.
    It takes pre-sampled frame tensors and extracts all necessary features.
    """
    visual_features = {}
    
    # Process SigLIP2 features (Content analysis) at 8 FPS
    processed_8fps = torch.stack([deps.siglip2_preprocess(frame) for frame in frames_8fps]).to(deps.device)
    visual_features['siglip2_feat'] = encode_video_with_siglip2(processed_8fps.unsqueeze(0), deps)

    # Process Synchformer features (Timing/Sync analysis) at 25 FPS
    processed_25fps = torch.stack([deps.syncformer_preprocess(frame) for frame in frames_25fps]).to(deps.device)
    visual_features['syncformer_feat'] = encode_video_with_sync(processed_25fps.unsqueeze(0), deps)

    # Audio length is determined by the duration of the sync stream (25 FPS)
    audio_len_in_s = frames_25fps.shape[0] / 25.0
    
    # Process Text features for both positive and negative prompts
    prompts = [neg_prompt, prompt]
    text_feat_res, _ = encode_text_feat(prompts, deps)
    
    text_feats = {'text_feat': text_feat_res[1:], 'uncond_text_feat': text_feat_res[:1]}
    return visual_features, text_feats, audio_len_in_s

# -----------------------------------------------------------------------------------
# NODE 1: Hunyuan Model Loader
# -----------------------------------------------------------------------------------
class HunyuanModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (folder_paths.get_filename_list("foley"),), "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"})}}
    
    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "audio/HunyuanFoley"

    def load_model(self, model_name, precision):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        
        model_path = folder_paths.get_full_path("foley", model_name)
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "hunyuanvideo-foley-xxl.yaml")
        if not os.path.exists(config_path): raise FileNotFoundError(f"Hunyuan config file not found at {config_path}")
        cfg = load_yaml(config_path)
        
        # Load weights onto the offload device first to save VRAM
        state_dict = load_torch_file(model_path, device=offload_device)
        
        # Initialize the model structure on the 'meta' device (no memory allocated yet)
        with init_empty_weights():
             foley_model = HunyuanVideoFoley(cfg, dtype=dtype)
        
        # Materialize the model on the target device with empty tensors
        foley_model.to_empty(device=device)
        
        # Now, load the state dict into the properly materialized model
        foley_model.load_state_dict(state_dict, strict=False)
        foley_model.eval()
        
        logger.info(f"Loaded HunyuanVideoFoley main model: {model_name}")
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
    # NEW: Define the list of available samplers for the dropdown
    SAMPLER_NAMES = ["euler", "heun-2", "midpoint-2", "kutta-4"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "frame_rate": ("INT", {"default": 16, "min": 1, "max": 120, "step": 1, "tooltip": "The framerate of the input image sequence"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 30.0, "step": 0.1, "tooltip": "Duration of the audio to generate in seconds"}),
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

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "hunyuanvideo-foley-xxl.yaml")
        if not os.path.exists(config_path): raise FileNotFoundError(f"Hunyuan config file not found at {config_path}")
        hunyuan_cfg = load_yaml(config_path)
        
        rng = torch.Generator(device="cpu").manual_seed(seed)

        hunyuan_model.to(device)
        for key in ['dac_model', 'syncformer_model', 'siglip2_model', 'clap_model']:
            hunyuan_deps[key].to(device)
        
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
            frames_8fps = image_slice[indices_8fps]
            
            # Resample to 25 FPS for Synchformer (sync analysis)
            indices_25fps = torch.linspace(0, num_frames_to_process - 1, int(duration * 25)).long()
            frames_25fps = image_slice[indices_25fps]

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
            
            visual_feats['siglip2_feat'] = hunyuan_model.get_empty_clip_sequence(bs=1, len=clip_seq_len).to(device)
            visual_feats['syncformer_feat'] = hunyuan_model.get_empty_sync_sequence(bs=1, len=sync_seq_len).to(device)
            
            # Process text features normally
            prompts = [negative_prompt, prompt]
            text_feat_res, _ = encode_text_feat(prompts, hunyuan_deps)
            text_feats = {'text_feat': text_feat_res[1:], 'uncond_text_feat': text_feat_res[:1]}

        model_dict_for_process = hunyuan_deps
        model_dict_for_process['foley_model'] = hunyuan_model
        
        logger.info(f"Generating {audio_len_in_s:.2f}s of audio...")
        logger.debug(f"Visual features keys ready for denoiser: {list(visual_feats.keys())}") # Added for debugging
        
        audio_batch_tensor, sample_rate = denoise_process_with_generator(
            visual_feats, text_feats, audio_len_in_s, 
            model_dict_for_process, hunyuan_cfg,
            guidance_scale=cfg_scale, num_inference_steps=steps, 
            batch_size=batch_size, sampler=sampler, generator=rng
        )

        # --- Model Offloading for VRAM Management ---
        if force_offload:
            logger.info("Offloading models to save VRAM...")
            hunyuan_model.to(offload_device)
            for key in ['dac_model', 'syncformer_model', 'siglip2_model', 'clap_model']:
                hunyuan_deps[key].to(offload_device)
            mm.soft_empty_cache()

        # --- Prepare the two separate outputs ---
        waveform_batch = audio_batch_tensor.float().cpu()

        # Output 1: A standard AUDIO dict with only the first waveform.
        # This is for convenience and direct connection to simple nodes like Preview Audio.
        first_waveform = waveform_batch[0].unsqueeze(0)
        audio_output_first = {"waveform": first_waveform, "sample_rate": sample_rate}
        
        # Output 2: An AUDIO dict containing the entire batch of waveforms.
        # This is for advanced workflows and compatibility with batch-aware nodes.
        audio_output_batch = {"waveform": waveform_batch, "sample_rate": sample_rate}
        
        return (audio_output_first, audio_output_batch)

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
    "SelectAudioFromBatch": SelectAudioFromBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanModelLoader": "Hunyuan-Foley Model Loader",
    "HunyuanDependenciesLoader": "Hunyuan-Foley Dependencies Loader",
    "HunyuanFoleySampler": "Hunyuan-Foley Sampler",
    "SelectAudioFromBatch": "Select Audio From Batch",
}