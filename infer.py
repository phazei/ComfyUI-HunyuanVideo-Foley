import os
import argparse
import random
import numpy as np
import torch
import pandas as pd
import torchaudio
from loguru import logger
from hunyuanvideo_foley.utils.model_utils import load_model
from hunyuanvideo_foley.utils.feature_utils import feature_process
from hunyuanvideo_foley.utils.model_utils import denoise_process
from hunyuanvideo_foley.utils.media_utils import merge_audio_video

def set_manual_seed(global_seed):
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

def infer(video_path, prompt, model_dict, cfg, guidance_scale=4.5, num_inference_steps=50, neg_prompt=None):
    visual_feats, text_feats, audio_len_in_s = feature_process(
        video_path,
        prompt,
        model_dict,
        cfg,
        neg_prompt=neg_prompt
    )

    audio, sample_rate = denoise_process(
        visual_feats,
        text_feats,
        audio_len_in_s,
        model_dict,
        cfg,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    return audio[0], sample_rate


def generate_audio(model_dict, cfg, csv_path, output_dir, guidance_scale=4.5, num_inference_steps=50, neg_prompt=None):

    os.makedirs(output_dir, exist_ok=True)
    test_df = pd.read_csv(csv_path)

    for index, row in test_df.iterrows():
        video_path = row['video']
        prompt = row['prompt']

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Prompt: {prompt}")

        output_audio_path = os.path.join(output_dir, f"{index:04d}.wav")
        output_video_path = os.path.join(output_dir, f"{index:04d}.mp4")

        if not os.path.exists(output_audio_path) or not os.path.exists(output_video_path):
            audio, sample_rate = infer(video_path, prompt, model_dict, cfg, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, neg_prompt=neg_prompt)
            torchaudio.save(output_audio_path, audio, sample_rate)

            merge_audio_video(output_audio_path, video_path, output_video_path)

    logger.info(f"All audio files saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo-Foley: Generate audio from video and text prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the pretrained model dir"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        help="Path to the configuration file (.yaml file). If not specified, will be inferred from model_size"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["xl", "xxl"],
        default="xxl",
        help="Model size (xl/xxl). Auto-selects config and model file (default: xxl)"
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--csv_path", 
        type=str,
        help="Path to CSV file containing video paths and text prompts (columns: 'video', 'text')"
    )
    input_group.add_argument(
        "--single_video", 
        type=str,
        help="Path to a single video file for inference"
    )
    parser.add_argument(
        "--single_prompt", 
        type=str,
        help="Text prompt for single video (required when using --single_video)"
    )
    parser.add_argument(
        "--neg_prompt", 
        type=str,
        default=None,
        help="Negative prompt to avoid during generation (default: 'noisy, harsh')"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save generated audio and video files"
    )
    
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=4.5,
        help="Guidance scale for classifier-free guidance (higher = more text adherence)"
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=50,
        help="Number of denoising steps for diffusion sampling"
    )
    parser.add_argument(
        "--audio_length", 
        type=float, 
        default=None,
        help="Maximum audio length in seconds (default: video length)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0,
        help="GPU ID to use when device is cuda"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for processing multiple videos"
    )
    parser.add_argument(
        "--skip_existing", 
        action="store_true",
        help="Skip processing if output files already exist"
    )
    parser.add_argument(
        "--save_video", 
        action="store_true", 
        default=True,
        help="Save video with generated audio merged"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--enable_offload", 
        action="store_true",
        help="Enable model offloading to reduce peak memory usage (good for small VRAM GPUs)"
    )
    
    args = parser.parse_args()
    
    if args.single_video and not args.single_prompt:
        parser.error("--single_prompt is required when using --single_video")
    
    # 如果指定了model_size，自动推断config_path和model文件
    if args.model_size:
        config_mapping = {
            "xl": "configs/hunyuanvideo-foley-xl.yaml",
            "xxl": "configs/hunyuanvideo-foley-xxl.yaml"
        }
        
        if not args.config_path:
            args.config_path = config_mapping[args.model_size]
            logger.info(f"Auto-selected config for {args.model_size} model: {args.config_path}")
    elif not args.config_path:
        args.model_size = "xxl"
        args.config_path = "configs/hunyuanvideo-foley-xxl.yaml"
        logger.info(f"Using default {args.model_size} model: {args.config_path}")
    
    return args


def setup_device(device_str, gpu_id=0):
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"Using CUDA device: {device}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        if device_str == "cuda":
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device(device_str)
        logger.info(f"Using specified device: {device}")
    
    return device


def process_single_video(video_path, prompt, model_dict, cfg, output_dir, args):
    logger.info(f"Processing single video: {video_path}")
    logger.info(f"Text prompt: {prompt}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_audio_path = os.path.join(output_dir, f"{video_name}_generated.wav")
    output_video_path = os.path.join(output_dir, f"{video_name}_with_audio.mp4")
    
    if args.skip_existing and os.path.exists(output_audio_path):
        logger.info(f"Skipping existing audio file: {output_audio_path}")
        if args.save_video and os.path.exists(output_video_path):
            logger.info(f"Skipping existing video file: {output_video_path}")
            return
    
    audio, sample_rate = infer(
        video_path, prompt, model_dict, cfg, 
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        neg_prompt=args.neg_prompt
    )
    
    torchaudio.save(output_audio_path, audio, sample_rate)
    logger.info(f"Audio saved to: {output_audio_path}")
    
    if args.save_video:
        merge_audio_video(output_audio_path, video_path, output_video_path)
        logger.info(f"Video with audio saved to: {output_video_path}")

def main():
    set_manual_seed(1)
    args = parse_args()
    
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level=args.log_level)
    
    device = setup_device(args.device, args.gpu_id)
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        exit(1)
    if not os.path.exists(args.config_path):
        logger.error(f"Config file not found: {args.config_path}")
        exit(1)
    
    if args.csv_path:
        if not os.path.exists(args.csv_path):
            logger.error(f"CSV file not found: {args.csv_path}")
            exit(1)
    elif args.single_video:
        if not os.path.exists(args.single_video):
            logger.error(f"Video file not found: {args.single_video}")
            exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    logger.info("Loading models...")
    model_dict, cfg = load_model(args.model_path, args.config_path, device, enable_offload=args.enable_offload, model_size=args.model_size)
    
    if args.single_video:
        process_single_video(
            args.single_video, args.single_prompt, 
            model_dict, cfg, args.output_dir, args
        )
    else:
        generate_audio(
            model_dict, cfg,
            args.csv_path, args.output_dir,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            neg_prompt=args.neg_prompt
        )
    
    logger.info("Processing completed!")



if __name__ == "__main__":
    main()