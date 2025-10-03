#!/usr/bin/env python3
"""
Command Line Interface for HunyuanVideo-Foley

Provides command-line access to the main inference functionality.
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HunyuanVideo-Foley: Generate Foley audio from video and text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video generation
  hunyuanvideo-foley --model_path ./models --single_video video.mp4 --single_prompt "footsteps on gravel"
  
  # Batch processing
  hunyuanvideo-foley --model_path ./models --csv_path batch.csv --output_dir ./outputs
  
  # Start Gradio interface  
  hunyuanvideo-foley --gradio --model_path ./models
        """
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the pretrained model directory")
    parser.add_argument("--config_path", type=str, 
                       default="configs/hunyuanvideo-foley-xxl.yaml",
                       help="Path to the model configuration file")
    
    # Input options
    group_input = parser.add_mutually_exclusive_group(required=True)
    group_input.add_argument("--single_video", type=str,
                           help="Path to single video file for processing")
    group_input.add_argument("--csv_path", type=str,
                           help="Path to CSV file with video paths and prompts")
    group_input.add_argument("--gradio", action="store_true",
                           help="Launch Gradio web interface")
    
    # Generation options
    parser.add_argument("--single_prompt", type=str,
                       help="Text prompt for single video (required with --single_video)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for generated audio files")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                       help="Guidance scale for generation (default: 4.5)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps (default: 50)")
    parser.add_argument("--neg_prompt", type=str,
                       help="Negative prompt to avoid certain audio characteristics")
    
    # System options
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for inference")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible generation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.single_video and not args.single_prompt:
        parser.error("--single_prompt is required when using --single_video")
    
    # Import here to avoid import errors if dependencies are missing
    try:
        if args.gradio:
            _launch_gradio(args)
        elif args.single_video:
            _process_single_video(args) 
        elif args.csv_path:
            _process_batch(args)
    except ImportError as e:
        print(f"Error: Missing required dependencies. Please install with: pip install hunyuanvideo-foley[all]")
        print(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def _launch_gradio(args):
    """Launch Gradio web interface."""
    import os
    os.environ["HIFI_FOLEY_MODEL_PATH"] = args.model_path
    
    # Import and launch gradio app
    import subprocess
    gradio_script = Path(__file__).parent.parent / "gradio_app.py"
    subprocess.run([sys.executable, str(gradio_script)])

def _process_single_video(args):
    """Process a single video file."""
    from . import infer
    
    print(f"Processing video: {args.single_video}")
    print(f"Prompt: {args.single_prompt}")
    
    # This would need to be implemented to match the actual infer.py interface
    # For now, redirect to the original script
    import subprocess
    cmd = [
        sys.executable, "infer.py",
        "--model_path", args.model_path,
        "--config_path", args.config_path,
        "--single_video", args.single_video,
        "--single_prompt", args.single_prompt,
        "--output_dir", args.output_dir,
        "--guidance_scale", str(args.guidance_scale),
        "--num_inference_steps", str(args.num_inference_steps)
    ]
    if args.neg_prompt:
        cmd.extend(["--neg_prompt", args.neg_prompt])
    
    subprocess.run(cmd)

def _process_batch(args):
    """Process a batch of videos from CSV."""
    import subprocess
    cmd = [
        sys.executable, "infer.py", 
        "--model_path", args.model_path,
        "--config_path", args.config_path,
        "--csv_path", args.csv_path,
        "--output_dir", args.output_dir,
        "--guidance_scale", str(args.guidance_scale),
        "--num_inference_steps", str(args.num_inference_steps)
    ]
    if args.neg_prompt:
        cmd.extend(["--neg_prompt", args.neg_prompt])
        
    subprocess.run(cmd)

if __name__ == "__main__":
    main()