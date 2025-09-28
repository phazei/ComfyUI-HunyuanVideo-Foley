import os
import tempfile
import gradio as gr
import torch
import torchaudio
from loguru import logger
from typing import Optional, Tuple
import random
import numpy as np

from hunyuanvideo_foley.utils.model_utils import load_model
from hunyuanvideo_foley.utils.feature_utils import feature_process
from hunyuanvideo_foley.utils.model_utils import denoise_process
from hunyuanvideo_foley.utils.media_utils import merge_audio_video

# Global variables for model storage
model_dict = None
cfg = None
device = None

# need to modify the model path
MODEL_PATH = os.environ.get("HIFI_FOLEY_MODEL_PATH", "./pretrained_models/")
ENABLE_OFFLOAD = os.environ.get("ENABLE_OFFLOAD", "false").lower() in ("true", "1", "yes")
MODEL_SIZE = os.environ.get("MODEL_SIZE", "xxl")  # default to xxl model
CONFIG_PATH = os.environ.get("CONFIG_PATH", "")

def setup_device(device_str: str = "auto", gpu_id: int = 0) -> torch.device:
    """Setup computing device"""
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

def auto_load_models() -> str:
    """Automatically load preset models"""
    global model_dict, cfg, device
    
    try:
        if not os.path.exists(MODEL_PATH):
            return f"❌ Model directory not found: {MODEL_PATH}"
        
        # Use GPU by default
        device = setup_device("auto", 0)
        
        # Auto-select config if not specified
        config_path = CONFIG_PATH
        if not config_path:
            config_mapping = {
                "xl": "configs/hunyuanvideo-foley-xl.yaml",
                "xxl": "configs/hunyuanvideo-foley-xxl.yaml"
            }
            config_path = config_mapping.get(MODEL_SIZE, "configs/hunyuanvideo-foley-xxl.yaml")
        
        # Load model
        logger.info("Auto-loading model...")
        logger.info(f"Model path: {MODEL_PATH}")
        logger.info(f"Model size: {MODEL_SIZE}")
        logger.info(f"Config path: {config_path}")
        logger.info(f"Offload mode: {'enabled' if ENABLE_OFFLOAD else 'disabled'}")
        
        model_dict, cfg = load_model(MODEL_PATH, config_path, device, enable_offload=ENABLE_OFFLOAD, model_size=MODEL_SIZE)
        
        logger.info("✅ Model loaded successfully!")
        return "✅ Model loaded successfully!"
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return f"❌ Model loading failed: {str(e)}"

def infer_single_video(
    video_file, 
    text_prompt: str, 
    neg_prompt: str = None,
    guidance_scale: float = 4.5, 
    num_inference_steps: int = 50,
    sample_nums: int = 1
) -> Tuple[list, str]:
    """Single video inference"""
    global model_dict, cfg, device
    
    if model_dict is None or cfg is None:
        return [], "❌ Please load the model first!"
    
    if video_file is None:
        return [], "❌ Please upload a video file!"
    
    # Allow empty text prompt, use empty string if no prompt provided
    if text_prompt is None:
        text_prompt = ""
    text_prompt = text_prompt.strip()
    
    try:
        logger.info(f"Processing video: {video_file}")
        logger.info(f"Text prompt: {text_prompt}")
        
        # Feature processing
        visual_feats, text_feats, audio_len_in_s = feature_process(
            video_file,
            text_prompt,
            model_dict,
            cfg,
            neg_prompt=neg_prompt
        )
        
        # Denoising process to generate multiple audio samples
        # Note: The model now generates sample_nums audio samples per inference
        # The denoise_process function returns audio with shape [batch_size, channels, samples]
        logger.info(f"Generating {sample_nums} audio samples...")
        audio, sample_rate = denoise_process(
            visual_feats,
            text_feats,
            audio_len_in_s,
            model_dict,
            cfg,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            batch_size=sample_nums
        )
        
        # Create temporary files to save results
        temp_dir = tempfile.mkdtemp()
        video_outputs = []
        
        # Process each generated audio sample
        for i in range(sample_nums):
            # Save audio file
            audio_output = os.path.join(temp_dir, f"generated_audio_{i+1}.wav")
            torchaudio.save(audio_output, audio[i], sample_rate)
            
            # Merge video and audio
            video_output = os.path.join(temp_dir, f"video_with_audio_{i+1}.mp4")
            merge_audio_video(audio_output, video_file, video_output)
            video_outputs.append(video_output)
        
        logger.info(f"Inference completed! Generated {sample_nums} samples.")
        return video_outputs, f"✅ Generated {sample_nums} audio sample(s) successfully!"
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return [], f"❌ Inference failed: {str(e)}"

def update_video_outputs(video_list, status_msg):
    """Update video outputs based on the number of generated samples"""
    # Initialize all outputs as None
    outputs = [None] * 6
    
    # Set values based on generated videos
    for i, video_path in enumerate(video_list[:6]):  # Max 6 samples
        outputs[i] = video_path
    
    # Return all outputs plus status message
    return tuple(outputs + [status_msg])

def create_gradio_interface():
    """Create Gradio interface"""
    
    # Custom CSS for beautiful interface with better contrast
    css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    .status-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e1e5e9;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .status-card label {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    .usage-guide h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .usage-guide p {
        color: #4a5568 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        margin: 0.5rem 0 !important;
    }
    
    .usage-guide strong {
        color: #1a202c !important;
        font-weight: 700 !important;
    }
    
    .usage-guide em {
        color: #1a202c !important;
        font-weight: 700 !important;
        font-style: normal !important;
    }
    
    .main-interface {
        margin-bottom: 2rem;
    }
    
    .input-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-right: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    .input-section h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    .input-section label {
        color: #4a5568 !important;
        font-weight: 500 !important;
    }
    
    .output-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-left: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    .output-section h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    .output-section label {
        color: #4a5568 !important;
        font-weight: 500 !important;
    }
    
    .examples-section h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    
    .generate-btn {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 12px 30px !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    

    
    .examples-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    .examples-section p {
        color: #4a5568 !important;
        margin-bottom: 1rem !important;
    }
    
    .example-row {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        align-items: center;
    }
    
    .example-row:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    
    .example-row .markdown {
        color: #2d3748 !important;
    }
    
    .example-row .markdown p {
        color: #2d3748 !important;
        margin: 0.5rem 0 !important;
        line-height: 1.5 !important;
    }
    
    .example-row .markdown strong {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Example grid layout styles */
    .example-grid-row {
        margin: 1rem 0;
        gap: 1rem;
    }
    
    .example-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1rem;
        transition: all 0.3s ease;
        margin: 0.25rem;
        max-width: 250px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .example-item:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    
    .example-caption {
        margin: 0.5rem 0 !important;
        min-height: 2.8rem !important;
        display: flex !important;
        align-items: flex-start !important;
    }
    
    .example-caption p {
        color: #2d3748 !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Multi-video gallery styles */
    .additional-samples {
        margin-top: 1rem;
        gap: 0.5rem;
    }
    
    .additional-samples .gradio-video {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Video gallery responsive layout */
    .video-gallery {
        display: grid;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .video-gallery.single {
        grid-template-columns: 1fr;
    }
    
    .video-gallery.dual {
        grid-template-columns: 1fr 1fr;
    }
    
    .video-gallery.multi {
        grid-template-columns: repeat(2, 1fr);
        grid-template-rows: auto auto auto;
    }
    
    .footer-text {
        color: #718096 !important;
        text-align: center;
        padding: 2rem;
        font-size: 0.9rem;
    }
    
    /* Video component styling for consistent size */
    .input-section video,
    .output-section video,
    .example-row video {
        width: 100% !important;
        height: 300px !important;
        object-fit: contain !important;
        border-radius: 10px !important;
        background-color: #000 !important;
    }
    
    .example-row video {
        height: 150px !important;
    }
    
    /* Fix for additional samples video display */
    .additional-samples video {
        height: 150px !important;
        object-fit: contain !important;
        border-radius: 10px !important;
        background-color: #000 !important;
    }
    
    .additional-samples .gradio-video {
        border-radius: 10px !important;
        overflow: hidden !important;
        background-color: #000 !important;
    }
    
    .additional-samples .gradio-video > div {
        background-color: #000 !important;
        border-radius: 10px !important;
    }
    
    /* Video container styling */
    .input-section .video-container,
    .output-section .video-container,
    .example-row .video-container {
        background-color: #000 !important;
        border-radius: 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
    }
    
    /* Ensure proper alignment */
    .example-row {
        display: flex !important;
        align-items: stretch !important;
    }
    
    .example-row > div {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    
    /* Video wrapper for better control */
    .video-wrapper {
        position: relative !important;
        width: 100% !important;
        background: #000 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    """
    
    with gr.Blocks(css=css, title="HunyuanVideo-Foley") as app:
        
        # Main header
        with gr.Column(elem_classes=["main-header"]):
            gr.HTML("""
            <h1>🎵 HunyuanVideo-Foley</h1>
            <p>Text-Video-to-Audio Synthesis: Generate realistic audio from video and text descriptions</p>
            """)
        
        # Usage Guide
        with gr.Column(elem_classes=["status-card"]):
            gr.Markdown("""
            ### 📋 Quick Start Guide
            **1.** Upload your video file\t**2.** Add optional text description\t**3.** Adjust sample numbers (1-6)\t**4.** Click Generate Audio
            
            💡 For quick start, you can load the prepared examples by clicking the button.
            """, elem_classes=["usage-guide"])
        
        # Main inference interface - Input and Results side by side
        with gr.Row(elem_classes=["main-interface"]):
            # Input section
            with gr.Column(scale=1, elem_classes=["input-section"]):
                gr.Markdown("### 📹 Video Input")
                
                video_input = gr.Video(
                    label="Upload Video",
                    info="Supported formats: MP4, AVI, MOV, etc.",
                    height=300
                )
                
                text_input = gr.Textbox(
                    label="🎯 Audio Description (English)",
                    placeholder="A person walks on frozen ice",
                    lines=3,
                    info="Describe the audio you want to generate (optional)"
                )
                
                neg_prompt_input = gr.Textbox(
                    label="🚫 Negative Prompt",
                    placeholder="noisy, harsh",
                    lines=2,
                    info="Describe what you want to avoid in the generated audio (optional, default: 'noisy, harsh')"
                )
                
                with gr.Row():
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=4.5,
                        step=0.1,
                        label="🎚️ CFG Scale",
                    )
                    
                    inference_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="⚡ Steps",
                    )
                    
                    sample_nums = gr.Slider(
                        minimum=1,
                        maximum=6,
                        value=1,
                        step=1,
                        label="🎲 Sample Nums",
                    )
                
                generate_btn = gr.Button(
                    "🎵 Generate Audio", 
                    variant="primary",
                    elem_classes=["generate-btn"]
                )
            
            # Results section
            with gr.Column(scale=1, elem_classes=["output-section"]):
                gr.Markdown("### 🎥 Generated Results")
                
                # Multi-video gallery for displaying multiple generated samples
                with gr.Column():
                    # Primary video (Sample 1)
                    video_output_1 = gr.Video(
                        label="Sample 1",
                        height=250,
                        visible=True
                    )
                    
                    # Additional videos (Samples 2-6) - initially hidden
                    with gr.Row(elem_classes=["additional-samples"]):
                        with gr.Column(scale=1):
                            video_output_2 = gr.Video(
                                label="Sample 2",
                                height=150,
                                visible=False
                            )
                            video_output_3 = gr.Video(
                                label="Sample 3", 
                                height=150,
                                visible=False
                            )
                        with gr.Column(scale=1):
                            video_output_4 = gr.Video(
                                label="Sample 4",
                                height=150,
                                visible=False
                            )
                            video_output_5 = gr.Video(
                                label="Sample 5",
                                height=150,
                                visible=False
                            )
                    
                    # Sample 6 - full width
                    video_output_6 = gr.Video(
                        label="Sample 6",
                        height=150,
                        visible=False
                    )
                
                result_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
        
        # Examples section at the bottom
        with gr.Column(elem_classes=["examples-section"]):
            gr.Markdown("### 🌟 Examples")
            gr.Markdown("Click on any example to load it into the interface above")
            
            # Define your custom examples here - 8 examples total
            examples_data = [
                # Example 1
                {
                    "caption": "A person walks on frozen ice",
                    "video_path": "examples/1_video.mp4",
                    "result_path": "examples/1_result.mp4"
                },
                # Example 2
                {
                    "caption": "With a faint sound as their hands parted, the two embraced, a soft 'mm' escaping between them.",
                    "video_path": "examples/2_video.mp4",
                    "result_path": "examples/2_result.mp4"
                },
                # Example 3
                {
                    "caption": "The sound of the number 3's bouncing footsteps is as light and clear as glass marbles hitting the ground. Each step carries a magical sound.", 
                    "video_path": "examples/3_video.mp4",
                    "result_path": "examples/3_result.mp4"
                },
                # Example 4
                {
                    "caption": "gentle gurgling of the stream's current, and music plays in the background which is a beautiful and serene piano solo with a hint of classical charm, evoking a sense of peace and serenity in people's hearts.",
                    "video_path": "examples/4_video.mp4",
                    "result_path": "examples/4_result.mp4"
                },
                # Example 5 - Add your new examples here
                {
                    "caption": "snow crunching under the snowboard's edge.",
                    "video_path": "examples/5_video.mp4",
                    "result_path": "examples/5_result.mp4"
                },
                # Example 6
                {
                    "caption": "The crackling of the fire, the whooshing of the flames, and the occasional crisp popping of charred leaves filled the forest.",
                    "video_path": "examples/6_video.mp4",
                    "result_path": "examples/6_result.mp4"
                },
                # Example 7
                {
                    "caption": "humming of the scooter engine accelerates slowly.",
                    "video_path": "examples/7_video.mp4",
                    "result_path": "examples/7_result.mp4"
                },
                # Example 8
                {
                    "caption": "splash of water and loud thud as person hits the surface.",
                    "video_path": "examples/8_video.mp4",
                    "result_path": "examples/8_result.mp4"
                }
            ]
            
            # Create example grid - 4 examples per row, 2 rows total
            example_buttons = []
            for row in range(2):  # 2 rows
                with gr.Row(elem_classes=["example-grid-row"]):
                    for col in range(4):  # 4 columns
                        idx = row * 4 + col
                        if idx < len(examples_data):
                            example = examples_data[idx]
                            
                            with gr.Column(scale=1, elem_classes=["example-item"]):
                                # Video thumbnail
                                if os.path.exists(example['video_path']):
                                    example_video = gr.Video(
                                        value=example['video_path'],
                                        label=f"Example {idx+1}",
                                        interactive=False,
                                        show_label=True,
                                        height=180
                                    )
                                else:
                                    example_video = gr.HTML(f"""
                                    <div style="background: #f0f0f0; padding: 15px; text-align: center; border-radius: 8px; height: 180px; display: flex; align-items: center; justify-content: center;">
                                        <div>
                                            <p style="color: #666; margin: 0; font-size: 12px;">📹 Video not found</p>
                                            <small style="color: #999; font-size: 10px;">{example['video_path']}</small>
                                        </div>
                                    </div>
                                    """)
                                
                                # Caption (truncated for grid layout)
                                caption_preview = example['caption'][:60] + "..." if len(example['caption']) > 60 else example['caption']
                                gr.Markdown(f"{caption_preview}", elem_classes=["example-caption"])
                                
                                # Load button
                                example_btn = gr.Button(
                                    f"Load Example {idx+1}",
                                    variant="secondary",
                                    size="sm"
                                )
                                example_buttons.append((example_btn, example))
        
        # Event handlers
        def process_inference(video_file, text_prompt, neg_prompt, guidance_scale, inference_steps, sample_nums):
            # Generate videos
            video_list, status_msg = infer_single_video(
                video_file, text_prompt, neg_prompt, guidance_scale, inference_steps, int(sample_nums)
            )
            # Update outputs with proper visibility
            return update_video_outputs(video_list, status_msg)
        
        # Add dynamic visibility control based on sample_nums
        def update_visibility(sample_nums):
            sample_nums = int(sample_nums)
            return [
                gr.update(visible=True),  # Sample 1 always visible
                gr.update(visible=sample_nums >= 2),  # Sample 2
                gr.update(visible=sample_nums >= 3),  # Sample 3
                gr.update(visible=sample_nums >= 4),  # Sample 4
                gr.update(visible=sample_nums >= 5),  # Sample 5
                gr.update(visible=sample_nums >= 6),  # Sample 6
            ]
        
        # Update visibility when sample_nums changes
        sample_nums.change(
            fn=update_visibility,
            inputs=[sample_nums],
            outputs=[video_output_1, video_output_2, video_output_3, video_output_4, video_output_5, video_output_6]
        )
        
        generate_btn.click(
            fn=process_inference,
            inputs=[video_input, text_input, neg_prompt_input, guidance_scale, inference_steps, sample_nums],
            outputs=[
                video_output_1,  # Sample 1 value
                video_output_2,  # Sample 2 value  
                video_output_3,  # Sample 3 value
                video_output_4,  # Sample 4 value
                video_output_5,  # Sample 5 value
                video_output_6,  # Sample 6 value
                result_text
            ]
        )
        
        # Add click handlers for example buttons
        for btn, example in example_buttons:
            def create_example_handler(ex):
                def handler():
                    # Check if files exist, if not, return placeholder message
                    if os.path.exists(ex['video_path']):
                        video_file = ex['video_path']
                    else:
                        video_file = None
                        
                    if os.path.exists(ex['result_path']):
                        result_video = ex['result_path']
                    else:
                        result_video = None
                    
                    status_msg = f"✅ Loaded example with caption: {ex['caption'][:50]}..."
                    if not video_file:
                        status_msg += f"\n⚠️ Video file not found: {ex['video_path']}"
                    if not result_video:
                        status_msg += f"\n⚠️ Result video not found: {ex['result_path']}"
                        
                    return video_file, ex['caption'], "noisy, harsh", result_video, status_msg
                return handler
            
            btn.click(
                fn=create_example_handler(example),
                outputs=[video_input, text_input, neg_prompt_input, video_output_1, result_text]
            )
        
        # Footer
        gr.HTML("""
        <div class="footer-text">
            <p>🚀 Powered by HunyuanVideo-Foley | Generate high-quality audio from video and text descriptions</p>
        </div>
        """)
    
    return app

def set_manual_seed(global_seed):
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

if __name__ == "__main__":
    set_manual_seed(1)
    # Setup logging
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")
    
    # Auto-load model
    logger.info("Starting application and loading model...")
    model_load_result = auto_load_models()
    logger.info(model_load_result)
    
    # Create and launch Gradio app
    app = create_gradio_interface()
    
    # Log completion status
    if "successfully" in model_load_result:
        logger.info("Application ready, model loaded")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=False,
        show_error=True
    )
