# HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation

<div align="center">
  <a href="https://szczesnys.github.io/hunyuanvideo-foley"><img src="https://img.shields.io/static/v1?label=Project&message=pages&color=green"></a> &ensp;
  <a href="https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a> &ensp;
  <a href="https://arxiv.org/abs/2506.17201"><img src="https://img.shields.io/badge/arXiv-HunyuanVideoFoley-red?logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-Foley"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Huggingface-ffc107?color=ffc107&logoColor=white"></a>
</div>

<div>
    <a href="" target="_blank">Sizhe Shan</a><sup>1,</sup><sup>2</sup><sup>*</sup>,</span>
    <a href="" target="_blank">Qiulin li</a><sup>1,</sup><sup>3</sup><sup>*</sup>, </span>
    <a href="" target="_blank">Yutao Cui</a><sup>1</sup>,</span>
    <a href="" target="_blank">Miles Yang</a><sup>1</sup>,</span>
    <a href="" target="_blank">Zhao Zhong</a><sup>1</sup>,</span>
    <a href="" target="_blank">Yuehai Wang</a><sup>2</sup>,</span>
    <a href="" target="_blank">Qun Yang</a><sup>3</sup>,</span>
    <a href="" target="_blank">Jin Zhou</a><sup>1</sup><sup>†</sup></span>
</div>

<div align="center" style="font-family: charter;">
    <sup>1</sup>Tencent, Hunyuan&emsp;
    </br>
    <sup>2</sup>Zhejiang University&emsp;
    </br>
    <sup>3</sup>Nanjing University of Aeronautics and Astronautics&emsp;
</div>


## **Demo**
TODO: pr video

<div align="center">
  <video src="https://github.com/user-attachments/assets/d8548970-4271-49eb-833d-1d346f5f31e0" width="70%"> </video>
</div>

## Abstract
Tencent Hunyuan officially open-sources the end-to-end video sound effect generation model HunyuanVideo-Foley! A professional-grade sound effect generation tool specifically designed for video content creators, widely applicable to diverse scenarios including short video creation, film production, advertising creativity, and game development.
The model's core highlights include:

* Multi-scenario audio-visual synchronization: Supports generating high-quality audio that is synchronized and semantically aligned with complex video scenes, enhancing realism and immersive experience, suitable for diverse creative needs in film/TV and gaming;
* Multi-modal semantic balance: Balances the analysis of visual and textual information, comprehensively orchestrates sound effect elements, avoids one-sided generation, and meets personalized dubbing requirements;
* High-fidelity audio output: Self-developed 48kHz audio VAE perfectly reconstructs sound effects, music, and vocals, achieving professional-grade audio generation.

On multiple evaluation benchmarks, HunyuanVideo-Foley's performance comprehensively leads the field, achieving new state-of-the-art (SOTA) levels in dimensions including audio fidelity, visual-semantic alignment, temporal alignment, and distribution matching, surpassing all open-source solutions.

![image](assets/pan_chart.png)

## Data Pipeline Design
![image](assets/data_pipeline.png)
The TV2A task presents a complex multimodal generation challenge that requires large-scale, high-quality text-video-audio datasets to produce robust and generalizable audio. Current open-source datasets, however, lack the necessary quality and scale to adequately support this demanding task. To bridge this gap, we develop a comprehensive data pipeline designed to systematically identify and exclude unsuitable content.

## Architecture
![image](assets/model_arch.png)
To achieve modality balance and high-quality TV2A generation, we introduce the HunyuanVideo-Foley framework. HunyuanVideo-Foley employs a hybrid architecture with N1 multimodal transformer blocks (visual-audio streams) followed unimodal transformer blocks (audio stream only). During training, video frames are encoded by a pre-trained visual encoder into visual features, while text captions are processed through a pre-trained text encoder to extract semantic features. Concurrently, raw audio undergoes an audio encoder to yield latent representations which are perturbed by additive Gaussian noise. The temporal alignment mechanism utilizes Synchformer derived frame-level synchronization features to coordinate generation process through gated modulation pathways.


## Performance

Objective and Subjective Evaluation Results on MovieGen-Audio-Bench

| Method | PQ ↑ | PC ↓ | CE ↑ | CU ↑ | IB ↑ | DeSync ↓ | CLAP ↑ | MOS-Q ↑ | MOS-S ↑ | MOS-T ↑ |
|--------|------|------|------|------|------|----------|---------|----------|----------|----------|
| FoleyGrafter | 6.27 | 2.72 | 3.34 | 5.68 | 0.17 | 1.29 | 0.14 | 3.36±0.78 | 3.54±0.88 | 3.46±0.95 |
| V-AURA | 5.82 | 4.30 | 3.63 | 5.11 | 0.23 | 1.38 | 0.14 | 2.55±0.97 | 2.60±1.20 | 2.70±1.37 |
| Frieren | 5.71 | 2.81 | 3.47 | 5.31 | 0.18 | 1.39 | 0.16 | 2.92±0.95 | 2.76±1.20 | 2.94±1.26 |
| MMAudio | 6.17 | 2.84 | 3.59 | 5.62 | 0.27 | 0.80 | 0.35 | 3.58±0.84 | 3.63±1.00 | 3.47±1.03 |
| ThinkSound | 6.04 | 3.73 | 3.81 | 5.59 | 0.18 | 0.91 | 0.20 | 3.20±0.97 | 3.01±1.04 | 3.02±1.08 |
| **HiFi-Foley (ours)** | **6.59** | **2.74** | **3.88** | **6.13** | **0.35** | **0.74** | **0.33** | **4.14±0.68** | **4.12±0.77** | **4.15±0.75** |


Objective Evaluation Results on Kling-Audio-Eval

| Method | FD_PANNs ↓ | FD_PASST ↓ | KL ↓ | IS ↑ | PQ ↑ | PC ↓ | CE ↑ | CU ↑ | IB ↑ | DeSync ↓ | CLAP ↑ |
|--------|------------|------------|------|------|------|------|------|------|------|----------|---------|
| FoleyGrafter | 22.30 | 322.63 | 2.47 | 7.08 | 6.05 | 2.91 | 3.28 | 5.44 | 0.22 | 1.23 | 0.22 |
| V-AURA | 33.15 | 474.56 | 3.24 | 5.80 | 5.69 | 3.98 | 3.13 | 4.83 | 0.25 | 0.86 | 0.13 |
| Frieren | 16.86 | 293.57 | 2.95 | 7.32 | 5.72 | 2.55 | 2.88 | 5.10 | 0.21 | 0.86 | 0.16 |
| MMAudio | 9.01 | 205.85 | 2.17 | 9.59 | 5.94 | 2.91 | 3.30 | 5.39 | 0.30 | 0.56 | 0.27 |
| ThinkSound | 9.92 | 228.68 | 2.39 | 6.86 | 5.78 | 3.23 | 3.12 | 5.11 | 0.22 | 0.67 | 0.22 |
| **HiFi-Foley (ours)** | **6.07** | **202.12** | **1.89** | **8.30** | **6.12** | **2.76** | **3.22** | **5.53** | **0.38** | **0.54** | **0.24** |



## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley
cd HunyuanVideo-Foley
```

### Installation Guide for Linux
We recommend CUDA versions 12.4 or 11.8 for the manual installation.
Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

```shell
pip install -r requirements.txt
```


## 🧱 Download Pretrained Models
<!-- The details of download pretrained models are shown [here](ckpts/README.md). -->


### Using Command Line

For a single video, you can use the following command:

```bash
python3 infer.py \
    --model-path MODEL_PATH_DIR \
    --config_path ./configs/hunyuanvideo-foley-xxl.yaml \
    --single_video video_path \
    --single_promot "audio description" \
    --output_dir OUTPUT_DIR
```

For batch inference, you can use the following command:

```bash
python3 infer.py \
    --model_path  MODEL_PATH_DIR \
    --config_path ./configs/hunyuanvideo-foley-xxl.yaml \
    --csv_path assets/test.csv \
    --output_dir OUTPUT_DIR
```


### Run a Gradio Server

```bash
python3 gradio_app.py
```

## Citation

## Acknowledgements

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [MMAudio](https://github.com/hkchengrex/MMAudio), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
