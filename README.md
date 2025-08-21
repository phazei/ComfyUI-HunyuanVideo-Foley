# HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation

<div align="center">
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project&message=Web&color=green"></a> &ensp;
  <a href="https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a> &ensp;
  <a href="https://arxiv.org/abs/2506.17201"><img src="https://img.shields.io/badge/ArXiv-2506.17201-red"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-Foley"><img src="https://img.shields.io/static/v1?label=Model&message=HuggingFace&color=yellow"></a>
</div>


## **Demo**
TODO: pr video

![full video](https://github.com/user-attachments/assets/d8548970-4271-49eb-833d-1d346f5f31e0)

## Abstract
Recent advances in video generation produce visually realistic content, yet the absence of synchronized audio severely compromises immersion. To address key challenges in video-to-audio generation, including multimodal data scarcity, modality imbalance and limited audio quality in existing methods, we propose HunyuanVideo-Foley, an end-to-end text-video-to-audio framework that synthesizes high-fidelity audio precisely aligned with visual dynamics and semantic context. Our approach incorporates three core innovations: (1) a scalable data pipeline curating 100k-hour multimodal datasets through automated annotation; (2) a representation alignment strategy using self-supervised audio features to guide latent diffusion training, efficiently improving audio quality and generation stability; (3) a novel multimodal diffusion transformer resolving modal competition, containing dual-stream audio-video fusion through joint attention, and textual semantic injection via cross-attention. Comprehensive evaluations demonstrate that HunyuanVideo-Foley achieves new state-of-the-art performance across audio fidelity, visual-semantic alignment, temporal alignment and distribution matching

## Data Pipeline Design
![image](assets/data_pipeline.png)
The TV2A task presents a complex multimodal generation challenge that requires large-scale, high-quality text-video-audio datasets to produce robust and generalizable audio. Current open-source datasets, however, lack the necessary quality and scale to adequately support this demanding task. To bridge this gap, we develop a comprehensive data pipeline designed to systematically identify and exclude unsuitable content.

## Architecture
![image](assets/model_arch.png)
To achieve modality balance and high-quality TV2A generation, we introduce the HunyuanVideo-Foley framework. HunyuanVideo-Foley employs a hybrid architecture with N1 multimodal transformer blocks (visual-audio streams) followed unimodal transformer blocks (audio stream only). During training, video frames are encoded by a pre-trained visual encoder into visual features, while text captions are processed through a pre-trained text encoder to extract semantic features. Concurrently, raw audio undergoes an audio encoder to yield latent representations which are perturbed by additive Gaussian noise. The temporal alignment mechanism utilizes Synchformer derived frame-level synchronization features to coordinate generation process through gated modulation pathways.


## Performance

Objective Evaluation Results on Kling-Audio-Eval

| Method | FD_PANNs ↓ | FD_PASST ↓ | KL ↓ | IS ↑ | PQ ↑ | PC ↓ | CE ↑ | CU ↑ | IB ↑ | DeSync ↓ | CLAP ↑ |
|--------|------------|------------|------|------|------|------|------|------|------|----------|---------|
| FoleyGrafter | 22.30 | 322.63 | 2.47 | 7.08 | 6.05 | 2.91 | 3.28 | 5.44 | 0.22 | 1.23 | 0.22 |
| V-AURA | 33.15 | 474.56 | 3.24 | 5.80 | 5.69 | 3.98 | 3.13 | 4.83 | 0.25 | 0.86 | 0.13 |
| Frieren | 16.86 | 293.57 | 2.95 | 7.32 | 5.72 | 2.55 | 2.88 | 5.10 | 0.21 | 0.86 | 0.16 |
| MMAudio | 9.01 | 205.85 | 2.17 | 9.59 | 5.94 | 2.91 | 3.30 | 5.39 | 0.30 | 0.56 | 0.27 |
| ThinkSound | 9.92 | 228.68 | 2.39 | 6.86 | 5.78 | 3.23 | 3.12 | 5.11 | 0.22 | 0.67 | 0.22 |
| **HiFi-Foley (ours)** | **6.07** | **202.12** | **1.89** | **8.30** | **6.12** | **2.76** | **3.22** | **5.53** | **0.38** | **0.54** | **0.24** |


Objective Evaluation Results on VGGSound-Test

| Method | FD_PANNs ↓ | FD_PASST ↓ | KL ↓ | IS ↑ | PQ ↑ | PC ↓ | CE ↑ | CU ↑ | IB ↑ | DeSync ↓ | CLAP ↑ |
|--------|------------|------------|------|------|------|------|------|------|------|----------|---------|
| FoleyGrafter | 20.65 | 171.43 | 2.26 | 14.58 | 6.33 | 2.87 | 3.60 | 5.74 | 0.26 | 1.22 | 0.19 |
| V-AURA | 18.91 | 291.72 | 2.40 | 8.58 | 5.70 | 4.19 | 3.49 | 4.87 | 0.27 | 0.72 | 0.12 |
| Frieren | 11.69 | 83.17 | 2.75 | 12.23 | 5.87 | 2.99 | 3.54 | 5.32 | 0.23 | 0.85 | 0.11 |
| MMAudio | 7.42 | 116.92 | 1.77 | 21.00 | 6.18 | 3.17 | 4.03 | 5.61 | 0.33 | 0.47 | 0.25 |
| ThinkSound | 8.46 | 67.18 | 1.90 | 11.11 | 5.98 | 3.61 | 3.81 | 5.33 | 0.24 | 0.57 | 0.16 |
| **HiFi-Foley (ours)** | **11.34** | **145.22** | **2.14** | **16.14** | **6.40** | **2.78** | **3.99** | **5.79** | **0.36** | **0.53** | **0.24** |


Objective and Subjective Evaluation Results on MovieGen-Audio-Bench

| Method | PQ ↑ | PC ↓ | CE ↑ | CU ↑ | IB ↑ | DeSync ↓ | CLAP ↑ | MOS-Q ↑ | MOS-S ↑ | MOS-T ↑ |
|--------|------|------|------|------|------|----------|---------|----------|----------|----------|
| FoleyGrafter | 6.27 | 2.72 | 3.34 | 5.68 | 0.17 | 1.29 | 0.14 | 3.36±0.78 | 3.54±0.88 | 3.46±0.95 |
| V-AURA | 5.82 | 4.30 | 3.63 | 5.11 | 0.23 | 1.38 | 0.14 | 2.55±0.97 | 2.60±1.20 | 2.70±1.37 |
| Frieren | 5.71 | 2.81 | 3.47 | 5.31 | 0.18 | 1.39 | 0.16 | 2.92±0.95 | 2.76±1.20 | 2.94±1.26 |
| MMAudio | 6.17 | 2.84 | 3.59 | 5.62 | 0.27 | 0.80 | 0.35 | 3.58±0.84 | 3.63±1.00 | 3.47±1.03 |
| ThinkSound | 6.04 | 3.73 | 3.81 | 5.59 | 0.18 | 0.91 | 0.20 | 3.20±0.97 | 3.01±1.04 | 3.02±1.08 |
| **HiFi-Foley (ours)** | **6.59** | **2.74** | **3.88** | **6.13** | **0.35** | **0.74** | **0.33** | **4.14±0.68** | **4.12±0.77** | **4.15±0.75** |



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
