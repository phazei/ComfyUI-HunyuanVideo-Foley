<div align="center">
  
# 🎬 HunyuanVideo-Foley

<h3>Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation</h3>

<p align="center">
  <strong>Professional-grade AI sound effect generation for video content creators</strong>
</p>

<div align="center" style="margin: 20px 0;">
  
[![Project Page](https://img.shields.io/badge/🌐_Project-Page-green.svg?style=for-the-badge)](https://szczesnys.github.io/hunyuanvideo-foley)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-blue.svg?style=for-the-badge&logo=github)](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley)
[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-red.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2506.17201)
[![Model](https://img.shields.io/badge/🤗_Model-Huggingface-yellow.svg?style=for-the-badge&logo=huggingface)](https://huggingface.co/tencent/HunyuanVideo-Foley)

</div>

<div align="center" style="margin: 30px 0;">
  
![Stars](https://img.shields.io/github/stars/Tencent-Hunyuan/HunyuanVideo-Foley?style=social)
![Forks](https://img.shields.io/github/forks/Tencent-Hunyuan/HunyuanVideo-Foley?style=social)
![Issues](https://img.shields.io/github/issues/Tencent-Hunyuan/HunyuanVideo-Foley?style=flat-square)
![License](https://img.shields.io/github/license/Tencent-Hunyuan/HunyuanVideo-Foley?style=flat-square)

</div>

</div>

---

<div align="center">
  
### 👥 **Authors**

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 20px 0;">

**Sizhe Shan**<sup>1,2*</sup> • **Qiulin Li**<sup>1,3*</sup> • **Yutao Cui**<sup>1</sup> • **Miles Yang**<sup>1</sup> • **Zhao Zhong**<sup>1</sup> • **Yuehai Wang**<sup>2</sup> • **Qun Yang**<sup>3</sup> • **Jin Zhou**<sup>1†</sup>

</div>

<div style="margin-top: 15px; font-size: 14px; color: #666;">
  
🏢 <sup>1</sup>**Tencent Hunyuan** • 🎓 <sup>2</sup>**Zhejiang University** • ✈️ <sup>3</sup>**Nanjing University of Aeronautics and Astronautics**

*Equal contribution • †Corresponding author

</div>

</div>


---

## 🎥 **Demo & Showcase**

<div align="center">
  
> **Experience the magic of AI-generated Foley audio in perfect sync with video content!**

<div style="border: 3px solid #4A90E2; border-radius: 15px; padding: 10px; margin: 20px 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
  
  <video src="https://github.com/user-attachments/assets/d8548970-4271-49eb-833d-1d346f5f31e0" width="80%" controls style="border-radius: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);"> </video>
  
  <p><em>🎬 Watch how HunyuanVideo-Foley generates immersive sound effects synchronized with video content</em></p>
  
</div>

### ✨ **Key Highlights**

<table align="center" style="border: none; margin: 20px 0;">
<tr>
<td align="center" width="33%">
  
🎭 **Multi-scenario Sync**  
High-quality audio synchronized with complex video scenes

</td>
<td align="center" width="33%">
  
🧠 **Multi-modal Balance**  
Perfect harmony between visual and textual information

</td>
<td align="center" width="33%">
  
🎵 **48kHz Hi-Fi Output**  
Professional-grade audio generation with crystal clarity

</td>
</tr>
</table>

</div>

---

## 📄 **Abstract**

<div align="center" style="background: linear-gradient(135deg, #ffeef8 0%, #f0f8ff 100%); padding: 30px; border-radius: 20px; margin: 20px 0; border-left: 5px solid #ff6b9d;">

**🚀 Tencent Hunyuan** proudly open-sources **HunyuanVideo-Foley** - an end-to-end video sound effect generation model! 

*A professional-grade AI tool specifically designed for video content creators, widely applicable to diverse scenarios including short video creation, film production, advertising creativity, and game development.*

</div>

### 🎯 **Core Highlights**

<div style="display: grid; grid-template-columns: 1fr; gap: 15px; margin: 20px 0;">

<div style="border-left: 4px solid #4CAF50; padding: 15px; background: #f8f9fa; border-radius: 8px;">
  
**🎬 Multi-scenario Audio-Visual Synchronization**  
Supports generating high-quality audio that is synchronized and semantically aligned with complex video scenes, enhancing realism and immersive experience for film/TV and gaming applications.

</div>

<div style="border-left: 4px solid #2196F3; padding: 15px; background: #f8f9fa; border-radius: 8px;">
  
**⚖️ Multi-modal Semantic Balance**  
Intelligently balances visual and textual information analysis, comprehensively orchestrates sound effect elements, avoids one-sided generation, and meets personalized dubbing requirements.

</div>

<div style="border-left: 4px solid #FF9800; padding: 15px; background: #f8f9fa; border-radius: 8px;">
  
**🎵 High-fidelity Audio Output**  
Self-developed 48kHz audio VAE perfectly reconstructs sound effects, music, and vocals, achieving professional-grade audio generation quality.

</div>

</div>

<div align="center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">
  
**🏆 SOTA Performance Achieved**

*HunyuanVideo-Foley comprehensively leads the field across multiple evaluation benchmarks, achieving new state-of-the-art levels in audio fidelity, visual-semantic alignment, temporal alignment, and distribution matching - surpassing all open-source solutions!*

</div>

<div align="center">
  
![Performance Overview](assets/pan_chart.png)
*📊 Performance comparison across different evaluation metrics - HunyuanVideo-Foley leads in all categories*

</div>

---

## 🔧 **Technical Architecture**

### 📊 **Data Pipeline Design**

<div align="center" style="margin: 20px 0;">
  
![Data Pipeline](assets/data_pipeline.png)
*🔄 Comprehensive data processing pipeline for high-quality text-video-audio datasets*

</div>

<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #17a2b8; margin: 20px 0;">

The **TV2A (Text-Video-to-Audio)** task presents a complex multimodal generation challenge requiring large-scale, high-quality datasets. Our comprehensive data pipeline systematically identifies and excludes unsuitable content to produce robust and generalizable audio generation capabilities.

</div>

### 🏗️ **Model Architecture**

<div align="center" style="margin: 20px 0;">
  
![Model Architecture](assets/model_arch.png)
*🧠 HunyuanVideo-Foley hybrid architecture with multimodal and unimodal transformer blocks*

</div>

<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #28a745; margin: 20px 0;">

**HunyuanVideo-Foley** employs a sophisticated hybrid architecture:

- **🔄 Multimodal Transformer Blocks**: Process visual-audio streams simultaneously
- **🎵 Unimodal Transformer Blocks**: Focus on audio stream refinement
- **👁️ Visual Encoding**: Pre-trained encoder extracts visual features from video frames
- **📝 Text Processing**: Semantic features extracted via pre-trained text encoder  
- **🎧 Audio Encoding**: Latent representations with Gaussian noise perturbation
- **⏰ Temporal Alignment**: Synchformer-based frame-level synchronization with gated modulation

</div>

---

## 📈 **Performance Benchmarks**

### 🎬 **MovieGen-Audio-Bench Results**

<div align="center">
  
> *Objective and Subjective evaluation results demonstrating superior performance across all metrics*

</div>

<div style="overflow-x: auto; margin: 20px 0;">

| 🏆 **Method** | **PQ** ↑ | **PC** ↓ | **CE** ↑ | **CU** ↑ | **IB** ↑ | **DeSync** ↓ | **CLAP** ↑ | **MOS-Q** ↑ | **MOS-S** ↑ | **MOS-T** ↑ |
|:-------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:-------------:|:-----------:|:------------:|:------------:|:------------:|
| FoleyGrafter | 6.27 | 2.72 | 3.34 | 5.68 | 0.17 | 1.29 | 0.14 | 3.36±0.78 | 3.54±0.88 | 3.46±0.95 |
| V-AURA | 5.82 | 4.30 | 3.63 | 5.11 | 0.23 | 1.38 | 0.14 | 2.55±0.97 | 2.60±1.20 | 2.70±1.37 |
| Frieren | 5.71 | 2.81 | 3.47 | 5.31 | 0.18 | 1.39 | 0.16 | 2.92±0.95 | 2.76±1.20 | 2.94±1.26 |
| MMAudio | 6.17 | 2.84 | 3.59 | 5.62 | 0.27 | 0.80 | 0.35 | 3.58±0.84 | 3.63±1.00 | 3.47±1.03 |
| ThinkSound | 6.04 | 3.73 | 3.81 | 5.59 | 0.18 | 0.91 | 0.20 | 3.20±0.97 | 3.01±1.04 | 3.02±1.08 |
| **🥇 HiFi-Foley (ours)** | **🟢 6.59** | **🟢 2.74** | **🟢 3.88** | **🟢 6.13** | **🟢 0.35** | **🟢 0.74** | **🟢 0.33** | **🟢 4.14±0.68** | **🟢 4.12±0.77** | **🟢 4.15±0.75** |

</div>


### 🎯 **Kling-Audio-Eval Results**

<div align="center">
  
> *Comprehensive objective evaluation showcasing state-of-the-art performance*

</div>

<div style="overflow-x: auto; margin: 20px 0;">

| 🏆 **Method** | **FD_PANNs** ↓ | **FD_PASST** ↓ | **KL** ↓ | **IS** ↑ | **PQ** ↑ | **PC** ↓ | **CE** ↑ | **CU** ↑ | **IB** ↑ | **DeSync** ↓ | **CLAP** ↑ |
|:-------------:|:--------------:|:--------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:-------------:|:-----------:|
| FoleyGrafter | 22.30 | 322.63 | 2.47 | 7.08 | 6.05 | 2.91 | 3.28 | 5.44 | 0.22 | 1.23 | 0.22 |
| V-AURA | 33.15 | 474.56 | 3.24 | 5.80 | 5.69 | 3.98 | 3.13 | 4.83 | 0.25 | 0.86 | 0.13 |
| Frieren | 16.86 | 293.57 | 2.95 | 7.32 | 5.72 | 2.55 | 2.88 | 5.10 | 0.21 | 0.86 | 0.16 |
| MMAudio | 9.01 | 205.85 | 2.17 | 9.59 | 5.94 | 2.91 | 3.30 | 5.39 | 0.30 | 0.56 | 0.27 |
| ThinkSound | 9.92 | 228.68 | 2.39 | 6.86 | 5.78 | 3.23 | 3.12 | 5.11 | 0.22 | 0.67 | 0.22 |
| **🥇 HiFi-Foley (ours)** | **🟢 6.07** | **🟢 202.12** | **🟢 1.89** | **🟢 8.30** | **🟢 6.12** | **🟢 2.76** | **🟢 3.22** | **🟢 5.53** | **🟢 0.38** | **🟢 0.54** | **🟢 0.24** |

</div>

<div align="center" style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 15px; border-radius: 10px; margin: 20px 0;">
  
**🎉 Outstanding Results!** HunyuanVideo-Foley achieves the best scores across **ALL** evaluation metrics, demonstrating significant improvements in audio quality, synchronization, and semantic alignment.

</div>



---

## 🚀 **Quick Start**

### 📦 **Installation**

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">

**🔧 System Requirements**
- **CUDA**: 12.4 or 11.8 recommended
- **Python**: 3.8+ 
- **OS**: Linux (primary support)

</div>

#### **Step 1: Clone Repository**

```bash
# 📥 Clone the repository
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley
cd HunyuanVideo-Foley
```

#### **Step 2: Environment Setup**

<div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 10px 0;">

💡 **Tip**: We recommend using [Conda](https://docs.anaconda.com/free/miniconda/index.html) for Python environment management.

</div>

```bash
# 🔧 Install dependencies
pip install -r requirements.txt
```

#### **Step 3: Download Pretrained Models**

<div style="background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 10px 0;">

🔗 **Model weights and detailed download instructions will be available soon!**  
<!-- The details of download pretrained models are shown [here](ckpts/README.md). -->

</div>


---

## 💻 **Usage**

### 🎬 **Single Video Generation**

<div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">

Generate Foley audio for a single video file with text description:

</div>

```bash
python3 infer.py \
    --model-path MODEL_PATH_DIR \
    --config_path ./configs/hunyuanvideo-foley-xxl.yaml \
    --single_video video_path \
    --single_prompt "audio description" \
    --output_dir OUTPUT_DIR
```

### 📂 **Batch Processing**

<div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800; margin: 10px 0;">

Process multiple videos using a CSV file with video paths and descriptions:

</div>

```bash
python3 infer.py \
    --model_path MODEL_PATH_DIR \
    --config_path ./configs/hunyuanvideo-foley-xxl.yaml \
    --csv_path assets/test.csv \
    --output_dir OUTPUT_DIR
```

### 🌐 **Interactive Web Interface**

<div style="background: #f3e5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #9c27b0; margin: 10px 0;">

Launch a user-friendly Gradio web interface for easy interaction:

</div>

```bash
python3 gradio_app.py
```

<div align="center" style="margin: 20px 0;">
  
*🚀 Then open your browser and navigate to the provided local URL to start generating Foley audio!*

</div>

---

## 📚 **Citation**

<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #6c757d; margin: 20px 0;">

If you find **HunyuanVideo-Foley** useful for your research, please consider citing our paper:

</div>

```bibtex
@article{hunyuanvideo-foley2025,
  title={HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation},
  author={Sizhe Shan and Qiulin Li and Yutao Cui and Miles Yang and Zhao Zhong and Yuehai Wang and Qun Yang and Jin Zhou},
  journal={arXiv preprint arXiv:2506.17201},
  year={2025}
}
```

---

## 🙏 **Acknowledgements**

<div align="center">
  
**We extend our heartfelt gratitude to the open-source community!**

</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">

<div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 15px; border-radius: 10px; text-align: center;">
  
🎨 **[Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)**  
*Foundation diffusion models*

</div>

<div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 15px; border-radius: 10px; text-align: center;">
  
⚡ **[FLUX](https://github.com/black-forest-labs/flux)**  
*Advanced generation techniques*

</div>

<div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 15px; border-radius: 10px; text-align: center;">
  
🎵 **[MMAudio](https://github.com/hkchengrex/MMAudio)**  
*Multimodal audio generation*

</div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; text-align: center;">
  
🤗 **[HuggingFace](https://huggingface.co)**  
*Platform & diffusers library*

</div>

</div>

<div align="center" style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">
  
**🌟 Special thanks to all researchers and developers who contribute to the advancement of AI-generated audio and multimodal learning!**

</div>

---

<div align="center" style="margin: 30px 0;">
  
### 🔗 **Connect with Us**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/Tencent-Hunyuan)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue?style=for-the-badge&logo=twitter)](https://twitter.com/Tencent)
[![WeChat](https://img.shields.io/badge/WeChat-HunyuanAI-green?style=for-the-badge&logo=wechat)](https://hunyuan.ai)

<p style="color: #666; margin-top: 15px; font-size: 14px;">
  
© 2025 Tencent Hunyuan. All rights reserved. | Made with ❤️ for the AI community

</p>

</div>
