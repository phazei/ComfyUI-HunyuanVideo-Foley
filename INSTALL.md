# 安装指南 - HunyuanVideo-Foley

本文档提供了将 HunyuanVideo-Foley 作为 Python 包安装和使用的详细指南。

## 安装方式

### 方式1：从源码安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley
cd HunyuanVideo-Foley

# 安装包（开发模式）
pip install -e .

# 或安装包含所有可选依赖
pip install -e .[all]
```

### 方式2：直接从GitHub安装

```bash
pip install git+https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley.git
```

### 方式3：构建wheel包安装

```bash
# 在项目根目录下
python setup.py bdist_wheel
pip install dist/hunyuanvideo_foley-1.0.0-py3-none-any.whl
```

## 特殊依赖安装

由于某些依赖不在PyPI上，需要单独安装：

```bash
# 安装audiotools（必需）
pip install git+https://github.com/descriptinc/audiotools

# 安装特定版本的transformers（支持SigLIP2）
pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2
```

## 可选依赖安装

```bash
# 安装开发依赖
pip install hunyuanvideo-foley[dev]

# 安装测试依赖  
pip install hunyuanvideo-foley[test]

# 安装Gradio界面依赖
pip install hunyuanvideo-foley[gradio]

# 安装所有可选依赖
pip install hunyuanvideo-foley[all]
```

## 验证安装

```bash
# 检查包是否正确安装
python -c "import hunyuanvideo_foley; print(hunyuanvideo_foley.__version__)"

# 检查命令行工具
hunyuanvideo-foley --help
```

## 使用方法

### 1. 作为Python包使用

```python
import hunyuanvideo_foley as hvf

# 加载模型
model_dict, cfg = hvf.load_model(
    model_path="path/to/model",
    config_path="configs/hunyuanvideo-foley-xxl.yaml"
)

# 处理特征
visual_feats, text_feats, audio_len = hvf.feature_process(
    video_path="video.mp4",
    prompt="footsteps on gravel",
    model_dict=model_dict,
    cfg=cfg
)

# 生成音频
audio, sample_rate = hvf.denoise_process(
    visual_feats, text_feats, audio_len,
    model_dict, cfg
)
```

### 2. 使用命令行工具

```bash
# 单个视频处理
hunyuanvideo-foley \
    --model_path ./pretrained_models \
    --single_video video.mp4 \
    --single_prompt "footsteps on gravel" \
    --output_dir ./outputs

# 批量处理
hunyuanvideo-foley \
    --model_path ./pretrained_models \
    --csv_path batch_videos.csv \
    --output_dir ./outputs

# 启动Gradio界面
hunyuanvideo-foley --gradio --model_path ./pretrained_models
```

### 3. 使用原始脚本（向后兼容）

```bash
# 使用原始infer.py脚本
python infer.py --model_path ./pretrained_models --single_video video.mp4 --single_prompt "audio description"

# 启动Gradio应用
export HIFI_FOLEY_MODEL_PATH=./pretrained_models
python gradio_app.py
```

## 开发环境设置

如果你想参与开发：

```bash
# 克隆项目
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley
cd HunyuanVideo-Foley

# 安装开发版本
pip install -e .[dev]

# 安装pre-commit钩子
pre-commit install

# 运行测试
python -m pytest

# 代码格式化
black --line-length 120 .
isort --profile black .

# 类型检查
mypy --ignore-missing-imports .
```

## 系统要求

- **Python**: 3.8+
- **操作系统**: Linux（主要支持），macOS，Windows
- **GPU内存**: 推荐 ≥24GB VRAM（如RTX 3090/4090）
- **CUDA版本**: 12.4 或 11.8（推荐）

## 故障排除

### 常见问题

1. **ImportError: No module named 'audiotools'**
   ```bash
   pip install git+https://github.com/descriptinc/audiotools
   ```

2. **CUDA内存不足**
   - 使用较小的批次大小
   - 确保GPU有足够的VRAM（推荐24GB+）

3. **transformers版本问题**
   ```bash
   pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2
   ```

### 获取帮助

- 查看项目README: [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley)
- 报告问题: [GitHub Issues](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley/issues)
- 论文: [arXiv:2508.16930](https://arxiv.org/abs/2508.16930)

## 模型下载

```bash
# 使用HuggingFace Hub
git clone https://huggingface.co/tencent/HunyuanVideo-Foley

# 或使用huggingface-cli
huggingface-cli download tencent/HunyuanVideo-Foley
```

## 配置文件

包安装后，配置文件位于：
- `hunyuanvideo_foley/configs/` 目录
- 默认配置：`configs/hunyuanvideo-foley-xxl.yaml`