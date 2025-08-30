# ComfyUI – HunyuanVideo‑Foley (Hybrid)

A tidy set of nodes for **Tencent HunyuanVideo‑Foley** that runs on modest GPUs and scales up nicely.

## Node overview (start here)

* **Hunyuan‑Foley Model Loader** – loads the main model. Two simple knobs:

  * **Precision**: runtime math quality (bf16/fp16/fp32).
  * **FP8 Quantization** (weight‑only): lowers VRAM usage < 12GB. *Turn this on if you’re GPU‑poor.*
* **Hunyuan‑Foley Dependencies Loader** – loads DAC‑VAE, SigLIP2, Synchformer, and CLAP.
* **Hunyuan‑Foley Sampler** – makes the audio. Images are **optional** (works great as **Text→Audio**). Supports **negative prompt** and **batching**.
* **Hunyuan‑Foley Torch Compile** (optional) – uses `torch.compile` for speed. First run compiles; repeats are **\~30% faster**.

## Quick start

1. Drop **Model Loader → Dependencies Loader → (optional) Torch Compile → Sampler**.
2. For **Text→Audio**, leave the image input empty. For **Video→Audio**, connect an image sequence and set `frame_rate`.
3. Tweak **Prompt** and **Negative Prompt**. Leave sampler on **Euler**, `CFG≈4.5`, `Steps≈50`.
4. Press **Queue** and preview the audio.

## Where to put the model files

Download from Hugging Face:
[https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main](https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main)

Place them in **`ComfyUI/models/foley/`**:

```
hunyuanvideo_foley.pth         # ~10.3 GB  main model
synchformer_state_dict.pth     # ~0.95 GB  sync encoder
vae_128d_48k.pth               # ~1.49 GB  DAC‑VAE
```

> Tested with **PyTorch 2.7 and 2.8**.

## The two dropdowns

* **Precision** = how carefully the math runs. `bf16`/`fp16` are fast and standard; `fp32` is heaviest. Pick `bf16` (default) or `fp16` on 30‑series GPUs if you prefer.
* **FP8 Quantization** = store big Linear weights in **FP8** to save memory. Compute still runs in `Precision`, so sound quality holds.

  * **`auto`** tries to match the checkpoint or uses a safe default.
  * Expect **less VRAM**, not more speed.

## Memory & speed at a glance

* Typical 5s / 50 steps on a 24 GB card:

  * Baseline: \~10–12 GB
  * With ping‑pong offloading (built‑in): \~9–10 GB
  * **With FP8 quant**: subtract another **\~1–2+ GB**
  * **Torch Compile**: after the first compile, runs are **\~30% faster**
* **Under‑12 GB recipe:** set **FP8 Quant** on, keep **batch\_size=1**, steps ≤ **50**. That’s it.

## Batching

* `batch_size` generates multiple variations at once. VRAM scales roughly with batch size.
* Use **Select Audio From Batch** to pick the clip you like.

## Tips & fixes

* If you OOM, drop `batch_size`, reduce `steps`, or enable **force\_offload** in the sampler.

## Credits

* Model & weights: **Tencent HunyuanVideo‑Foley**.
* ComfyUI and community for the scaffolding.
* This repo adds VRAM‑friendly loading, **FP8** weight‑only option, and an optional **torch.compile** speed path.
