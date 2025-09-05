#!/bin/bash

# Download MoviegenAudioBenchSfx 10 videos
curl -O https://texttoaudio-train-1258344703.cos.ap-guangzhou.myqcloud.com/hunyuanvideo-foley_demo/MovieGenAudioBenchSfx.tar.gz
tar -xzvf MovieGenAudioBenchSfx.tar.gz -C ./assets
rm MovieGenAudioBenchSfx.tar.gz

# Download gradio example video
curl -O https://texttoaudio-train-1258344703.cos.ap-guangzhou.myqcloud.com/hunyuanvideo-foley_demo/examples.tar.gz
tar -xvzf examples.tar.gz
rm examples.tar.gz
