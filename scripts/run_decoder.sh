# !/usr/bin/bash
# coding: utf-8

python -u video_watermark.py \
    --input_videos_dir ./data/output_results/results_0523/encoded_videos \
    --output_dir ./data/output_results/results_0523 \
    --watermark_path ./data/water_marking/water_marking_32_2000.pickle \
    --label_path ./data/output_results/results_0523/videomark_label.npy \
    --decode
