# STREAM : Spatio-TempoRal Evaluation and Analysis Metric for Video Generative Models

Authors : Pumjun Kim, SeoJun Kim, [Jaejun Yoo](https://scholar.google.co.kr/citations?hl=en&user=7NBlQw4AAAAJ)

<a href='https://openreview.net/pdf?id=7JfKCZQPxJ'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/seo-jun-kim/STREAM'><img src='https://img.shields.io/badge/Code-Github-green'></a>

## 📌 News 📌
[2024.01.17] - 🎊 **STREAM** has been accepted by ICLR 2024! 🎊

## Abstract
> Image generative models have made significant progress in generating realistic and diverse images, supported by comprehensive guidance from various evaluation metrics. However, current video generative models struggle to generate even short video clips, with limited tools that provide insights for improvements. Current video evaluation metrics are simple adaptations of image metrics by switching the embeddings with video embedding networks, which may underestimate the unique characteristics of video. Our analysis reveals that the widely used Frechet Video Distance (FVD) has a stronger emphasis on the spatial aspect than the temporal naturalness of video and is inherently constrained by the input size of the embedding networks used, limiting it to 16 frames. Additionally, it demonstrates considerable instability and diverges from human evaluations. To address the limitations, we propose STREAM, a new video evaluation metric uniquely designed to independently evaluate spatial and temporal aspects. This feature allows comprehensive analysis and evaluation of video generative models from various perspectives, unconstrained by video length. We provide analytical and experimental evidence demonstrating that STREAM provides an effective evaluation tool for both visual and temporal quality of videos, offering insights into area of improvement for video generative models. To the best of our knowledge, STREAM is the first evaluation metric that can separately assess the temporal and spatial aspects of videos.

## Overview of STREAM-T and STREAM-S

