# STREAM : Spatio-TempoRal Evaluation and Analysis Metric for Video Generative Models

Authors : Pumjun Kim, SeoJun Kim, [Jaejun Yoo](https://scholar.google.co.kr/citations?hl=en&user=7NBlQw4AAAAJ)

<a href='https://openreview.net/pdf?id=7JfKCZQPxJ'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/seo-jun-kim/STREAM'><img src='https://img.shields.io/badge/Code-Github-green'></a> <a href='https://colab.research.google.com/drive/1LriUNB9g8lIrbPCeKDuQbN4I_ALvrg8m?usp=sharing'><img src='https://img.shields.io/badge/Code-Colab-F9AB00'></a>

## 📌 News 📌
[2024.01.17] - 🎊 **STREAM** has been accepted by ICLR 2024! 🎊

## Abstract
> Image generative models have made significant progress in generating realistic and diverse images, supported by comprehensive guidance from various evaluation metrics. However, current video generative models struggle to generate even short video clips, with limited tools that provide insights for improvements. Current video evaluation metrics are simple adaptations of image metrics by switching the embeddings with video embedding networks, which may underestimate the unique characteristics of video. Our analysis reveals that the widely used Frechet Video Distance (FVD) has a stronger emphasis on the spatial aspect than the temporal naturalness of video and is inherently constrained by the input size of the embedding networks used, limiting it to 16 frames. Additionally, it demonstrates considerable instability and diverges from human evaluations. To address the limitations, we propose STREAM, a new video evaluation metric uniquely designed to independently evaluate spatial and temporal aspects. This feature allows comprehensive analysis and evaluation of video generative models from various perspectives, unconstrained by video length. We provide analytical and experimental evidence demonstrating that STREAM provides an effective evaluation tool for both visual and temporal quality of videos, offering insights into area of improvement for video generative models. To the best of our knowledge, STREAM is the first evaluation metric that can separately assess the temporal and spatial aspects of videos.

## 💡 Overview of STREAM 💡
![stream_overview](https://github.com/pro2nit/STREAM/files/14048894/figure1.pdf)

# Quick Start
You can install our method using `pip` command!
```
pip install v-stream
```

## How to use
In this example, we evaluate generated samples in `./video/fake` within `./video/real`. (you can change directory)
To follow this example, video data should follow conditions as :
```bash
<vid_dir>
├── vid_00000.npy
├── vid_00001.npy
├── ...
├── vid_02046.npy
└── vid_02047.npy

vid_%05d.npy
* numpy.ndarray
* dtype : np.uint8 (0 ~ 255)
* shape : (f, h, w, c) 
```
### 1. Call packages
```python
# Call packages
from stream import STREAM

# for DINOv2 : mode = 'dinov2'
stream = STREAM(num_frame=16, model='swav')
```

<details> <summary> <h4> CUSTOM EMBEDDER </h4></summary>
 
  * current stream version(0.1.0) supports embedder with `SwAV` and `DINOv2`.
  
  ```python
  # swav
  embedder = torch.hub.load('facebookresearch/swav:main', 'resnet50')
  # dinov2
  embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
  ```
    
  * If you want custom embedder, you can try below :
  ```python
  NUM_EMBED = "LENGTH OF EMBEDDING VECTOR - int"
  CUSTOM_EMBEDDER = "CUSTOM EMBEDDER - torch.nn.Module"

  stream = STREAM(num_frame=16, num_embed=NUM_EMBED)
  stream.embedder = CUSTOM_EMBEDDER
  ```

</details>

### 2. Calculate Skewness & Compute Mean signal
```python
real_dir = './video/real'
fake_dir = './video/fake/'

real_skewness, real_mean_signal = stream.calculate_skewness(real_dir, 'cuda', batch_size=4)
fake_skewness, fake_mean_signal = stream.calculate_skewness(fake_dir, 'cuda', batch_size=4)
```

### 3. Compute ***STREAM-T*** between real and fake skewness
```python
# STREAM-Temporal
stream_T = stream.stream_T(real_skewness, fake_skewness)
print('STREAM-T :', stream_T)
```
above code will print out as below :
```
> STREAM-T : 0.729215577505656
```

### 4. Compute ***STREAM-F*** and ***STREAM-D*** between real and fake mean signals
```python
# STREAM-Spatio
stream_S = stream.stream_S(real_mean_signal, fake_mean_signal)

# STREAM-Fidelity
stream_F = stream_S['stream_F']
# STREAM-Diversity
stream_D = stream_S['stream_D']

print('STREAM-F :', stream_F)
print('STREAM-D :', stream_D)
```
above code will print out as below :
```
> Num real: 100 Num fake: 100
> STREAM-F : 0.96
> STREAM-D : 0.87 
```

# Citation
If you find this repository useful for your research, please cite the following work.
```
@article{kim2024stream,
  title={STREAM : Spatio-TempoRal Evaluation and Analysis Metric for Video Generative Models},
  author={Kim, Pum Jun and Kim, Seojun and and Yoo, Jaejun},
  journal={arXiv preprint arXiv:2306.08013},
  year={2024}
}
```





