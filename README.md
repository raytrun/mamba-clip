# CLIP-Mamba: CLIP Pretrained Mamba Models with OOD and Hessian Evaluation 
[[Paper](https://arxiv.org/pdf/2404.19394)][[🤗HF](https://huggingface.co/weiquan/mamba-clip/tree/main)]
## Abstract
'''State space models and Mamba-based models have been increasingly applied
across various domains, achieving state-of-the-art performance. This technical
report introduces the first attempt to train a transferable Mamba model utilizing
contrastive language-image pretraining (CLIP). We have trained Mamba models
of varying sizes and undertaken comprehensive evaluations of these models on 26
zero-shot classification datasets and 16 out-of-distribution (OOD) datasets. Our
findings reveal that a Mamba model with 67 million parameters is on par with a 307
million-parameter Vision Transformer (ViT) model in zero-shot classification tasks,
highlighting the parameter efficiency of Mamba models. In tests of OOD generalization, Mamba-based models exhibit exceptional performance in conditions of
OOD image contrast or when subjected to high-pass filtering. However, a Hessian
analysis indicates that Mamba models feature a sharper and more non-convex
landscape compared to ViT-based models, making them more challenging to train.'''

## Main results
# Zero-shot performance of different architectures trained with CLIP
| Methods  | Food-101 | CIFAR-10 | CIFAR-100 |  CUB  | SUN397 | Cars  | Aircraft |  DTD  | Pets  | Caltech-101 | Flowers | MNIST | FER-2013 | STL-10 | EuroSAT | RESISC45 | GTSRB | KITTI | Country211 | PCAM | UCF101 | Kinetics700 | CLEVR | HatefulMemes | SST2 | ImageNet | Average |  
|:--------:|:--------:|:--------:|:---------:|:-----:|:------:|:-----:|:--------:|:-----:|:-----:|:-----------:|:-------:|:-----:|:-------:|:------:|:-------:|:-------:|:-----:|:-----:|:---------:|:----:|:------:|:----------:|:-----:|:-----------:|:----:|:--:|:-------:|  
| VMamba_B (89M) | 48.5 | 58.0 | 29.9 | 36.5 | 50.4 | 5.8 | 8.5 | 26.5 | 30.2 | 64.7 | 52.8 | 9.7 | 19.6 | 91.9 | 16.0 | 30.4 | 7.9 | 40.2 | 10.2 | 59.9 | 35.2 | 25.6 | 12.6 | 51.6 | 50.1 | 38.3 |
| VMamba_S (50M) | 49.4 | 70.3 | 34.3 | 39.1 | 53.9 | 6.9 | 8.4 | 26.0 | 31.3 | 68.7 | 54.1 | 10.1 | 9.8 | 92.8 | 17.6 | 31.4 | 6.9 | 23.5 | 10.9 | 54.2 | 38.4 | 27.1 | 13.2 | 50.5 | 50.0 | 40.0 |
| VMamba_T220 (30M) | 46.5 | 50.9 | 22.9 | 35.6 | 51.1 | 5.7 | 6.8 | 25.1 | 31.0 | 64.9 | 54.0 | 10.1 | 12.5 | 91.6 | 13.9 | 25.4 | 10.7 | 32.3 | 9.9 | 55.0 | 34.0 | 25.1 | 12.7 | 53.9 | 50.6 | 38.7 |
| Simba_L (66.6M) | 52.7 | 67.4 | 31.0 | 39.1 | 52.7 | 6.9 | 9.1 | 27.8 | 33.4 | 68.9 | 55.9 | 8.0 | 16.0 | 93.9 | 17.4 | 32.3 | 8.9 | 41.5 | 11.1 | 58.1 | 35.7 | 27.9 | 12.1 | 54.9 | 50.1 | 41.6 |
| VIT_B(84M) | 50.6 | 66.0 | 34.5 | 38.8 | 51.1 | 4.0 | 5.4 | 21.2 | 28.5 | 60.9 | 53.3 | 8.4 | 17.3 | 90.5 | 30.2 | 21.5 | 6.1 | 35.1 | 10.5 | 53.5 | 28.5 | 22.1 | 10.8 | 52.4 | 50.7 | 37.6 |
| VIT-L(307M) | 59.5 | 72.9 | 41.5 | 40.3 | 53.6 | 6.9 | 6.4 | 20.6 | 27.9 | 65.4 | 55.0 | 10.3 | 34.5 | 94.2 | 22.7 | 28.8 | 5.8 | 41.4 | 12.5 | 54.9 | 34.3 | 24.0 | 12.9 | 54.3 | 50.1 | 40.4 |



## Acknowledgment
This project is based on A-CLIP ([paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Attentive_Mask_CLIP_ICCV_2023_paper.html), [code](https://github.com/microsoft/A-CLIP)), VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), SiMBA ([paper](https://arxiv.org/html/2403.15360v2), [code](https://github.com/badripatro/simba)), thanks for their excellent works.
