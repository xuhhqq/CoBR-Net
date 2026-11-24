This repository is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

# CoBR-Net
Official PyTorch implementation of **CoBR-Net** for high-resolution remote sensing road extraction, accepted/submitted to GIScience & Remote Sensing.

CoBR-Net is a coarse-to-fine semantic-boundary refinement network. It integrates the C2F-SE Encoder, Guided Adaptive Interaction Module (GAIM), and Edge-Guided Detail Aggregation module (EGDA) to jointly model semantic abstraction, multi-scale feature interaction and edge-aware detail recovery. The network aims to improve road structural consistency and boundary discrimination, achieving competitive accuracy-efficiency trade-off for remote sensing road extraction.

## Datasets

We evaluate CoBR‑Net on three public road‑extraction benchmarks and perform zero‑shot generalization tests on large‑scale scenes from the **LSRV (Large‑Scale Road Validation Dataset)**.
This repository **does not contain any raw dataset files**. Please download datasets from their respective official sources and organize them following the official protocols and our data configuration files.

### Benchmark Datasets
1. **DeepGlobe Road Dataset**
Official challenge page: https://competitions.codalab.org/competitions/18467#participate-get_data
> A high‑resolution road segmentation dataset released for the DeepGlobe 2018 Challenge. We follow the dataset split configuration from existing published works.

2. **CHN6‑CUG Road Dataset**
Official repository: https://github.com/CUG‑URS/CHN6‑CUG‑Roads‑Dataset
> Large‑scale Chinese road dataset covering six representative cities with diverse road types and surface conditions.

3. **SpaceNet Road Dataset (SpaceNet 3 Road Network Detection)**
Official page: https://spacenet.ai/spacenet‑roads‑dataset/
> Multi‑city satellite road dataset, road masks are generated from OSM centerlines with fixed‑width rendering. Our data split follows previous literature.

### Large‑scale Zero‑shot Generalization Scenes (LSRV Dataset)
The large‑scale Birmingham and Wuhan (Shanghai in original LSRV release) zero‑shot test scenes are sourced from the **LSRV dataset**, published by RSIDEA group, Wuhan University.
Official download page: https://rsidea.whu.edu.cn/resource_LSRV_sharing.htm

> Note: The Wuhan scene in our paper corresponds to the Shanghai large‑scale image from LSRV, naming convention inherited from prior research. LSRV is for academic‑research‑only use; please read its license and cite the original LSRV paper when you use it.

## Environment
- Python >=3.8
- PyTorch >=1.10
- torchvision
- OpenCV, albumentations, rasterio, tqdm
