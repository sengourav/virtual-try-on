# Virtual Try-On (VTON) Framework

![Virtual Try-On Demo](docs/images/banner.png)

> An extensive framework implementing four different approaches to virtual clothing try-on

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/your-username/virtual-try-on)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

## Introduction

This repository contains implementations of four different approaches to virtual clothing try-on:

1. **Traditional CV-based**: Using image warping and blending techniques to transfer garments
2. **U-Net based**: Leveraging U-Net architecture for try-on generation
3. **GAN-based**: Using our improved VITON (Virtual Try-On Network) with attention mechanism
4. **Diffusion-based**: Implementing a diffusion model for high-quality try-on synthesis

Each method represents different stages in the evolution of virtual try-on technology, from traditional computer vision techniques to advanced generative models.

## Models

### 1. Traditional CV-based Approach

This approach uses classical computer vision techniques:
- Agnostic person representation generation
- Garment warping based on body pose estimation
- Alpha blending to combine the warped garment with the person

### 2. U-Net Based Approach

Implementation using U-Net architecture:
- Encoder-decoder network with skip connections
- Direct image-to-image translation
- Focuses on preserving details from both inputs

### 3. GAN Based Approach

GAN-based architecture with key components:
- `ImprovedUNetGenerator` that creates try-on results
- `PatchDiscriminator` for adversarial training
- Attention and residual blocks to improve quality
- Multiple loss functions (L1, perceptual, adversarial)

### 4. Diffusion Model Approach

State-of-the-art diffusion model for try-on:
- Denoising diffusion probabilistic model
- Progressive generation with controlled guidance
- Superior texture preservation and realistic details

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### GAN-based Model Training

```bash
python models/GAN/main.py \
  --dataset_path /path/to/dataset.zip \
  --extract_dir viton_dataset \
  --batch_size 4 \
  --epochs 50 \
  --use_gan \
```

### Inference

```bash
python models/GAN/inference_script.py \
  --checkpoint /path/to/checkpoint \
  --person /path/to/person_image \
  --cloth /path/to/garment_image \
  --output /path/to/output/filename.png
```


## Documentation

Detailed documentation for each approach:
- `method_descriptions.md`: Technical overview of each approach
- `model_specifications.md`: Architecture details and parameters
- `training_procedures.md`: Training protocols and hyperparameters
- `results_comparison.md`: Quantitative and qualitative comparison

## Hugging Face Demo

Try our demo on Hugging Face Spaces: [Virtual Try-On Demo](https://huggingface.co/spaces/your-username/virtual-try-on)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
