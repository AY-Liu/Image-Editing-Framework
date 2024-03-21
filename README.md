# Editing-Framework

## Introduction

Welcome to this repository, a comprehensive collection of the four pivotal algorithms in the [specific domain] domain: [P2P](https://arxiv.org/abs/2208.01626), [MasaCtrl](https://arxiv.org/abs/2304.08465), [PnP](https://arxiv.org/abs/2211.12572), and [Pix2Pix-zero](https://arxiv.org/abs/2302.03027). This repository aims to provide an accessible, well-documented, and easy-to-understand resource for both beginners and experienced practitioners looking to explore and implement these algorithms.

## Why do I make this repository?

Different editing methods demand specific environments, making it challenging to compare them due to varying requirements, such as diffusers versions. This often complicates the research process for those needing to install numerous environments. Our project standardizes these methods and is updated for the latest diffusers version (0.27), simplifying comparisons.

## Difference with official repositories

1. The project integrates various inversion methods and interfaces for different stable diffusion versions (1.5, 2.1, XL), allowing researchers to easily switch between methods.
2. It clarifies the logic behind editing methods by structuring the codes, helping researchers understand their similarities and differences.
3. It Provides access to the PIE Benchmark, facilitating testing of editing methods on the same benchmark.
4. Importantly, to ensure fairness in comparing different methods, specific engineering techniques unique to each method (such as using masks for local editing in p2p, or the use of Edit Directions in p2p-zero) are omitted. However, researchers can easily incorporate these elements.

## Environment

### Python

```bash
pip install -r requirements.txt
```

> We assume that you have already installed the pytorch.

### Weights

You could download the weights of SD from [huggingface](https://huggingface.co/), and modify the file `sd_mapping.py` for each method.

## Getting Started

We give three simple examples for using. All methods share the same structure.

### Edit the real image using p2p

```bash
cd p2p
python edit_real.py --sd_version 1.5 --device 0 --seed 0 --source_prompt "a gray horse in the field" --target_prompt "a whie horse in the field" --source_image ./test.jpg --inversion_type null-text
```

### Edit the synthesis image using pnp

```
cd pnp
python edit_syn.py --xx xx..
```

### Test masactrl method on PIE-Bench

First you should download the [PIE-Bench](https://docs.google.com/forms/d/e/1FAIpQLSftGgDwLLMwrad9pX3Odbnd4UXGvcRuXDkRp6BT1nPk8fcH_g/viewform?usp=send_form) dataset.

```bash
cd masactrl
python test.py
```

## Acknowledgement

Our code is based on [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero) , [Plug-and-Play](https://github.com/MichalGeyer/plug-and-play), [DirectionInversion](https://github.com/cure-lab/PnPInversion) thanks to all the contributors!