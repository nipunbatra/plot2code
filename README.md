# Plot2Code

A vision-language model pipeline for generating matplotlib code from plot images using Qwen2.5-VL with PEFT/LoRA fine-tuning.

## Overview

This project generates a dataset of matplotlib plots paired with their source code, evaluates the Qwen2.5-VL model on zero-shot code generation, fine-tunes the model using LoRA, and compares performance with comprehensive metrics including SSIM, BLEU, edit distance, and API overlap.

## Usage

Run `plot2code_pipeline.ipynb` to generate data, train the model with LoRA (100 examples, 10 epochs), and evaluate on a held-out test set (30 examples). The notebook produces comparison tables, loss curves, and performance visualizations.

## Requirements

Python 3.8+, PyTorch, transformers, peft, datasets, matplotlib, numpy, pillow, scikit-image, nltk, pandas, tqdm
