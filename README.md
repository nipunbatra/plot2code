# Plot2Code

A vision-language model pipeline for generating matplotlib code from plot images. This project uses the Qwen2.5-VL model to perform zero-shot code generation from visualization images.

## Overview

Plot2Code generates a dataset of matplotlib plots paired with their source code, then evaluates vision-language models on their ability to recreate plots by generating the corresponding matplotlib code.

## Features

- Automated dataset generation from matplotlib examples
- Zero-shot code generation using Qwen2.5-VL-7B-Instruct
- Comprehensive evaluation metrics including:
  - Visual similarity (SSIM)
  - Code similarity (BLEU, edit distance)
  - API usage overlap
- Side-by-side visual comparison of original and regenerated plots

## Project Structure

```
plot2code/
├── plot2code_pipeline.ipynb  # Main pipeline notebook
├── plot2code_exec.jsonl      # Generated dataset (image paths + code)
├── mpl_examples/             # Downloaded matplotlib examples
├── plot_gallery_exec/        # Generated plot images
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Transformers library
- matplotlib
- numpy
- Pillow
- scikit-image
- nltk
- tqdm

## Installation

```bash
pip install torch transformers matplotlib numpy pillow scikit-image nltk tqdm
```

## Usage

### 1. Dataset Generation

The pipeline automatically downloads matplotlib examples and generates plot images with their source code:

```python
# Configure which plot categories to include
keep = [
    "lines_bars_and_markers",
    "statistics",
    "pie_and_polar_charts",
    "mplot3d",
]
```

Run the data generation cells in [plot2code_pipeline.ipynb](plot2code_pipeline.ipynb) to create the dataset.

### 2. Model Loading

The pipeline uses Qwen2.5-VL-7B-Instruct for code generation:

```python
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)
```

### 3. Code Generation

Generate matplotlib code from a plot image:

```python
code = gen_code_for_image("path/to/plot.png")
```

### 4. Evaluation

Evaluate generated code with comprehensive metrics:

```python
evaluate_plot2code_verbose("path/to/plot.png", reference_code)
```

This provides:
- Visual similarity score (SSIM)
- Code similarity metrics (BLEU, edit distance)
- API usage analysis (matched, missed, and extra function calls)
- Side-by-side visual comparison

## Evaluation Metrics

### Visual Metrics
- **SSIM (Structural Similarity Index)**: Measures pixel-level similarity between original and regenerated plots

### Code Metrics
- **BLEU**: Token-level similarity between reference and generated code
- **Edit Similarity**: Character-level edit distance
- **API Overlap**: Proportion of matplotlib function calls that match the reference

### Composite Scores
- **CodeScore**: Weighted combination of BLEU (0.4), edit similarity (0.3), and API overlap (0.3)
- **Overall Score**: Equal weighting of visual (SSIM) and code metrics (0.5 each)

## Dataset Format

The generated dataset is stored in JSONL format:

```json
{"image": "plot_gallery_exec/example.png", "code": "import matplotlib.pyplot as plt\n..."}
```

## Future Improvements

- Run batch evaluation over all dataset images
- Implement exec@1 metric (execution success rate)
- Fine-tune model on the generated dataset
- Create separate train/test splits
- Compare zero-shot vs fine-tuned performance
- Support additional plot libraries (seaborn, plotly)

## License

This project uses matplotlib examples which are under the [matplotlib license](https://matplotlib.org/stable/users/project/license.html).

## Citation

If you use this work, please cite:

```bibtex
@software{plot2code,
  title = {Plot2Code: Vision-Language Model for Plot Code Generation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/plot2code}
}
```

## Acknowledgments

- Matplotlib development team for the comprehensive example gallery
- Qwen team for the Qwen2.5-VL model
- Hugging Face for the Transformers library
