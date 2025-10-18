# Plot2Code

A vision-language model pipeline for generating matplotlib code from plot images. This project uses the Qwen2.5-VL model with PEFT (Parameter-Efficient Fine-Tuning) to perform both zero-shot and fine-tuned code generation from visualization images.

## Overview

Plot2Code generates a dataset of matplotlib plots paired with their source code, then evaluates vision-language models on their ability to recreate plots by generating the corresponding matplotlib code. The project includes comprehensive evaluation metrics, fine-tuning capabilities with LoRA, and detailed performance comparisons.

## Features

### Core Capabilities
- ✨ Automated dataset generation from matplotlib examples
- 🤖 Zero-shot code generation using Qwen2.5-VL-7B-Instruct
- 🔧 PEFT/LoRA fine-tuning for improved performance
- 📊 Comprehensive evaluation metrics including:
  - Visual similarity (SSIM)
  - Code similarity (BLEU, edit distance)
  - API usage overlap
  - Execution success rate
- 📈 Training loss visualization and tracking
- 🔍 Side-by-side visual comparison of original and regenerated plots
- 📉 Zero-shot vs fine-tuned performance comparison

### Evaluation & Analysis
- Quantitative metrics on test set with mean ± std
- Qualitative analysis showing best/worst performing examples
- Detailed comparison tables with absolute and relative improvements
- Visual charts for performance comparison
- Training convergence analysis with loss curves

## Project Structure

```
plot2code/
├── plot2code_pipeline.ipynb       # Main pipeline notebook
├── generate_data.ipynb             # Data generation utilities
├── plot2code_exec.jsonl            # Generated dataset (image paths + code)
├── mpl_examples/                   # Downloaded matplotlib examples
├── plot_gallery_exec/              # Generated plot images
├── plot2code_lora/                 # Fine-tuned LoRA checkpoints
├── logs/                           # Training logs
├── zeroshot_eval_results.csv       # Zero-shot baseline results
├── finetuned_eval_results.csv      # Fine-tuned model results
├── comparison_zeroshot_vs_finetuned.csv  # Comparison table
├── training_loss_curve.png         # Training loss visualization
├── comparison_zeroshot_vs_finetuned.png  # Performance comparison chart
└── README.md                       # This file
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended for GPU acceleration)
- Transformers library (with Qwen2.5-VL support)
- PEFT library (for LoRA fine-tuning)
- matplotlib
- numpy
- Pillow
- scikit-image
- nltk
- tqdm
- pandas
- datasets

## Installation

```bash
# Install core dependencies
pip install torch transformers peft datasets

# Install evaluation dependencies
pip install matplotlib numpy pillow scikit-image nltk tqdm pandas seaborn
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

Run the data generation cells in `plot2code_pipeline.ipynb` to create the dataset.

### 2. Train/Test Split

Create disjoint datasets for training and evaluation:

```python
# 80/20 split with random seed for reproducibility
train_dataset = Dataset.from_list(train_data)  # 100 examples
test_dataset = Dataset.from_list(test_data)    # 30 examples
```

### 3. Zero-Shot Baseline

Evaluate the pre-trained model without fine-tuning:

```python
# Load model
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)

# Run zero-shot evaluation
results = evaluate_on_test_set(test_dataset)
```

### 4. Fine-Tuning with PEFT/LoRA

Apply parameter-efficient fine-tuning:

```python
# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

# Train for 10 epochs on 100 examples
trainer.train()
```

### 5. Evaluation & Comparison

Compare zero-shot vs fine-tuned performance:

```python
# Evaluate both models
zeroshot_results = evaluate_model(test_dataset, model_zeroshot)
finetuned_results = evaluate_model(test_dataset, model_finetuned)

# Generate comparison table and visualizations
create_comparison_table(zeroshot_results, finetuned_results)
```

## Evaluation Metrics

### Visual Metrics
- **SSIM (Structural Similarity Index)**: Measures pixel-level similarity between original and regenerated plots (range: 0-1)

### Code Metrics
- **BLEU**: Token-level similarity between reference and generated code
- **Edit Similarity**: Character-level edit distance (SequenceMatcher ratio)
- **API Overlap**: Proportion of matplotlib function calls that match the reference

### Composite Scores
- **CodeScore**: Weighted combination of BLEU (0.4), edit similarity (0.3), and API overlap (0.3)
- **Overall Score**: Equal weighting of visual (SSIM) and code metrics (0.5 each)

### Success Metrics
- **Execution Success Rate**: Percentage of generated code that executes without errors

## Training Configuration

### Dataset
- **Training examples**: 100 (code length < 1000 chars)
- **Test examples**: 30 (disjoint from training)
- **Train/test split**: 80/20 with random seed 42

### Hyperparameters
- **Epochs**: 10
- **Batch size**: 1 (per device)
- **Gradient accumulation**: 4 (effective batch size = 4)
- **Learning rate**: 2e-4
- **Warmup steps**: 30
- **LoRA rank**: 16
- **LoRA alpha**: 32
- **LoRA dropout**: 0.05

## Results & Outputs

The pipeline generates:

1. **CSV Files**:
   - `zeroshot_eval_results.csv` - Zero-shot baseline metrics
   - `finetuned_eval_results.csv` - Fine-tuned model metrics
   - `comparison_zeroshot_vs_finetuned.csv` - Comparison table
   - `training_loss_history.csv` - Loss values over training

2. **Visualizations**:
   - `training_loss_curve.png` - Training loss over steps
   - `comparison_zeroshot_vs_finetuned.png` - Performance comparison charts
   - `finetuned_metrics_distribution.png` - Metrics distributions
   - `finetuned_metrics_boxplot.png` - Boxplot comparison

3. **Model Checkpoints**:
   - `plot2code_lora_final/` - Final LoRA adapter weights
   - `plot2code_lora/checkpoint-*` - Epoch checkpoints

## Dataset Format

The generated dataset is stored in JSONL format:

```json
{"image": "plot_gallery_exec/example.png", "code": "import matplotlib.pyplot as plt\nimport numpy as np\n\nplt.plot([1, 2, 3], [1, 4, 9])\nplt.show()"}
```

Each line contains:
- `image`: Path to the generated plot image
- `code`: Source code that generated the plot

## Notebook Structure

### plot2code_pipeline.ipynb

**Steps 1-9**: Setup, data generation, model loading, evaluation functions
- Download matplotlib examples
- Generate plot dataset
- Load Qwen2.5-VL model
- Define code generation and evaluation functions

**Step 9.5**: Prepare train/test split
- Create disjoint datasets (100 train / 30 test)

**Step 9.6**: Zero-shot baseline evaluation
- Evaluate pre-trained model on test set

**Step 10**: Apply PEFT/LoRA
- Configure and apply LoRA adapters

**Step 11**: Training setup and execution
- Define collator function
- Configure trainer with 10 epochs
- Run fine-tuning

**Step 11.5**: Visualize training loss
- Plot loss curve with trend line
- Display convergence statistics

**Step 12**: Fine-tuned evaluation
- Evaluate fine-tuned model on test set

**Step 12.5**: Comparison table
- Compare zero-shot vs fine-tuned
- Calculate improvements (absolute & relative)
- Generate comparison visualizations

**Step 13**: Qualitative analysis
- Show best/worst performing examples
- Visual comparisons

**Step 14**: Results visualization
- Save all results and charts

## Example Results

Typical improvements from fine-tuning (example):
- **Overall Score**: 0.650 → 0.720 (+10.8% relative improvement)
- **SSIM (Visual)**: 0.750 → 0.780 (+4.0%)
- **Code Score**: 0.550 → 0.660 (+20.0%)
- **Execution Success**: 85% → 95% (+11.8%)

## Performance Tips

1. **For better results**:
   - Increase training examples (100+ recommended)
   - Train for more epochs (10-15)
   - Use larger LoRA rank (16-32)

2. **For faster training**:
   - Reduce batch size if OOM
   - Use gradient accumulation
   - Enable bf16/fp16 mixed precision

3. **For better evaluation**:
   - Use larger test set (30+ examples)
   - Filter examples by complexity
   - Analyze loss curve for convergence

## Future Improvements

- ✅ ~~Fine-tune model on the generated dataset~~ (Completed)
- ✅ ~~Create separate train/test splits~~ (Completed)
- ✅ ~~Compare zero-shot vs fine-tuned performance~~ (Completed)
- ✅ ~~Add training loss visualization~~ (Completed)
- 🔄 Scale to larger datasets (1000+ examples)
- 🔄 Implement exec@k metrics (k > 1)
- 🔄 Support additional plot libraries (seaborn, plotly)
- 🔄 Add validation set for early stopping
- 🔄 Experiment with different VLM architectures

## License

This project uses matplotlib examples which are under the [matplotlib license](https://matplotlib.org/stable/users/project/license.html).

## Citation

If you use this work, please cite:

```bibtex
@software{plot2code,
  title = {Plot2Code: Vision-Language Model for Plot Code Generation with PEFT},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/plot2code}
}
```

## Acknowledgments

- Matplotlib development team for the comprehensive example gallery
- Qwen team for the Qwen2.5-VL model
- Hugging Face for the Transformers and PEFT libraries
- The open-source community for evaluation tools and libraries

## Troubleshooting

### Common Issues

**Q: Getting OOM (Out of Memory) errors during training?**
A: Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps`, or reduce the number of training examples.

**Q: Training loss not decreasing?**
A: Check the loss curve visualization. Try increasing learning rate, reducing warmup steps, or training for more epochs.

**Q: Low execution success rate?**
A: Filter dataset for simpler examples (shorter code), or increase training examples to improve code quality.

**Q: Zero-shot and fine-tuned scores are similar?**
A: Increase dataset size (100+ examples), train for more epochs (15+), or use higher LoRA rank (32+).

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository.
