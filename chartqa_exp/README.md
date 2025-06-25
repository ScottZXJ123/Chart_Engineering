# ChartQA Experiment

This repository contains the experimental setup for ChartQA research.

## Directory Structure

```
chartqa_exp/
├── env.yml          # Conda environment configuration
├── data/            # Data directory
│   ├── pot/         # Program of Thought data
│   └── raw/         # Raw data files
├── scripts/         # Python scripts and utilities
└── README.md        # This file
```

## Setup

1. Create the conda environment:
```bash
conda env create -f env.yml
```

2. Activate the environment:
```bash
conda activate chartqa_exp
```

## Usage

Place your data files in the appropriate directories:
- Raw data files go in `data/raw/`
- Program of Thought (PoT) related data goes in `data/pot/`
- Python scripts and utilities go in `scripts/`

## Dependencies

The environment includes:
- PyTorch 2.0.1 with CUDA 11.8 support
- Transformers 4.31.0 for NLP models
- Computer vision libraries (OpenCV, Pillow)
- Data science tools (NumPy, Pandas, Matplotlib)
- Chart-specific libraries (Plotly, Chart-studio)
- ML experiment tracking (Weights & Biases, TensorBoard)
- Configuration management (Hydra, OmegaConf)

See `env.yml` for the complete list of dependencies.