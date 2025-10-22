# Pneumonia Detection

Deep learning-based pneumonia detection from chest X-ray images using VGG16.

## Features

- Binary classification (Pneumonia vs No Pneumonia)
- VGG16 transfer learning
- Data augmentation
- Class balancing
- DICOM support
- Comprehensive evaluation metrics

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone <your-repo-url>
cd pneumonia-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Create Sample Data
```bash
# Create 50 sample images and labels
python scripts/create_sample_data.py 50
```

This creates:
- `data/raw/images/` - Sample X-ray images
- `data/raw/sample_labels.csv` - Labels CSV

### 3. Train Model
```bash
# Train with default settings (quick test with 5 epochs)
python scripts/train.py --epochs 5 --batch-size 8

# Full training
python scripts/train.py --epochs 50 --batch-size 32
```

### 4. Evaluate Model
```bash
python scripts/evaluate.py --model models/saved/final_model.h5
```

### 5. Make Predictions
```bash
python scripts/predict.py path/to/image.dcm --model models/saved/final_model.h5
```

## Project Structure
```
pneumonia-detection/
├── config/              # Configuration
├── data/                # Data handling
├── models/              # Model architectures
├── training/            # Training utilities
├── evaluation/          # Evaluation tools
├── inference/           # Inference pipeline
├── scripts/             # Executable scripts
├── tests/               # Unit tests
└── data/raw/            # Data directory
```

## Testing
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Usage Examples

### Training with Custom Parameters
```bash
python scripts/train.py \
    --data-dir data/raw \
    --epochs 30 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --use-clahe
```

### Using Your Own Data

1. Prepare your data:
   - Images in a directory (PNG/JPEG)
   - CSV with columns: `Image Index`, `Finding Labels`, `Patient Age`, `Patient Gender`

2. Train:
```bash
python scripts/train.py \
    --data-dir path/to/your/data \
    --labels-file your_labels.csv \
    --images-dir your_images
```

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- See `requirements.txt` for full list

## License

MIT License

## Contributing

Contributions welcome! Please open an issue first.