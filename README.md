# Domain Adaptation Techniques

A comprehensive implementation of domain adaptation methods for transferring knowledge from source to target domains, featuring adversarial training, CORAL alignment, and advanced evaluation metrics.

## Safety Disclaimer

**This project is for research and educational purposes only.**

- **Not for production use**: These models are research demonstrations and should not be used for real-world decision making
- **Limited generalization**: Models are trained on specific datasets and may not generalize to other domains
- **Human oversight required**: Any critical applications require human supervision and validation
- **No guarantees**: Results should be interpreted with caution and proper domain expertise

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Domain-Adaptation-Techniques.git
cd Domain-Adaptation-Techniques

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from domain_adaptation.models import AdversarialDomainAdaptationModel
from domain_adaptation.data import load_mnist_data, load_usps_data
from domain_adaptation.train import AdversarialTrainer

# Load data
source_data, source_targets = load_mnist_data()
target_data, target_targets = load_usps_data()

# Create model
model = AdversarialDomainAdaptationModel()

# Train
trainer = AdversarialTrainer(model)
trainer.train(source_loader, target_loader, num_epochs=50)
```

### Interactive Demo

```bash
# Run the Streamlit demo
streamlit run demo/streamlit_demo.py
```

## Dataset Schema

### MNIST (Source Domain)
- **Size**: 60,000 training images, 10,000 test images
- **Format**: 28×28 grayscale images
- **Classes**: 10 digit classes (0-9)
- **License**: Public domain

### USPS (Target Domain)
- **Size**: 7,291 training images, 2,007 test images
- **Format**: 16×16 grayscale images
- **Classes**: 10 digit classes (0-9)
- **License**: Public domain

### Domain Shift Characteristics
- Different image resolutions (28×28 vs 16×16)
- Different writing styles and digit appearances
- Varying image quality and preprocessing

## Architecture

### Project Structure

```
Domain-Adaptation-Techniques/
├── src/
│   └── domain_adaptation/
│       ├── data/           # Data loading and preprocessing
│       ├── models/         # Model architectures
│       ├── losses/         # Loss functions
│       ├── metrics/        # Evaluation metrics
│       ├── train/          # Training utilities
│       ├── eval/           # Evaluation utilities
│       ├── utils/          # Utility functions
│       └── viz/            # Visualization tools
├── configs/                # Configuration files
├── data/                   # Data storage
├── assets/                 # Generated assets
├── tests/                  # Unit tests
├── scripts/                # Training scripts
├── demo/                   # Interactive demos
└── notebooks/              # Jupyter notebooks
```

### Model Architectures

#### 1. Source-Only Baseline
- **Purpose**: Baseline trained only on source domain
- **Architecture**: CNN feature extractor + classifier
- **Use case**: Establishes performance floor

#### 2. Target-Only Baseline
- **Purpose**: Baseline trained only on target domain
- **Architecture**: CNN feature extractor + classifier
- **Use case**: Establishes performance ceiling

#### 3. Adversarial Domain Adaptation
- **Purpose**: Learn domain-invariant features
- **Architecture**: Feature extractor + label classifier + domain classifier
- **Key component**: Gradient reversal layer
- **Loss**: Classification loss + adversarial loss

#### 4. CORAL Domain Adaptation
- **Purpose**: Align feature distributions
- **Architecture**: Feature extractor + classifier
- **Key component**: CORAL loss for covariance alignment
- **Loss**: Classification loss + CORAL loss

## Methods Implemented

### Baselines
- **Source-Only**: Train only on source domain
- **Target-Only**: Train only on target domain

### Advanced Methods
- **Adversarial Training**: Domain adversarial neural networks (DANN)
- **CORAL**: Correlation alignment for domain adaptation
- **MMD**: Maximum Mean Discrepancy (available in losses)

### Evaluation Metrics
- **Classification**: Accuracy, F1-score, Precision, Recall
- **Domain Shift**: MMD distance, CORAL distance, Wasserstein distance
- **Adversarial**: Domain classification accuracy, confusion metrics

## Training Commands

### Basic Training

```bash
# Train with default configuration
python src/train.py

# Train with custom configuration
python src/train.py --config configs/custom.yaml --exp-name my_experiment

# Train with custom seed
python src/train.py --seed 123
```

### Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model parameters
model:
  feature_dim: 128
  dropout: 0.5

# Training parameters
training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.001

# Loss weights
loss:
  classification_weight: 1.0
  adversarial_weight: 1.0
  coral_weight: 0.1
```

## Expected Results

### Performance Ranges

| Method | Source Accuracy | Target Accuracy | Domain Shift |
|--------|----------------|-----------------|--------------|
| Source Only | 0.95-0.98 | 0.60-0.75 | High |
| Target Only | 0.50-0.70 | 0.90-0.95 | N/A |
| Adversarial | 0.85-0.95 | 0.75-0.85 | Medium |
| CORAL | 0.90-0.97 | 0.70-0.80 | Medium |

*Note: Results may vary based on hyperparameters and random seeds*

### Key Insights
- **Domain shift impact**: Significant performance drop without adaptation
- **Adversarial effectiveness**: Good target performance but may hurt source
- **CORAL balance**: Good compromise between source and target performance
- **Baseline importance**: Essential for understanding adaptation benefits

## Interactive Demo

The Streamlit demo provides:

1. **Model Comparison**: Train and compare different methods
2. **Domain Analysis**: Visualize domain shift characteristics
3. **Parameter Tuning**: Interactive hyperparameter adjustment
4. **Results Visualization**: Performance metrics and plots

### Demo Features
- Real-time model training simulation
- Interactive parameter adjustment
- Domain shift visualization
- Performance comparison charts
- Best practices and insights

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## API Reference

### Core Classes

#### `AdversarialDomainAdaptationModel`
```python
model = AdversarialDomainAdaptationModel(
    input_channels=1,
    feature_dim=128,
    num_classes=10,
    dropout=0.5,
    gradient_reversal_alpha=1.0
)
```

#### `DomainAdaptationTrainer`
```python
trainer = DomainAdaptationTrainer(
    model=model,
    device=device,
    learning_rate=0.001,
    weight_decay=1e-4
)
```

#### `DomainAdaptationLoss`
```python
loss_fn = DomainAdaptationLoss(
    classification_weight=1.0,
    adversarial_weight=1.0,
    coral_weight=0.1,
    mmd_weight=0.0
)
```

## Development

### Code Quality
- **Type hints**: Full type annotation coverage
- **Formatting**: Black + Ruff for code formatting
- **Linting**: MyPy for type checking
- **Testing**: Pytest with coverage reporting

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- **Author**: [kryptologyst](https://github.com/kryptologyst)
- **GitHub**: https://github.com/kryptologyst
- **Inspiration**: Domain adaptation research community
- **Datasets**: MNIST and USPS datasets

## References

1. Ganin, Y., et al. "Domain-adversarial training of neural networks." JMLR 2016.
2. Sun, B., et al. "Return of frustratingly easy domain adaptation." AAAI 2016.
3. Long, M., et al. "Learning transferable features with deep adaptation networks." ICML 2015.

## Limitations and Considerations

- **Dataset specific**: Results are specific to MNIST-USPS domain adaptation
- **Computational cost**: Adversarial training requires more resources
- **Hyperparameter sensitivity**: Performance depends on careful tuning
- **Evaluation scope**: Limited to digit classification task
- **Generalization**: May not apply to other domain pairs

---

**Disclaimer**: This project is for educational and research purposes only. The authors make no warranties about the accuracy, completeness, or suitability for any purpose. Use at your own risk.
# Domain-Adaptation-Techniques
