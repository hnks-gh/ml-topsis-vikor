# ML-MCDM: Panel Data Multi-Criteria Decision Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready framework combining **Machine Learning** and **TOPSIS** methods for comprehensive panel data analysis and sustainability assessment.

## Overview

ML-MCDM provides an integrated econometric-ML hybrid approach for multi-criteria decision making with panel data. It combines:

- **Multiple MCDM Methods**: TOPSIS, Dynamic TOPSIS, VIKOR, Fuzzy TOPSIS
- **Machine Learning**: Panel Regression, Random Forest, LSTM Forecasting, Rough Sets
- **Ensemble Methods**: Stacking meta-learning, Borda/Copeland rank aggregation
- **Advanced Analysis**: Convergence analysis, Sensitivity analysis, Cross-validation

## Features

- ✅ Panel data support (entities × time periods × criteria)
- ✅ Multiple weighting methods (Entropy, CRITIC, Ensemble)
- ✅ Four MCDM methods with rank aggregation
- ✅ ML-based validation and feature importance
- ✅ Time-series forecasting with LSTM
- ✅ Comprehensive visualization suite
- ✅ Automated report generation
- ✅ Production-ready with full error handling

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-repo/ML-MCDM.git
cd ML-MCDM

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Project Structure

```
ML-MCDM/
├── main.py                # Run this file to start analysis
├── pyproject.toml         # Package configuration
├── requirements.txt       # Dependencies
├── README.md             # Documentation
│
├── src/                  # Source code
│   ├── main.py           # Pipeline orchestrator
│   ├── config.py         # Configuration
│   ├── data_loader.py    # Data handling
│   ├── visualization.py  # Visualizations
│   ├── mcdm/             # MCDM methods
│   ├── ml/               # ML methods
│   ├── ensemble/         # Ensemble methods
│   └── analysis/         # Analysis tools
│
├── data/                 # Data files
├── docs/                 # Documentation
├── tests/                # Test suite
└── outputs/              # Generated outputs
```

## Usage

### Quick Start

Simply run `main.py`:

```bash
python main.py
```

To use your own data file:

```bash
python main.py data/panel_data.csv
```

### Configuration

Edit the `CONFIG` dictionary at the top of `main.py`:

```python
CONFIG = {
    'data_path': None,           # Set to CSV path or None for synthetic
    'n_provinces': 64,           # Number of entities
    'n_years': 5,                # Number of time periods  
    'n_components': 20,          # Number of criteria
    'output_dir': 'outputs',     # Output directory
}
```

### Python API

```python
from src import MLTOPSISPipeline, get_default_config

# Quick run with defaults
from src import run_pipeline
result = run_pipeline()

# Or with custom configuration
config = get_default_config()
config.panel.n_provinces = 30
config.panel.years = [2020, 2021, 2022, 2023, 2024]

pipeline = MLTOPSISPipeline(config)
result = pipeline.run('data/panel_data.csv')

# Access results
print(f"Top entity: {result.panel_data.entities[result.topsis_rankings.argmin()]}")
print(f"Kendall's W: {result.aggregated_ranking.kendall_w:.4f}")
print(f"Execution time: {result.execution_time:.2f}s")

# Get ranking DataFrame
rankings = result.get_final_ranking_df()
print(rankings.head(10))
```

## Input Data Format

Panel data CSV with the following structure:

```csv
Year,Province,C01,C02,C03,...,C20
2020,P01,0.75,0.82,0.68,...,0.71
2020,P02,0.62,0.74,0.81,...,0.65
2021,P01,0.78,0.85,0.70,...,0.73
...
```

- `Year`: Time period identifier
- `Province`: Entity identifier
- `C01-C20`: Criteria/component values (normalized 0-1 recommended)

## Output

### Generated Files

| File | Description |
|------|-------------|
| `final_ranking.csv` | Entity rankings with scores |
| `weights.csv` | Criterion weights (Entropy, CRITIC, Ensemble) |
| `feature_importance.csv` | Random Forest feature importance |
| `analysis_report.txt` | Summary report |

### Visualizations

- Score evolution over time
- Method comparison plots
- Convergence analysis
- Weight sensitivity
- Ensemble weights

## Methodology

### Pipeline Phases

1. **Data Loading**: Load/generate panel data, create views
2. **Weight Calculation**: Entropy, CRITIC, and ensemble weights
3. **MCDM Analysis**: TOPSIS, Dynamic TOPSIS, VIKOR, Fuzzy TOPSIS
4. **ML Analysis**: Panel regression, Random Forest, LSTM, Rough Sets
5. **Ensemble Integration**: Stacking and rank aggregation
6. **Advanced Analysis**: Convergence and sensitivity analysis
7. **Visualization**: Generate all figures and reports

### Key Methods

| Component | Methods |
|-----------|---------|
| Weighting | Entropy, CRITIC, Ensemble (50/50) |
| MCDM | TOPSIS, Dynamic TOPSIS, VIKOR, Fuzzy TOPSIS |
| ML | Panel FE/RE, Random Forest TS-CV, LSTM, Rough Sets |
| Ensemble | Ridge Stacking, Borda Count, Copeland |
| Analysis | Beta Convergence, Monte Carlo Sensitivity |

## Configuration

Key settings in `src/config.py`:

```python
# Panel data
n_provinces = 64
n_components = 20
years = [2020, 2021, 2022, 2023, 2024]

# TOPSIS
normalization = "vector"  # or "minmax"
temporal_discount = 0.95

# Random Forest
n_estimators = 200
n_splits = 2  # Time-series CV folds

# Validation
n_simulations = 1000  # Sensitivity Monte Carlo
```

## Requirements

Core dependencies:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

Optional:
- shap >= 0.40.0 (for explainability)
- factor-analyzer >= 0.4.0 (for factor analysis)

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ml_topsis,
  title={ML-MCDM: Panel Data Multi-Criteria Decision Analysis Framework},
  author={ML-MCDM Development Team},
  year={2024},
  url={https://github.com/your-repo/ML-MCDM}
}
```
