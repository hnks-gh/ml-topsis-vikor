# ML-MCDM

Machine Learning enhanced Multi-Criteria Decision Making for panel data analysis.

## Overview

This framework integrates **MCDM methods** with **Machine Learning** to rank entities (provinces, companies, etc.) across multiple time periods and criteria. It combines traditional decision-making techniques with predictive modeling for comprehensive analysis.

### Technical Approach

```
Panel Data (entities × time × criteria)
         │
         ├─► Weight Calculation (Entropy, CRITIC, Ensemble)
         │
         ├─► MCDM Ranking
         │     ├─ Traditional: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS
         │     └─ Fuzzy: Handles uncertainty via triangular fuzzy numbers
         │
         ├─► ML Analysis
         │     ├─ Panel Regression (Fixed/Random Effects)
         │     ├─ Random Forest with Time-Series CV
         │     ├─ Gradient Boosting Ensemble
         │     └─ Neural Networks with Attention
         │
         ├─► Ensemble Integration (Stacking, Borda, Copeland)
         │
         └─► Analysis (Convergence, Sensitivity, Validation)
```

## Project Structure

```
ml-topsis-vikor/
├── run.py                  # Entry point
├── pyproject.toml          # Package config
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration dataclasses
│   ├── pipeline.py         # Main orchestrator
│   ├── data_loader.py      # Panel data I/O
│   ├── output_manager.py   # Results export
│   ├── visualization.py    # Charts (300 DPI)
│   ├── logger.py
│   │
│   ├── mcdm/               # Decision methods
│   │   ├── weights.py      # Entropy, CRITIC, Ensemble
│   │   ├── topsis.py       # Distance to ideal solution
│   │   ├── vikor.py        # Compromise ranking
│   │   ├── promethee.py    # Outranking flows
│   │   ├── copras.py       # Proportional assessment
│   │   ├── edas.py         # Distance from average
│   │   ├── fuzzy_base.py   # Triangular fuzzy numbers
│   │   └── fuzzy_*.py      # Fuzzy variants
│   │
│   ├── ml/                 # Machine learning
│   │   ├── panel_regression.py     # FE/RE/Pooled OLS
│   │   ├── random_forest_ts.py     # RF with temporal CV
│   │   ├── advanced_forecasting.py # Gradient boosting
│   │   ├── neural_forecasting.py   # MLP + Attention
│   │   ├── unified_forecasting.py  # Ensemble orchestrator
│   │   ├── lstm_forecast.py
│   │   └── rough_sets.py           # Feature reduction
│   │
│   ├── ensemble/           # Aggregation
│   │   ├── stacking.py     # Meta-learner
│   │   └── aggregation.py  # Borda, Copeland
│   │
│   └── analysis/           # Validation
│       ├── convergence.py  # Beta/Sigma convergence
│       ├── sensitivity.py  # Weight perturbation
│       └── validation.py   # Bootstrap, CV
│
├── data/
│   └── data.csv            # Input: Year, Province, C01-C20
│
├── outputs/
│   ├── figures/            # PNG charts
│   ├── results/            # CSV files
│   └── reports/            # Analysis reports
│
├── tests/
│   └── test_core.py
│
└── docs/
    ├── METHODS.md          # Mathematical formulations
    └── WORKFLOW.md         # Pipeline phases
```

## Methods

| Category | Methods |
|----------|---------|
| **Weighting** | Entropy (information content), CRITIC (contrast + correlation), Ensemble |
| **MCDM** | TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS — each with fuzzy variant |
| **ML** | Panel Regression, Random Forest, Gradient Boosting, Neural Networks |
| **Ensemble** | Stacking meta-learner, Borda Count, Copeland |
| **Analysis** | β/σ convergence, Monte Carlo sensitivity, Bootstrap validation |

## Usage

```bash
python run.py                  # Default data
python run.py data/custom.csv  # Custom data
```

```python
from src import MLTOPSISPipeline, get_default_config

config = get_default_config()
pipeline = MLTOPSISPipeline(config)
result = pipeline.run('data/data.csv')

print(result.get_final_ranking_df().head(10))
```

## Input Format

```csv
Year,Province,C01,C02,...,C20
2020,P01,0.75,0.82,...,0.71
2020,P02,0.62,0.74,...,0.65
2021,P01,0.78,0.85,...,0.73
```

## Output

| Directory | Contents |
|-----------|----------|
| `results/` | Rankings, weights, scores (CSV) |
| `figures/` | Visualizations (300 DPI PNG) |
| `reports/` | Analysis summary |

## Requirements

Python 3.8+ with numpy, pandas, scipy, scikit-learn, matplotlib, seaborn.

```bash
pip install -e .
```

## License

MIT
