# NeoGx Level IV Model Metrics

Machine learning pipeline for genetic diagnostics prediction in neonatal cohorts using HPO and phecode vocabularies.

## Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone this repository
2. Install dependencies (pip install or create a virtual environment):

## Environment Setup

#### Option 1 (Recommended): uv
##### macOS/Linux
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
##### Windows (PowerShell)
```PowerShell
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

#### Option 2: standard Python venv
##### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

##### Windows (PowerShell)
```PowerShell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the data and assets from Zenodo (run from repo root):
```bash
aws s3 sync s3://nch-ods-data-science/projects/neogx/neogx-iv-manuscript-data/ .
```

4. The expected directory structure is:
```
.
├── src/
│   ├── analysis.ipynb
│   ├── ml_pipeline.py
│   ├── utils_training.py
│   └── utils_analysis.py
├── data/
│   ├── neogx-cohort-dataframe-for-metrics.csv.gz
│   ├── feature_matrices/
│   │   ├── static_features.csv.gz
│   │   └── pheno_features-*.csv.gz
│   └── search-results.csv.gz
├── assets/
│   ├── phecodes/
│   │   └── phecodeX_R_labels.csv
│   ├── hpo-v2025-10-22/
│   │   └── (HPO ontology files)
│   └── hpo_representative_map-*.json
├── classifiers/        (written by notebook: neogx_classifier-*.pkl, model_info-*.json)
└── manuscript-figures/ (written by notebook: Figure-*.{pdf,svg,json}, SHAP-summary.xlsx)
```

## Usage

Run the main analysis notebook from inside `src/` so its `../data` and `../assets` relative paths resolve correctly:
```bash
cd src && jupyter notebook "analysis.ipynb"
```

## Project Structure

- `src/analysis.ipynb` - Main analysis notebook
- `src/ml_pipeline.py` - Machine learning pipeline definitions (sklearn `Pipeline`, classifier registry, scorers)
- `src/run_random_search.py` - CLI for running `RandomizedSearchCV` to produce grid CSVs
- `src/utils_training.py` - Feature matrix building and grid-result config round-tripping
- `src/utils_analysis.py` - Metrics, threshold/optimization, and figure functions
- `src/feature_preprocessing.py` - Feature engineering helpers (used to build feature matrices upstream)
- `src/hpo_utils.py` - HPO ontology helpers

## Data Files

### Required Input Files
- `data/cohort-dataframe-calibration-validation.csv.gz` - Main cohort data with patient demographics, outcome labels, and prediction-time covariates
- `cohort-split.csv` - Development/Calibration/Validation cohort membership
- `data/feature_matrices/static_features.csv.gz` - Static clinical features (sex, GA, birth-weight Z, etc.)
- `data/feature_matrices/pheno_features-*.csv.gz` - Phenotype matrices for each (vocab, encoding, level4_day) combination
- `data/search-results.csv.gz` - Hyperparameter grid search results
- `assets/phecodes/phecodeX_R_labels.csv` - Phecode labels and descriptions
- `assets/hpo-v2025-10-22/` - HPO ontology files
- `assets/hpo_representative_map-*.json` - HPO representative maps for feature aggregation

### Output Files (written by the notebook)
- `classifiers/neogx_classifier-*.pkl` - Trained classifier pipelines
- `classifiers/model_info-*.json` - Feature names, HPO match map, calibrated probability thresholds
- `manuscript-figures/Figure-*.pdf` - Generated figures for manuscript
- `manuscript-figures/Figure-*.svg` - Vector versions of figures
- `manuscript-figures/Figure-*.json` - Plotly figure data
- `manuscript-figures/SHAP-summary.xlsx` - SHAP value summary table

## Notes

- Large data files (`*.csv`, `*.csv.gz`, `*.pkl`), and the entire `data/`, `assets/`, `classifiers/`, and `manuscript-figures/` directories are excluded from git via `.gitignore`
- All runtime paths in the notebook are relative to `src/` (e.g. `../data/...`, `../assets/...`); always launch jupyter from inside `src/`
