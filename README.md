# CUDA Error Resolution Analysis

> Comprehensive causal analysis of CUDA-related error resolution patterns on PyTorch discussion forums using advanced statistical and machine learning methods.

## Research Questions

1. **Descriptive**: What are the characteristics of CUDA-related forum topics vs non-CUDA topics?
2. **Predictive**: Can we predict resolution time based on topic features (code blocks, error traces, views)?
3. **Causal**: Does CUDA-related content causally affect time to resolution?

## Project Structure

```
root@ADA_Project > tree 

ADA_Project/
├── src/
│   ├── scraper/
│   │   ├── forum_scraper.py      # PyTorch forum scraper (REST API)
│   │   └── feature_engineer.py   # JSON → CSV feature extraction
│   └── analysis/
│       ├── descriptive.py        # Summary stats, correlation, ANOVA
│       ├── predictive.py         # ML models for resolution time
│       └── causal.py             # PSM, S/T/X-learners, Double ML
├── notebooks/                    # Jupyter analysis workflows
├── dashboard/                    # Streamlit interactive dashboard
└── data/
    ├── raw/                     # Scraped JSON batches
    └── processed/               # Cleaned CSV + figures
```

## Quick Start

### 1. Setup Environment

```bash
# Clone and enter project
cd ADA_Project

# Run setup script (installs UV + dependencies)
./setup.sh

# Activate environment
source .venv/bin/activate
```

### 2. Collect Data (~24-48 hours)

```bash
# Test first (5 pages, ~15 min)
python src/scraper/forum_scraper.py --max-pages 5

# Full scrape (recommended overnight)
python src/scraper/forum_scraper.py
# → Creates topics_batch_*.json in data/raw/

# Process to CSV (~30-60 min)
python src/scraper/feature_engineer.py
# → Creates forum_data.csv in data/processed/
```

### 3. Run Analysis

**Option A: Jupyter Notebooks (Recommended)**
```bash
jupyter notebook
# Open: 01_descriptive_analysis.ipynb
#       02_predictive_models.ipynb
#       03_causal_inference.ipynb
```

**Option B: Python Scripts**
```bash
python src/analysis/descriptive.py
python src/analysis/predictive.py
python src/analysis/causal.py
```

**Option C: Interactive Dashboard**
```bash
streamlit run dashboard/app.py
# → Opens http://localhost:8501
```

## � Data Pipeline

```
PyTorch Forum API
       ↓
forum_scraper.py (24-48 hrs)
       ↓
topics_batch_*.json (~500 MB, 80K topics)
       ↓
feature_engineer.py (30-60 min)
       ↓
forum_data.csv (~50 MB, 15+ features)
       ↓
Analysis (descriptive/predictive/causal)
```

## Analysis Methods

### Descriptive Analysis
- Summary statistics (mean, median, quartiles)
- Correlation heatmaps
- ANOVA (category effects)
- Distribution plots (histograms, box plots)

### Predictive Models
- Simple Linear Regression
- Multiple Regression (OLS)
- Polynomial Features
- Lasso Regularization
- Cross-validation & feature importance

### Causal Inference
- Naive Comparison (baseline)
- Propensity Score Matching (PSM)
- S-Learner (single model)
- T-Learner (two models)
- X-Learner (advanced)
- Double ML (debiased)
- Sensitivity Analysis (hidden confounding)

## Key Dependencies

- **Data**: `pandas`, `numpy`, `beautifulsoup4`, `requests`
- **ML**: `scikit-learn`, `econml` (causal inference)
- **Viz**: `matplotlib`, `seaborn`, `plotly`
- **App**: `streamlit`, `jupyter`

## Expected Results

- **80,000+ forum topics** collected
- **~10-15% CUDA-related** topics
- **Resolution time** analysis (CUDA vs non-CUDA)
- **Treatment effect** estimation with confidence intervals
- **Interactive dashboard** for exploration

## License

MIT License - See [LICENSE](LICENSE) file

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

**Built with:** Python 3.11 • UV Package Manager • PyTorch Forum Data
