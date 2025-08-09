# smFRET HMM Analysis — Reproducible Notebook

This repository contains a Jupyter Notebook performing a full **single‑molecule FRET (smFRET)** analysis workflow with **Hidden Markov Models (HMMs)**. The notebook is annotated cell‑by‑cell for clarity and preserves the original execution order.

## Pipeline Overview

1. **I/O & Preprocessing**
   - Batch load raw trace files (`*.txt`) from disk
   - Optionally append **file indicators** and combine traces
   - Filter / clean combined arrays

2. **Modeling**
   - Configure and run **Gaussian HMM** decoding (user‑settable `n_components`)
   - Extract hidden states, dwell times, and state statistics

3. **Visualization**
   - Matplotlib/Seaborn figures: histograms, KDE, scatter/heatmaps
   - Figure export hooks

4. **Aggregation & Export**
   - Summaries (per‑file and global) with `tabulate`/`pandas`
   - Excel/CSV exports of results and summaries

## Files

- **`HMM_modelling_for_smFRET_traces_analysis_ANNOTATED.ipynb`** — main, fully annotated notebook (this is the primary entry point).
- **`HMM_modelling_for_smFRET_traces_analysis_ANNOTATED.py`** — script export for users who prefer plain Python.
- **`requirements.txt`** — Python dependencies.

## Data Expectations

- Input traces are assumed to be **tabular text** (e.g., `.txt`/`.csv`) with time and FRET columns (names vary; adjust in the **🔧 USER INPUT REQUIRED** cells).
- Batch processing uses wildcards like `*.txt` (editable in the input cells).
- Output summaries and figures are written to `output_directory` (also flagged).

## Usage

1. **Install dependencies** (ideally in a fresh virtual environment):
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Open the notebook** and edit parameters flagged as **🔧 USER INPUT REQUIRED** (e.g., `input_directory`, `output_directory`, file patterns, and HMM `n_components`).

3. **Run cells in order** from top to bottom. Figures and exports will appear as specified in the code comments.

## Technical Notes

- HMM decoding is based on `hmmlearn`'s `GaussianHMM` (full covariance by default in many examples). Convergence parameters and `n_components` are exposed in code comments for reproducibility.
- For robust dwell‑time statistics, ensure sampling interval (`dt`) and filtering thresholding are consistent across datasets.
- If you use **Seaborn** for KDE/heatmaps, ensure version compatibility with your **pandas**/**matplotlib** installs.

## Reproducibility & Provenance

- Each code cell begins with a banner indicating purpose and potential user‑editable parameters.
- Markdown cells before each code cell summarize expected **inputs/outputs**.
- The `.py` export mirrors the notebook structure with Markdown headers converted to comments.

## Citation

If this notebook supports a manuscript, please cite your repository DOI (e.g., Zenodo) and include software versions from `pip freeze` in your supplement.
