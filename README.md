# Explainable Fuzzy Clustering on Iris (TSK-inspired, Python)

This microproject implements an **explainable clustering** pipeline inspired by the paper *"Takagi–Sugeno–Kang Fuzzy Clustering by Direct Fuzzy Inference on Fuzzy Rules"*.
It provides **two runnable modes**:

1. **CLI (no GUI)** – run clustering, print metrics, and save plots.
2. **Streamlit GUI** – interactive app to explore parameters and view rules.

> Note: This is a **pedagogical, simplified implementation** that *adapts* the paper’s ideas for student projects.
> We form **interpretable fuzzy rules** as IF-THEN structures (linguistic partitions + linear consequents)
> and cluster in the **fuzzy-inferred feature space** `xg`. We then explain clusters by the **rules with highest average firing per cluster**.

## Quickstart

```bash
# (Recommended) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 1) Run non-GUI version
python run_cli.py

# 2) Run GUI (Streamlit)
streamlit run app_streamlit.py
```

Artifacts are saved to `outputs/`.

## Files

- `tsk_fc_iris.py` – core library: fuzzy partitions, rule base, fuzzy inference embedding, clustering, explanations
- `run_cli.py` – command-line entry (no GUI)
- `app_streamlit.py` – Streamlit GUI
- `requirements.txt` – dependencies
- `outputs/` – metrics, rule tables, and plots

## Method (Simplified)

1. **Linguistic partitions**: Each feature gets `M` Gaussian membership functions with terms like {Very Low, Low, ..., Very High}.
2. **Rule base**: Randomly sample `R` rules by choosing a term per feature.
3. **Fuzzy inference embedding**: For each sample `x`, compute normalized rule firing `μ̃_r(x)`, build `xg = concat_r μ̃_r(x) * [1, x]` (first-order TSK-like vector).
4. **Clustering in xg**: Run k-means (K clusters) on `xg`.
5. **Explanation**: For each cluster, rank rules by average firing of cluster members. Display top rules in linguistic form.
6. **Evaluation**: Report ACC (via Hungarian mapping) and NMI.

This captures the **rule-driven interpretability** and **feature-space lift** from the paper, while keeping code concise for a mini-project.

## Citations

- Gu et al., *Takagi–Sugeno–Kang Fuzzy Clustering by Direct Fuzzy Inference on Fuzzy Rules*, IEEE TETCI, 2024.
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 2011.

