🚀 Project Overview

**BioFusion-Stability** is an advanced multi-task deep learning system for predicting protein stability changes (ΔΔG) and classifying the effects of single-point mutations (stabilizing, neutral, destabilizing). This project integrates:

- **Multi-task MLP architecture** for regression (ΔΔG) and classification (mutation effect)
- **Non-PLM engineered features**: physicochemical, structural, and experimental
- **Explainable AI** via SHAP for global and local feature interpretation
- **Interactive Gradio frontend** for intuitive research and bioinformatics workflows

The system is designed for researchers in **drug discovery, protein engineering, and computational biology**, offering fast, interpretable, and deployable insights.

---

🎯 Motivation

Protein stability is a fundamental determinant of function and disease relevance. Single-point mutations can drastically alter stability, making accurate ΔΔG prediction crucial for:

- Guiding experimental designs
- Accelerating therapeutic development
- Understanding mutation-driven disease mechanisms

Traditional PLM embeddings are computationally expensive. **BioFusion-Stability** leverages non-PLM features for efficiency without sacrificing predictive quality.

---

🛠 Methodology

1. **Data Acquisition & Preprocessing**
   - Dataset: `thermomutdb.json`
   - Features: 30+ numeric + categorical descriptors (e.g., `blosum62`, `rsa`, `phi`, `psi`)
   - Missing value imputation and consistent label alignment

2. **Feature Engineering**
   - Numerical scaling with `StandardScaler`
   - One-hot encoding for categorical variables
   - Efficient ColumnTransformer pipeline for reproducible preprocessing

3. **Model Architecture**
   - Shared hidden layers: 256 → 128, ReLU + Dropout
   - Regression head: predicts ΔΔG
   - Classification head: predicts mutation effect
   - Optimized via 5-fold Group K-Fold CV

4. **Explainability**
   - SHAP (GradientExplainer) for local & global insights
   - Visualization of top features contributing to predictions

5. **Deployment**
   - Gradio interface with tabbed layout for predictions, SHAP insights, and structural summaries
   - Ready for Hugging Face Spaces deployment

---

📈 Results

- **Regression (ΔΔG):** MAE ≈ 1.175 kcal/mol, RMSE ≈ 1.758 kcal/mol, R² ≈ 0.275
- **Classification:** Accuracy ≈ 0.609, F1-macro ≈ 0.536
- SHAP analysis highlights `blosum62`, `rsa`, `res_depth`, `phi`, `psi`, and `temperature` as top contributors

The multi-task approach balances predictive accuracy with interpretability, offering a robust tool for bioinformatics applications.

---
