# Deepfake Detection using Hybrid CNN–LightGBM Ensemble  

**Team:** AIML Enthusiasts  

---

## Overview  

This repository presents a **hybrid deepfake detection system** developed for the *Deepfake Detection Hackathon*.  
Our approach combines **CNN-based embeddings**, **handcrafted forensic features**, and a **LightGBM meta-learner**, followed by **logistic calibration** to achieve strong accuracy, robustness, and scalability.  

The complete workflow is implemented in a single Jupyter notebook:  
`DeepfakeMLChallenge_final.ipynb`

---

## Methodology Overview  

### 1. Data Preprocessing  
- Mounted Google Drive and verified dataset structure  
- Created `df_image_table.csv` for unified indexing and label access  
- Applied preprocessing pipeline:
  - **Resizing** and **RGB conversion**  
  - **FFT / DCT channel extraction** for frequency-domain representation  
  - **Normalization** for stable model convergence  

---

### 2. Feature Extraction  

| Type | Features | Description |
|------|-----------|-------------|
| **Handcrafted** | 15 | Sharpness, entropy, edge density, color statistics, ELA, FFT/DCT features |
| **CNN Embedding** | 256 | Learned deep texture and structural representations |
| **CNN OOF Score** | 1 | Meta-confidence signal from CNN predictions |

Each image was represented as a **272-dimensional vector**:  
```
X = [CNN Embedding (256) + Handcrafted Features (15) + CNN_OOF (1)]
```

This design fuses **deep**, **statistical**, and **confidence-level** cues, improving generalization and interpretability.

---

### 3. Model Training — LightGBM Stack  

A **LightGBM gradient boosting model** was trained on the fused feature space.

**Configuration:**
```python
objective = "binary"
metric = "auc"
learning_rate = 0.03
num_leaves = 127
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = 5
min_child_samples = 20
```

**Setup:**
- 5-Fold Stratified Cross-Validation  
- Early stopping = 100 rounds  
- Metric = AUC  

**Results:**
- Mean OOF AUC = **0.963 ± 0.006**  
- Consistent folds (0.968 – 0.983)  

---

### 4. Hyperparameter Optimization — Optuna  

To maximize generalization and minimize overfitting, we tuned LightGBM with **Optuna Bayesian optimization** (40 trials).

**Search Space:**
```
learning_rate ∈ [0.01, 0.1]
num_leaves ∈ [31, 255]
feature_fraction ∈ [0.6, 1.0]
bagging_fraction ∈ [0.6, 1.0]
min_child_samples ∈ [5, 100]
λ₁, λ₂ ∈ [0.0, 5.0]
```

**Best Trial (#38):**
```
AUC = 0.97596
learning_rate = 0.0806
num_leaves = 187
feature_fraction = 0.6736
bagging_fraction = 0.7115
bagging_freq = 4
min_child_samples = 75
lambda_l1 = 0.066
lambda_l2 = 0.855
```

Tuned model saved as `lgb_stack_tuned.pkl`

---

### 5. Calibrated Ensemble — CNN + LightGBM  

Even though LightGBM achieved high AUC, its output probabilities were slightly uncalibrated.  
We applied a **Logistic Regression calibration** on top of CNN and LGBM predictions.

**Steps:**
1. Align CNN and LGBM OOF predictions by image path  
2. Train Logistic Regression on `[cnn_oof, lgb_oof] → y`  
3. Generate calibrated predictions on test set  

**Result:**
- Calibrated AUC = **0.964**  

---

### 6. Robustness & Generalization  

- **5-Fold Stratified CV** prevents overfitting  
- **Test-Time Augmentation (TTA)** boosts robustness to spatial variations  
- **Feature diversity** (spatial + frequency) improves generalization to unseen generators  
- **No external pretrained weights**, ensuring reproducibility and fairness  

---

### 7. Final Submission  

**Output JSON Format:**
```json
[
  {"index": 1, "prediction": 0.0024},
  {"index": 2, "prediction": 0.9721},
  ...
]
```

**File:** `AIML_Enthusiasts_prediction.json`

---

## Performance Summary  

| Model | AUC | Key Strength |
|-------|-----|--------------|
| CNN | 0.922 | Learns local texture & spatial cues |
| LightGBM | 0.963 | Captures statistical & frequency-domain features |
| **Calibrated Ensemble** | **0.964** | Combines deep, handcrafted, and confidence-level cues |

---

## Challenges & Solutions  

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Data Leakage | Handcrafted CSV contained labels | Removed target/json_label columns before training |
| Limited Data | Risk of overfitting | Used 5-fold CV and augmentation |
| Probability Misalignment | Raw outputs not well-calibrated | Applied logistic regression calibration |
| Path Mismatch | Inconsistent index alignment | Implemented strict path alignment checks |

---

## Requirements  
```bash
pip install lightgbm optuna scikit-learn pandas numpy matplotlib seaborn
```

---

## Usage  

1. Open `DeepfakeMLChallenge_final.ipynb`  
2. Execute all cells sequentially  
3. Outputs will be generated in the working directory  

---

## License  

This project is submitted for the Deepfake Detection Hackathon.

---

## Acknowledgments  

Special thanks to the hackathon organizers and the open-source community for providing tools and datasets that made this work possible.
