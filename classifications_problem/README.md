# Classifications Problems

**Repository Link:** [LKOBUI/stat_analysis](https://github.com/LKOBUI/stat_analysis/tree/main)
**Target Directory:** `classifications_problem`

### Repository Learning Objectives
* Establish foundational baseline metrics for classifying non-continuous target distributions.
* Understand the statistical boundary differences between linear regression predictions and discrete target spaces.
* Master early diagnostic data structures required to convert continuous predictors into probability vectors.

### Detailed Notebook Breakdowns

#### 1. Baseline Classification Mapping (`classification_baseline.ipynb`)
* **Objective:** Define baseline performance benchmarks for qualitative decision boundaries across simple coordinate systems.
* **Model in Use & Justification:** **Linear Probability Model (LPM)**. This model maps binary classification boundaries using standard Ordinary Least Squares (OLS). It is implemented as a historical and mathematical baseline to explicitly illustrate the flaws of using flat linear regressions for classification, such as producing out-of-bound predictions (less than 0 or greater than 1) and violating the assumption of homoscedastic error distributions.
* **Mathematical Formulation:**
  The baseline utilizes a standard additive predictor plane to estimate the conditional probability of an event:
  $$P(Y=1|X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k$$
  The model minimizes the vertical spatial gaps directly across binary states. Because the true outcome $Y$ is restricted to 0 or 1, the calculated error variance is heteroscedastic and scales predictably based on the predictor coordinates:
  $$\text{Var}(\epsilon|X) = P(X)[1 - P(X)]$$

#### 2. Decision Threshold Analysis (`threshold_diagnostics.ipynb`)
* **Objective:** Evaluate how altering decision thresholds impacts classification precision and the distribution of discrete error terms.
* **Model in Use & Justification:** **Threshold-Bounded Classifier Engine**. This framework translates continuous prediction planes into explicit, discrete classification labels based on customizable cutoff points. It is selected to study the direct balance between false-positive rates and true-negative rates across irregular data distributions.
* **Mathematical Formulation:**
  Continuous predicted values are mapped to a binary decision space using a indicator function bounded by a threshold parameter $\tau$:
  $$\hat{Y} = \mathbb{I}(\hat{y} \ge \tau) = \begin{cases} 1 & \text{if } \hat{y} \ge \tau \\ 0 & \text{if } \hat{y} < \tau \end{cases}$$
  Altering $\tau$ directly shifts model performance, mapping the path used to build diagnostic curves like Receiver Operating Characteristics (ROC).

#### 3. Qualitative Group Variance Review (`group_variance_study.ipynb`)
* **Objective:** Analyze cross-group variance behavior to understand where linear classification separations drop in efficiency.
* **Model in Use & Justification:** **Pooled Variance Multi-Group Estimator**. This tool isolates and tracks variance differences across distinct categorical target subsets. It is used to identify when group-level variances are unequal (heteroscedasticity), which violates the classical assumptions needed for linear classification tools.
* **Mathematical Formulation:**
  The model computes the shared spatial variance across separate qualitative classes to find an optimized, pooled baseline:
  $$S_p^2 = \frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2 + \dots + (n_g - 1)S_g^2}{\sum (n_i - 1)}$$
  Where $S_i^2$ represents the individual sample variance tracking specific categories, and $n_i$ monitors group counts. Large differences between $S_i^2$ values indicate that basic linear decision boundaries may struggle to separate the groups accurately.

### Performance Metrics Captured
* **Empirical Error Rates:** Counts the raw percentage of misclassified observations relative to the overall sample scale.
* **Sensitivity Index:** Measures the model's ability to correctly identify positive instances across shifting decision thresholds.
* **Residual Sum Bounds:** Tracks the behavior of non-Gaussian errors along the classification boundaries.
