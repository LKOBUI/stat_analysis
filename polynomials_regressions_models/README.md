# Polynomial Regression Models & Splines

This directory focuses on extending linear regression to non-linear relationships using **Polynomial Regression Models**, **Scikit-Learn implementations**, custom regression modules, and advanced smoothing techniques like **Basis Splines (B-Splines)**.

---

## Directory Structure & Subdirectories

### 1. `DataSet/`
* **Purpose:** Contains the structured data files (such as `.csv` or `.xlsx`) used throughout the notebooks to train, validate, and test polynomial and spline-based models.

### 2. `PolynomialInPython/`
* **Purpose:** Contains foundational Jupyter Notebooks and pure Python implementations demonstrating the raw mechanics of building polynomial models from scratch using core libraries like `NumPy` and `SciPy`. 

### 3. `PolynomialsRegressionsModules/`
* **Purpose:** A dedicated module folder containing helper functions, custom scripts (`.py`), or modular class definitions designed to wrap repetitive polynomial regression workflows (e.g., automated degree selection, plotting residual trends).

- Implements polynomial regression using the `statsmodels` API.
- Covers:
  - Adding polynomial terms manually or via `PolynomialFeatures` from `sklearn`
  - Evaluating model summary (`R²`, `Adj. R²`, `p-values`)
  - Residual diagnostics and goodness‑of‑fit
- Example workflow:
  ```python
  import statsmodels.api as sm
  from sklearn.preprocessing import PolynomialFeatures

  poly = PolynomialFeatures(degree=3)
  X_poly = poly.fit_transform(X)
  model = sm.OLS(y, X_poly).fit()
  print(model.summary())```

### 4. `Scikit_Lern Plynomials/`
* **Purpose:** Notebooks demonstrating native workflows within the Scikit-Learn ecosystem. This highlights the use of `sklearn.preprocessing.PolynomialFeatures` chained together with `sklearn.linear_model.LinearRegression` inside an engine-driven `Pipeline`.


\[
  y = \beta_0 + \beta_1x + \beta_2x^2 + \dots + \beta_nx^n + \epsilon
  \]


- Compares linear vs. polynomial fits visually.
- Includes code for:
  - Data generation
  - Model fitting using `numpy.polyfit` and `statsmodels`
  - Plotting regression curves

---

## Root Notebooks & Files

* **`Basis_Spline.ipynb`**  
  An advanced study notebook implementing **Basis Splines (B-Splines)**. It demonstrates how to fit local polynomial segments joined smoothly at specific boundary points called **knots**, overcoming the global oscillation issues (Runge's phenomenon) often caused by high-degree global polynomials.
  
* **`PolynomialsRegressionsAnalysis_Nots.ipynb`**  
  A core theoretical and analytical notebook containing step-by-step conceptual walkthroughs, performance evaluation metrics ($R^2$, Adjusted $R^2$, MSE), and structural analysis regarding how polynomial model complexity scales.

* **`Commit.txt`**  
  A maintenance log file tracking internal adjustments, execution benchmarks, and versioning checkpoints for this chapter block.

---

## Core Statistical Concepts Covered

1. **Polynomial Models as Linear Regression Extensions:** Understanding that while the relationship between $X$ and $Y$ is non-linear, the model remains *linear in terms of its parameters ($\beta$)*:
   $$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \dots + \beta_k X^k + \epsilon$$
2. **The Overfitting vs. Underfitting Dilemma:** Testing lower vs. higher degree polynomials to balance the Bias-Variance tradeoff.
3. **Spline Regressions:** Segmenting data ranges using piecewise polynomials bound by smooth continuity constraints instead of a single global curve.

---

## Required Python Environment

Install the essential dependencies needed to run all the notebooks inside this directory tree:
```bash
pip install numpy pandas matplotlib scipy scikit-learn statsmodels
```
