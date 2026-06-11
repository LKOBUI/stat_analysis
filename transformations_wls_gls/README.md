# Transformations and Weighting to Correct Model Assumptions

This repo focuses on techniques used when standard Ordinary Least Squares (OLS) regression assumptions are violated, specifically addressing **non-constant variance (heteroscedasticity)** and **non-linear structural relationships**.

---

### Primary Goals of Transformations & Weighting

When fitting linear regressions, OLS assumes that errors have a constant variance ($\sigma^2$) and are normally distributed. When these assumptions fail, transformations and alternative weighting strategies serve two critical goals:

1. **Stabilizing Error Variance (Homoscedasticity):** If the spread of the residuals grows or shrinks alongside your predictions, applying a variance-stabilizing transformation or assigning weights forces the error variance back to a constant state. This ensures that your standard errors, hypothesis tests (t-tests, F-tests), and confidence intervals are mathematically valid.

2. **Linearizing the Relationship:** When the true physical process connecting the independent variable ($X$) and dependent variable ($Y$) is inherently curved (e.g., exponential growth or power functions), mathematical transformations morph the data configuration into a straight-line space. This allows you to fit a highly accurate linear model without forcing unnecessary global polynomial complexity.

---

### Overview of Key Methodologies

* **Box-Cox Transformation:** An automated power transformation pipeline targeting the dependent variable ($Y$) to fix non-normality and heteroscedasticity simultaneously.

* **Weighted Least Squares (WLS):** A modification where each data point is weighted inversely by its error variance ($w_i = 1/\sigma_i^2$), prioritizing stable points and penalizing noisy ones.

* **Generalized Least Squares (GLS):** A broader framework handling situations where errors are not only heteroscedastic but also correlated with each other (autocorrelation).

## Detailed File Profiles (`.ipynb` Extensions)

### Master Concepts & Theory

* **`TRANSFORMATIONS AND WEIGHTING TO CORRECT Notes.ipynb`**: The core foundational notebook of the entire chapter. It outlines the mathematical theory behind structural transformations, explains variance stabilization proof structures, and guides you through choosing the proper corrective strategy.

* **`GLS And WLS Note.ipynb`**: A specialized theoretical notebook defining the matrix algebra behind Generalized Least Squares (GLS) and Weighted Least Squares (WLS). It provides a bridge between basic scalar adjustments and advanced multi-dimensional matrix operations.

---

### Box-Cox Power Transformation Frameworks

* **`Box-Cox Transformation Nots.ipynb`**: Focuses on the mathematical derivation of the Box-Cox parameter $\lambda$. It details how maximum likelihood estimation (MLE) is deployed to find the optimal power transformation to normalize skewed $Y$ distributions.

* **`Box_COx.ipynb`**: A functional baseline notebook executing standard Box-Cox pipelines on generic datasets using libraries like SciPy (`scipy.stats.boxcox`).

---

### Real-World Box-Cox Case Studies

* **`Box-Cox_ElectricUtility.ipynb`**: A practical energy-sector case study. It applies power transformations to highly volatile electrical consumption and utility load demand data to stabilize variance patterns over shifting usage peaks.

* **`BoxCoxWindMilData.ipynb`**: An engineering regression application analyzing windmill performance data. It models aerodynamic output variables, using Box-Cox to flatten the non-linear forces inherent to wind velocity tracking.

---

### Weighting & Non-Constant Variance Solutions

* **`WLS.ipynb`**: Implements Weighted Least Squares (WLS). It teaches you how to define custom weight vectors when variance changes systematically with the size of an independent regressor.

* **`GLS.ipynb`**: Implements **Generalized Least Squares (GLS)**. It demonstrates model building under conditions where the error terms show systemic clustering or tracking correlation.

* **`Coverience Matrix.ipynb`**: A focused mathematical environment dedicated to constructing and analyzing structural **Error Covariance Matrices ($\Omega$)**. This matrix acts as the foundational heart needed to configure accurate GLS pipelines.

* **`HC_se_tutorials.ipynb`**: A crucial notebook exploring **Heteroscedasticity-Consistent (HC) Standard Errors** (specifically targeting variants like HC0 through HC3). It demonstrates how to calculate robust standard errors that remain valid during variance distortion without needing to transform the raw data.
