# Regression Analysis with Applied Statistics and Detailed Mathematical Foundations

# Learning Goals & Architectural Summary: Simple and Multiple Regression Models

**Repository Target:** [LKOBUI/stat_analysis (Main Branch)](https://github.com/LKOBUI/stat_analysis/tree/main)  
**Target Directory Ecosystem:** `simple_and_multiple_regressions/` (Signal & Multiples Regressor Space)

## 1. Core Objectives & Granular Notebook Summaries

The foundational objective of this directory framework is to isolate and map the transition from single-signal statistical representations to multidimensional multi-regressor systems using Ordinary Least Squares (OLS) estimation. It establishes the baseline math required for inferential modeling before advanced transformations are implemented.

### A. Simple Linear Regression Pipeline (`simple_and_multiple_regressions/` Base Framework)
*   **Objective:** Mathematically isolate the deterministic and stochastic relationships between a single isolated continuous feature and a continuous dependent variable target.
*   **Detailed Analytical Summary:** This process computes structural baseline estimations by fitting a bivariate geometric plane over observed scatter distributions. It processes point estimations of error variances and evaluates the predictive confidence boundaries flanking the computed slopes. It validates initial single-feature correlation limits before complicating the predictor space.

### B. Multiple Linear Regression & Parameter Space Expansion
*   **Objective:** Scale the structural design framework from a simple two-dimensional matrix into a vector space handling $p$ simultaneous regressors.
*   **Detailed Analytical Summary:** This analytical module structures how adding features affects the behavior of remaining variables, isolating individual covariate weights while controlling for hidden mutual dependencies. It actively parses partial regression coefficients, resolving how shifting a target variable operates when secondary signal dimensions are fixed.

### C. Structural Model Adequacy & Diagnostic Verifications
*   **Objective:** Programmatically audit the OLS engine to verify compliance with fundamental Gauss-Markov and classical linear distribution constraints.
*   **Detailed Analytical Summary:** This phase runs localized diagnostic checks on error behavior across the regression plane. It evaluates normal Quantile-Quantile distributions to verify Gaussian error trends and maps residual behaviors alongside calculated target values to flag variance drops or structural deviations.

## 2. Comprehensive Mathematical Formulations & Explanations

The statistical computations engine relies entirely on matrix-algebraic formulations of Ordinary Least Squares (OLS).

### A. Matrix Layout of the Multidimensional Linear System
To scale to an arbitrary count of signal dimensions, the structural variables map into standard algebraic matrices:

$$y = X\beta + \epsilon$$

*   **Explanation:** 
    *   $y$ is an $n \times 1$ column matrix containing all observed target values.
    *   $X$ represents an $n \times (p+1)$ design matrix; its first vertical column is filled entirely with $1$s to calculate the baseline intercept $\beta_0$, while the remaining $p$ columns contain raw predictor data.
    *   $\beta$ represents a $(p+1) \times 1$ column matrix holding the unknown target parameters.
    *   $\epsilon$ maps an $n \times 1$ column vector tracking unobservable random errors.

### B. The OLS Normal Matrix Equations
The OLS framework minimizes the vertical spatial gaps across the data footprint by isolating the Residual Sum of Squares ($RSS$):

$$RSS(\beta) = \epsilon^T\epsilon = (y - X\beta)^T(y - X\beta)$$

Taking the partial derivative relative to $\beta$ and equating it to zero generates the **Normal Equations**:

$$X^TX\beta = X^Ty$$

Isolating the vector $\beta$ delivers the definitive parameter estimating solution used natively inside the repository notebooks:

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

*(Note: This calculation depends on the matrix $X^TX$ remaining non-singular and fully invertible, a requirement that directly links to the repository's dedicated section on multicollinearity).*

### C. The Adjusted Coefficient of Determination ($R^2_{adj}$)
Standard $R^2$ artificially grows when more signal elements are introduced, even if those elements contain random noise. To counter this, the notebooks track the penalized explanatory efficiency metric:

$$R^2_{adj} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$

*   **Explanation:** This metric adjusts standard scaling by factoring in the sample size ($n$) alongside the active regressor count ($p$). It penalizes the final accuracy score if newly added features do not supply enough predictive weight to offset the loss of degrees of freedom.

## 3. Analytical Modeling Profiling & Structural Rationales

The repository implements structural variants of linear estimators, balancing inferential validation tables with rapid prediction pipelines.

### A. Model Class: `statsmodels.regression.linear_model.OLS`
*   **Methodology Context:** Pure Gauss-Markov Inferential Estimator.
*   **Rationale for Selection:** Selected because it calculates comprehensive statistical summaries. It evaluates explicit Student's t-distributions, global Fisher-test F-statistics, and omnibus error bounds. This framework is prioritized for **inference, hypothesis verification, and coefficient validation** over raw, automated predictions.

### B. Model Class: `sklearn.linear_model.LinearRegression`
*   **Methodology Context:** Algorithmic Array-Based Prediction Engine.
*   **Rationale for Selection:** Selected for its clean object-oriented architecture and seamless integration with validation pipelines. It handles array operations efficiently, making it the ideal baseline model for **rapid cross-validation, iterative feature checks, and metric tracking**.

### Operational Justification Over Complex Machine Learning Alternatives:
1.  **Direct Structural Transparency:** Every element inside $\hat{\beta}$ translates to a direct, visible unit shift in the target space, avoiding the transparency issues common in black-box models.
2.  **Architectural Benchmark:** These models establish an absolute baseline error rate. The repository uses OLS to demonstrate exactly when an analyst must upgrade to specialized tools (like Ridge/Lasso regularizations, Generalized Linear Models, or Robust IRLS) when basic structural assumptions fail.

---
