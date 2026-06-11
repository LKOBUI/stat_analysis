# stat_analysis

Here is a comprehensive directory breakdown and structural tree for the LKOBUI/stat_analysis repository. 
This repository focuses on the mathematical foundations, diagnostic tools, and practical implementations 
of regression modeling and statistical analysis. This repository is dedicated to exploring the mathematical 
foundations of regression analysis across multiple domains:

* Linear regression
* Polynomial regression
* Multicollinearity handling
* Generalized Linear Models (GLM)
* Robust regression techniques

Each directory and subdirectory contains Jupyter Notebooks (`.ipynb`) or scripts (`.py`) with detailed explanations of the statistical methods, their mathematical derivations, and applied examples.

# Directory Structure
stat_analysis/

│

├── polynomials_regressions_models/

│   ├── B_SPLINE_kagalDataset.ipynb

│   ├── B_SPLINE_uswages.ipynb

│   ├── HardwoodData.ipynb

│   ├── LINEAR INTERPOLATION.ipynb

│   ├── SplinesPiecewisePolynomial.ipynb

│   ├── dmatrix_spline.ipynb

│   ├── polynomial_scikitlern.ipynb

│   ├── polynomial.ipynb

│   ├── FwSelectionBwElimination.ipynb

│   ├── RST.txt

│   ├── result.txt

│
├── multicollinearity_models/

│   ├── Visual_PCA_Multicollinearity.ipynb

│   ├── Classical_Ridge_Regression.ipynb

│
├── glm_models/

│   ├── Prediction_with_GLM.ipynb

│   ├── Logistic_Regression_Iris.ipynb

│   ├── Likelihood_Ratio_Test.ipynb

│
├── robust_models/

│   ├── IRLS_Robust_Regression.ipynb

│
├── applied_datasets/

│   ├── Depression.ipynb

│   ├── birthSmoker.ipynb

│   ├── Tool_Life_Data.ipynb

│   ├── Indicator_Var_More_Thn_2_Levels.py

The repository is organized into distinct directories for EDA, core regression, diagnostics, and advanced modeling:

* **Fundamental Math & Setup:** `ml_Math/` (linear algebra/calculus), `predata_analysis/` (EDA), `ScallingWhyRequired/` (scaling).
* **Linear Regression:** `simple_and_multiple_regressions/`, `OLS-Param Analysis.txt`.
* **Diagnostics & Validation:** `MultiCollinearity/`, `model_adequacy_checking_method/`, `plot-adequency/`, `importance_of_detecting_influential_obs/`, `PureError_LackOfFitTest/`, `Validitions of Regressions Models/`, `Confidance Interval/`.
* **Model Selection:** `StepwiseRegressions_Fw_BW_Seletions/`, `ExtraSumOfSquareStaticalMethod.ipynb`.
* **Advanced Modeling:** `polynomials_regressions_models/`, `INTRODUCTION TO NONLINEAR REGRESSION/`, `NonLinearRegressions/`, `transformations_wls_gls/`, `Regularized Linear Models/` (Ridge/Lasso), `GLM_Classifications Problems/`, `ROBUST REGRESSION/`, `TreeInML/`.
* **Optimization & Unsupervised:** `Scikit_Lern_Gradient_Descent/`, `Contput_Plot_Tutorials.ipynb`, `EarlyStopping/`, `CrossValidationsROCCurve/`, `Scikit-Learn DimencityReductions/`, `Scikit-Lern-Unsupervised Learning/`.

# Learning Goals by Directory

* **Foundations:** Master the mathematical rigor (matrix algebra, calculus) and initial data cleaning necessary for modeling.
* **Regression Modeling:** Learn to build OLS, polynomial, and non-linear regression models, understanding the parameters involved.
* **Diagnostics & Reliability:** Utilize tools to check for multicollinearity, heteroscedasticity, and influential observations to ensure model validity.
* **Advanced Techniques:** Explore regularization (Ridge/Lasso), generalized linear models (GLM), robust regression, and unsupervised techniques for complex data scenarios.
* **Validation:** Learn to properly validate model performance and interpret results through visual diagnostics and statistics.

### 1. polynomials_regressions_models

* **Goal:** Learn how polynomial regression extends linear models, and how **splines & knots** define flexible piecewise polynomials.
* **Focus:** Mathematical derivations, spline basis functions, interpolation, and feature selection.
* **Key Notebooks:**
  * `B_SPLINE_kagalDataset.ipynb` $\rightarrow$ B-spline regression on Kaggle dataset.
  * `SplinesPiecewisePolynomial.ipynb` $\rightarrow$ Theory of piecewise polynomials with continuity constraints.
  * `polynomial.ipynb` $\rightarrow$ Manual derivation of polynomial regression equations.

### 2. multicollinearity_models

* **Goal:** Understand the problem of **multicollinearity** in regression and apply dimensionality reduction or regularization.
* **Focus:** PCA visualization, ridge regression.
* **Key Notebooks:**
  * `Visual_PCA_Multicollinearity.ipynb` $\rightarrow$ Detecting correlated predictors.
  * `Classical_Ridge_Regression.ipynb` $\rightarrow$ Regularization to stabilize estimates.

### 3. glm_models

* **Goal:** Explore **Generalized Linear Models (GLM)** for categorical and non-normal data.
* **Focus:** Logistic regression, likelihood ratio tests, GLM extensions.
* **Key Notebooks:**
  * `Prediction_with_GLM.ipynb` $\rightarrow$ GLM applied to real dataset.
  * `Logistic_Regression_Iris.ipynb` $\rightarrow$ Classification with logistic regression.
  * `Likelihood_Ratio_Test.ipynb` $\rightarrow$ Hypothesis testing framework.

### 4. robust_models

* **Goal:** Learn robust regression techniques that reduce sensitivity to outliers.
* **Focus:** Iteratively Reweighted Least Squares (IRLS).
* **Key Notebook:**
  * `IRLS_Robust_Regression.ipynb` $\rightarrow$ Robust fitting with weighted residuals.

## 5. Transformations, WLS & GLS

This directory focuses on advanced regression techniques that go beyond ordinary least squares (OLS). The aim is to understand how transformations and generalized least squares (GLS) improve model accuracy and robustness when classical assumptions are violated.

### Key Objectives

* **Transformations of Variables**
  * Learn how log, square-root, and Box–Cox transformations stabilize variance and normalize residuals.
  * Understand when transformations are necessary to meet linear regression assumptions.

* **Weighted Least Squares (WLS)**
  * Explore regression when error variance is not constant (heteroscedasticity).
  * Learn how weights are applied to give less influence to high-variance observations.
  * Mathematical form:

$$\hat{\beta} = (X^T W X)^{-1} X^T W y$$

where $W$ is a diagonal weight matrix.

* **Generalized Least Squares (GLS)**
  * Extend regression to handle correlated errors.
  * Learn how GLS accounts for covariance structure in residuals.
  * Mathematical form:

$$\hat{\beta}_{GLS} = (X^T \Sigma^{-1} X)^{-1} X^T \Sigma^{-1} y$$

where $\Sigma$ is the error covariance matrix.

## 6. Simple & Multiple Regression

This directory introduces the foundations of regression analysis, starting from simple linear regression with one predictor and extending to multiple regression with several predictors. The aim is to build a strong mathematical and practical understanding of how regression models explain relationships between variables.

### Key Objectives

* **Simple Linear Regression**
  * Learn how a single predictor variable explains variation in a response.
  * Derive the regression line using Ordinary Least Squares (OLS):

$$y = \beta_0 + \beta_1 x + \epsilon$$

  * Interpret slope and intercept in real-world contexts.

* **Multiple Regression**
  * Extend regression to multiple predictors:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$$
  * Understand how each predictor contributes while controlling for others.
  * Explore multicollinearity detection and its impact on coefficient stability.

* **Model Diagnostics**
  * Residual analysis (fitted vs residuals, QQ plots).
  * Goodness-of-fit measures: $R^2$, Adjusted $R^2$.
  * Hypothesis testing with p-values and F-statistics.

### Why It Matters

* Forms the core building block for advanced regression methods (polynomials, splines, GLM, robust regression).
* Provides intuition for **prediction vs inference** in statistical modeling.
* Equips learners with tools to evaluate model assumptions and reliability.

## 7. Model Adequacy Checking Methods

This directory is dedicated to evaluating whether regression models are appropriate and reliable. The focus is on diagnostic techniques that test assumptions, detect influential observations, and ensure the model provides valid statistical inference.

### Key Objectives

* **Assumption Checking**
  * Verify linearity, independence, homoscedasticity (constant variance), and normality of residuals.
  * Use diagnostic plots (residual vs fitted, QQ plots, scale-location plots).

* **Influential Observations**
  * Learn the importance of detecting points that disproportionately affect regression estimates.
  * Apply measures such as **Cook's Distance, Leverage values, DFFITS, and DFBETAs**.
  * **Goal:** Ensure model stability by identifying and addressing outliers or high-leverage points.
* **Goodness-of-Fit & Adequacy Tests**
  * Evaluate $R^2$, Adjusted $R^2$, and overall F-statistics.
  * Apply lack-of-fit tests to determine whether the chosen model form is sufficient.

* **Robustness & Alternatives**
  * Explore when transformations, weighted least squares (WLS), or generalized least squares (GLS) are needed.
  * Connect adequacy checking with corrective modeling strategies.

### 8.Indicator Variables in Regression

This directory focuses on the role of indicator (dummy) variables in regression analysis. The aim is to understand how categorical predictors are encoded and interpreted within linear and multiple regression models.

### Key Objectives

* **Encoding Categorical Variables**
  * Learn how to represent qualitative data (e.g., gender, smoker/non-smoker, treatment groups) using indicator variables.
  * Understand dummy variable coding (0/1) and extensions for variables with more than two levels.

* **Regression with Indicator Variables**
  * Explore how indicator variables shift intercepts and slopes in regression equations.
  * Example model:

$$y = \beta_0 + \beta_1 x + \beta_2 D + \epsilon$$

where $D$ is an indicator variable (0 or 1).
* **Multiple Categories**
  * Learn how to handle categorical predictors with more than two levels using multiple dummy variables.
  * Avoid the **dummy variable trap** (perfect multicollinearity).

* **Interpretation**
  * Practice interpreting regression coefficients when indicator variables are present.
  * Understand baseline category vs comparison categories.


### 9. applied_datasets

* **Goal:** Apply regression techniques to diverse real-world datasets.
* **Focus:** Health, industrial, and categorical variable modeling.
* **Key Notebooks:**
  * `Depression.ipynb` $\rightarrow$ Logistic regression on mental health dataset.
  * `birthSmoker.ipynb` $\rightarrow$ Regression with categorical smoking indicator.
  * `Tool_Life_Data.ipynb` $\rightarrow$ Polynomial regression on tool wear dataset.
  * `Indicator_Var_More_Thn_2_Levels.py` $\rightarrow$ Dummy variable encoding for categorical predictors.

## Mathematical Foundation

### Polynomials

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d + \epsilon$$

* Captures nonlinear trends.
* Higher degree increases flexibility but risks overfitting.

### Splines & Knots

$$S(x) = \sum_{j=1}^{k} \beta_j B_j(x)$$

* Knots divide the domain into intervals.
* Each interval fitted with its own polynomial.
* Smoothness constraints ensure continuity across knots.
