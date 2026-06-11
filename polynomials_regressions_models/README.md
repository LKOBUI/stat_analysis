## Polynomial Regression Models & Splines:

This repo provides an in-depth implementation and mathematical analysis of Polynomial Regression Models, Piecewise Polynomials, and Smoothing/Basis Splines (B-Splines)

# Primary Learning Goals: 

## Mathematical Mastery of Polynomials: 

Understanding how non-linear curve fitting is achieved through transformations 
while maintaining a model that is linear in its parameters.

## Splines and Knots Definition: 

Learning how to split a global variable's range into local intervals 
using boundary points called knots. Piecewise polynomials are then fit 
within these segments under smooth continuity constraints to prevent 
global oscillations (Runge's Phenomenon)

## Mathematical Approach of Polynomials:

A polynomial regression model of degree ($k$) in a single variable is expressed as:

$Y=\beta _{0}+\beta _{1}X+\beta _{2}X^{2}+\dots +\beta _{k}X^{k}+\epsilon$

## Key Mathematical Characteristics:

* **Linearity in Parameters:** Although the geometric relationship between $X$ and $Y$ is highly non-linear, the model is mathematically linear with respect to its coefficients ($\beta_0, \beta_1, \dots, \beta_k$). This allows standard Ordinary Least Squares (OLS) estimation techniques to be applied directly.

* **Ill-Conditioning Risks:** As the power $k$ increases, the columns of the design matrix ($X, X^2, X^3, \dots$) become heavily correlated, creating severe multicollinearity. This leads to an ill-conditioned $(\mathbf{X}'\mathbf{X})$ matrix, making estimates unstable.

* **The Spline Alternative:** Instead of increasing $k$ globally, splines introduce a piecewise approach. For a spline of degree $d$ with $K$ internal knots, the model utilizes a linear combination of basis functions (like B-splines) ensuring that the function, along with its first $d - 1$ derivatives, remains perfectly continuous at every knot checkpoint.

## Polynomial Regression Frameworks

#### polynomial.ipynb:

Implements raw polynomial expansions from scratch. Focuses on manual feature matrix construction using NumPy and fitting coefficients using core algebraic functions.

#### polynomial_scikitlern.ipynb

Demonstrates structured workflows using sklearn.preprocessing.PolynomialFeatures and LinearRegression chained within execution pipelines. Includes step-by-step visual evaluations comparing linear vs. higher-order polynomial bounds.

#### FwSelectionBwElimination.ipynb:

Explores automated feature selection methods (Forward Selection and Backward Elimination) to determine the optimal degree \(k\) of a polynomial model, preventing overfitting by systematically dropping statistically insignificant higher-order terms..

#### Real-World Application & Basics

#### HardwoodData.ipynb:

A practical regression case study analyzing physical data (such as hardwood processing properties). Applies polynomial curve fitting to analyze trends and validate residual diagnostics against practical engineering requirements.

#### LINEAR INTERPOLATION.ipynb:

Focuses on baseline piecewise linear structures. Connects successive data points using direct linear segments, establishing the structural foundation for understanding smoother higher-order splines.

## 🔗 Advanced Splines & Piecewise Estimators

* **`SplinesPiecewisePolynomial.ipynb`**: Explores the mechanics of splitting data domains at specified knots and fitting independent local polynomials within each partition. Demonstrates truncating power bases to establish continuity.

* **`dmatrix_spline.ipynb`**: Utilizes the `patsy` library to generate Design Matrices (`dmatrix`) for complex statistical models. Automates the conversion of raw inputs into spline basis expansions (e.g., natural splines or B-splines).

* **`B_SPLINE_kagalDataset.ipynb`**: Implements robust Basis Spline (B-Spline) regressions on Kaggle datasets. Focuses on tuning hyper-parameters, optimizing knot counts, and evaluating predictive performance on test data.

* **`B_SPLINE_uswages.ipynb`**: Applies B-Splines to the classic US Wages econometric dataset. Models the non-linear relationship between worker experience/education and wages, capturing complex wage curves smoothly without global polynomial distortion.

#### Text Logs & Reference Files:

RST.txt / result.txtStores training output summaries, evaluation scores (\(R^{2}\), MSE), and convergence logs generated during spline optimization benchmarks

#### Environment Configuration:

pip install numpy pandas matplotlib scipy scikit-learn statsmodels patsy
