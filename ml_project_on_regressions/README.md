# A). Decentral Smart Grid Control (DSGC)

The file Smart-Grid-nominal-power-study.ipynb located within the ML_Project/ml_project_notebook/ directory evaluates Decentral Smart Grid Control (DSGC) systems. It models stability and power dynamics across decentralized energy networks.
# Objectives of this notebook

The core technical and pedagogical objectives of this notebook include:

* **Decentralized Grid Physics:** Understanding the 4-node star architecture (comprising one central power supply source and three localized consumer nodes).
* **Stability Optimization:** Modeling how nominal power production, consumption fluctuations, and participant response times alter AC grid frequency stability.
* **Price Elasticity Mechanics:** Evaluating economic-engineering feedback loops, specifically tracking how consumers alter their energy footprints in response to real-time grid price adjustments.
* **Advanced Feature Interactions:** Extracting non-linear operational bounds where structural load balances break, causing systemic power grid failure.
# Detailed Summary of the Notebook

The notebook is divided into structured phases following a production-grade machine learning workflow:

### 1. Theoretical Setup & Math Framework

The notebook starts by implementing the **Decentral Smart Grid Control (DSGC)** mathematical framework. Instead of relying on a rigid, top-down traditional distribution schema, it analyzes a modern two-way smart framework where price dictates load balance. Stability is treated as a function of grid frequency deviations: if production and consumption drift out of sync, the system risks cascading brownouts.

### 2. Feature & Attribute Analysis

The notebook utilizes the standard 14-attribute smart grid stability dataset (typically sourced from Vadim Arzamasov / Kaggle). It breaks down twelve structural predictors to isolate their impact on nominal power:

* **Tau ($\tau_1$ to $\tau_4$):** Reaction time coefficients for each network participant adjusting to price signals.
* **Power ($p_1$ to $p_4$):** Nominal power values. $p_1$ represents the primary generator source (always positive), while $p_2, p_3, p_4$ handle consumer nodes (always negative, indicating consumption).
* **Gamma ($\gamma_1$ to $\gamma_4$):** Price elasticity parameters mapping participant sensitivity to electricity price variations.
### 3. Statistical Analysis & Feature Engineering

The notebook processes the feature space to separate target definitions. It evaluates how variances in the continuous root features map to system traits:

* **Mathematical Target Balancing:** Inspecting the continuous stability root differential (`stab`) and its binary classification counterpart (`stabf`: stable vs. unstable).
* **Multicollinearity Checks:** Evaluating cross-correlations between the nominal power consumption levels of the consumer nodes ($p_2, p_3, p_4$) and their physical impact on overall network resilience.
### 4. Baseline Modeling & Behavior Insights

By evaluating how the system balances nominal power constraints against pricing elasticity, this study establishes reference criteria for predictive classifiers. It highlights that **participant response times ($\tau$)** and **load volumes ($p$)** serve as the leading indicators for capturing critical thresholds where localized grids flip from operational equilibria to unstable structural failures.

# B). diagnostics for linear regression using the classic Auto MPG dataset

The file auto-mpg-regression-outliers-residuals.ipynb located within the ML_Project/ml_project_notebook/ directory focuses on classical diagnostics for linear regression using the classic Auto MPG dataset. It serves as an end-to-end practical study on how anomalies and residual patterns impact standard Ordinary Least Squares (OLS) models.

# Objectives of this notebook

The core statistical and diagnostic objectives of this notebook include:

* **Residual Analysis Mechanics:** Learning how to visually and statistically evaluate residuals to verify standard OLS assumptions (linearity, homoscedasticity, and normality).
* **Outlier & Leverage Identification:** Detecting data points with extreme feature values (high leverage) or extreme target errors (outliers) that skew predictive metrics.
* **Influential Observation Metrics:** Calculating quantitative mathematical metrics like Cook’s Distance and DFBETAS to identify specific data points that disproportionately alter model coefficients.
* **Model Remediation:** Understanding when and how to safely transform variables or prune specific anomalies to build stable, reliable regressions.
# Detailed Summary of the Notebook

The notebook walks through a rigorous workflow designed to critique and improve a baseline linear model:

### 1. Exploratory Data Analysis & Baseline OLS

* **Data Prep:** The notebook loads vehicle specifications (such as cylinders, displacement, horsepower, weight, and acceleration) to predict fuel efficiency (`mpg`).
* **Initial Model Fitting:** A baseline multiple linear regression model is trained using OLS. The initial summary parameters ($R^2$, adjusted $R^2$, and $t$-statistics) are generated to establish a benchmarking reference.
### 2. Residual Diagnostics

The notebook systematically evaluates the errors of the baseline model to check for violations of OLS assumptions:

* **Non-linearity & Heteroscedasticity:** Plots of *Residuals vs. Fitted Values* are generated. Curvatures in this plot signal missing polynomial terms (non-linearity), while a funnel-like expansion signals changing error variance (heteroscedasticity).
* **Non-Normality:** Quantile-Quantile (Q-Q) plots compare the sample residual distribution against a theoretical normal distribution to identify heavy tails or skewed error distributions.
### 3. High Leverage and Outlier Detection

The notebook isolates unusual observations by separating their geometric position in the feature space from their actual target errors:

* **Leverage (Hat Matrix):** Computes the hat values ($h_{ii}$) to flag vehicles with extreme or rare technical specifications that exert high geometric leverage over the orientation of the regression line.
* **Studentized Residuals:** Residuals are divided by their estimated standard deviation. Data points with studentized residuals exceeding $\pm2$ or $\pm3$ are flagged as statistical outliers.
### 4. Quantifying Influential Observations

To bridge the gap between leverage and outliers, the notebook calculates metrics to pinpoint observations that single-handedly shift the model's coefficients:

* **Cook's Distance ($D_i$):** Measures the aggregate shift in *all* predicted values when a specific vehicle is deleted from the training pool. Points exceeding the classic threshold of $4/n$ are flagged for review.
* **DFBETAS:** Examines exactly how much an individual regression coefficient ($\beta_j$) changes when a specific outlier is excluded, isolating exactly which vehicle features are most sensitive to anomalous data.
### 5. Model Refinement & Comparison

* The notebook demonstrates the downstream effect of removing highly influential, corrupted, or unrepresentative data points.
* By refitting the OLS model on the filtered data, it illustrates how diagnostic pruning stabilizes standard errors, rectifies residual distributions, and yields a generalized, trustworthy regression equation.

# C). Ridge Regression:
The file classical-ridge-regression-for-housing.ipynb located within the ML_Project/ml_project_notebook/ directory focuses on mitigating overfitting and handling multicollinearity using L2 regularization. It provides a step-by-step framework for transitioning from Ordinary Least Squares (OLS) to Ridge Regression when working with highly correlated spatial and structural housing attributes.
# Objectives of this notebook

The core statistical and implementation objectives of this notebook include:

* **Regularization Mechanics:** Understanding the mathematical theory of adding an L2 penalty term (squared magnitude of coefficients) to the OLS loss function.
* **Multicollinearity Mitigation:** Learning how Ridge regression stabilizes coefficient estimates and reduces variance when feature vectors are collinear.
* **Hyperparameter Tuning ($\alpha$ / $\lambda$):** Implementing validation loops (such as $k$-fold cross-validation) to locate the optimal regularization strength.
* **Bias-Variance Trade-off:** Visualizing and assessing how introducing a small amount of bias into the parameter estimates drops the overall model variance.
# Detailed Summary of the Notebook

The notebook follows a systematic applied machine learning pipeline to compare baseline models against their regularized variants:

### 1. Exploratory Data Analysis & Feature Scaling

* **Dataset Ingestion:** The notebook processes housing indicators (typically variables like square footage, bedrooms, bathrooms, and location metrics) to map real estate market valuations.
* **Standardization:** Because Ridge regression penalizes the squared magnitude of coefficients, features on larger scales (e.g., total square feet) will be unfairly penalized compared to smaller scales (e.g., bedroom count). The notebook scales the feature matrix $X$ using standard scores ($z$-scores) before entering the model framework.
### 2. The Limits of Baseline OLS

* **Unconstrained Fitting:** An initial standard multiple linear regression model is trained.
* **Variance Evaluation:** The notebook inspects the baseline's susceptibility to multicollinearity. When predictors track too closely together, standard errors inflate, making the resulting OLS model overly sensitive to random noise in the training subset.
### 3. Transition to Ridge Regression (L2 Regularization)

The notebook introduces the Ridge cost minimization objective function:

$$\text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{m} \beta_j^2$$

* **Shrinkage Action:** It illustrates how the penalty multiplier ($\alpha$ or $\lambda$) shrinks the magnitude of the $\beta$ coefficients asymptotically toward zero. This prevents any single feature from capturing disproportionate leverage over the system.
### 4. Hyperparameter Tuning and Cross-Validation

* **Alpha Path Inspection:** The notebook plots the *regularization path*, showing how individual feature coefficients drop as $\alpha$ increases from $0$ to large values.
* **Validation:** Utilizing tools like `RidgeCV` or manual $k$-fold cross-validation loops, it calculates evaluation parameters (such as Root Mean Squared Error (RMSE) or $R^2$) across a spectrum of alpha possibilities to select the parameter that yields the highest generalization on test data.

### 5. Performance Benchmarking

* **Final Comparisons:** The regularized Ridge model is evaluated side-by-side with the unconstrained OLS model across test sets.
* **Key Insight:** The notebook highlights that while Ridge regression incurs slightly higher bias on the training split, it yields a major drop in testing error (RMSE), making it a much more robust option for real-world real estate deployment.

