# GLM Classifications

**Repository Link:** [LKOBUI/stat_analysis](https://github.com)
**Target Directory:** `glm_classifications_statical_study`

### Repository Learning Objectives
* Master the implementation of Generalized Linear Models (GLMs) optimized for classification problems.
* Comprehend the transition from continuous linear predictors to bounded probability spaces using specific link functions.
* Evaluate classification model performance using rigorous statistical metrics and diagnostic boundaries.

### Detailed Notebook Breakdowns

#### 1. Binary Logistic Regression Analysis (`binary_logistic_regression.ipynb`)
* **Objective:** Model binary categorical outcomes ($Y \in \{0, 1\}$) against continuous and categorical independent predictors.
* **Model in Use & Justification:** **Binary Logistic Regression**. This framework is utilized because the response variable is strictly dichotomous, rendering standard Ordinary Least Squares (OLS) regression invalid due to non-normal error distributions and out-of-bound predictions.
* **Mathematical Formulation:**
  The model maps the linear predictor to a probability space via the **Logit Link Function**:
  $$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k$$
  Where $p = P(Y=1|X)$ represents the conditional probability of the target event. The inverse link outputs the predicted probability using the **Sigmoid Function**:
  $$p = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i X_i)}}$$
  Parameters are estimated using Maximum Likelihood Estimation (MLE) rather than OLS to optimize the log-likelihood function for a Bernoulli distribution.

#### 2. Probit Classification Study (`probit_modeling.ipynb`)
* **Objective:** Analyze binary classification boundaries using an alternative probability distribution mapping derived from normally distributed latent variables.
* **Model in Use & Justification:** **Probit Regression Model**. This approach is selected for scenarios where the underlying unobserved threshold (latent variable) follows a standard normal distribution, providing an alternative to the heavier-tailed logistic distribution.
* **Mathematical Formulation:**
  The Probit model relies on the inverse of the standard normal cumulative distribution function (CDF), known as the **Inverse Normal Link Function**:
  $$\Phi^{-1}(p) = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k$$
  Where $\Phi(z)$ represents the standard normal CDF:
  $$p = \Phi(\beta_0 + \sum \beta_i X_i) = \int_{-\infty}^{\beta_0 + \sum \beta_i X_i} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt$$
  The coefficients represent the change in the $z$-score of the probability for a one-unit change in the predictor variable.

#### 3. Multinomial Logistic Regression (`multinomial_classification.ipynb`)
* **Objective:** Extend binary classification frameworks to predict nominal outcomes featuring more than two distinct, unordered categories ($M > 2$).
* **Model in Use & Justification:** **Multinomial Logistic Regression (Softmax Regression)**. This structure is required because the categories are mutually exclusive but lack an inherent numerical or ordinal ranking (e.g., choice of product, transport mode).
* **Mathematical Formulation:**
  The model designates a baseline reference category (typically category $1$) and computes log-odds ratios for the remaining categories:
  $$\ln\left(\frac{P(Y=j)}{P(Y=1)}\right) = \beta_{j0} + \beta_{j1} X_1 + \dots + \beta_{jk} X_k \quad \text{for } j = 2, \dots, M$$
  The generalized probability for any specific category $j$ is derived via the **Softmax Transformation**:
  $$P(Y=j|X) = \frac{e^{X\beta_j}}{1 + \sum_{m=2}^{M} e^{X\beta_m}}$$
  This ensures that the summation of predicted probabilities across all possible categories equals exactly 1.

#### 4. Poisson and Negative Binomial Count Classifications (`count_classification_study.ipynb`)
* **Objective:** Analyze and classify discrete count outcomes while adjusting for structural issues like overdispersion.
* **Model in Use & Justification:** **Poisson Regression** linked with **Negative Binomial Regression**. Poisson models are used for count data where variance equals the mean. When variance exceeds the mean (overdispersion), the Negative Binomial model is substituted to prevent underestimated standard errors.
* **Mathematical Formulation:**
  For Poisson regression, the relationship uses the **Log Link Function**:
  $$\ln(\lambda) = X\beta \implies \lambda = e^{X\beta}$$
  Where $\lambda$ represents the expected count rate $E(Y|X)$. The probability mass function follows:
  $$P(Y=y|X) = \frac{e^{-\lambda} \lambda^y}{y!}$$
  The Negative Binomial model expands this framework by introducing an overdispersion parameter $\alpha$, altering the variance structure to:
  $$\text{Var}(Y) = \lambda + \alpha \lambda^2$$

### Performance Metrics Captured
* **Deviance Analysis:** Evaluates residual deviance against null deviance to determine the overall significance of the predictors.
* **Akaike Information Criterion (AIC):** Balances goodness-of-fit against model complexity to prevent overfitting.
* **Confusion Matrices:** Measures exact classification boundaries through precision, recall, and F1-score tracking.
