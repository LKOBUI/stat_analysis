# Simple and Multiple Linear Regressions: Causes of Wrong Signs

This repository explores linear regression analysis, focusing heavily on why regression coefficients ($\widehat{\beta}$) can sometimes exhibit unexpected or "wrong" signs (contrary to theory or simple correlation) and how the choice of experimental design impacts estimator variance.

## 📂 Repository Contents

* **`SingleAndMultiRegressions.ipynb` / `simpal_and_multipal_linear_regressions_notebook.ipynb`** — Introduction to fitting and contrasting simple vs. multiple linear regression models.
* **`REGRESSION Cofficient Wrong Sign.ipynb`** — Comprehensive simulations demonstrating how data spread, omitted variables, and multicollinearity flip the mathematical sign of coefficients.
* **`NotsOn_HIDDEN EXTRAPOLATION_PG_200.ipynb`** — Detailed study on the dangers of hidden extrapolation when predicting outside the joint region of the regressors.

---

## 🔬 Core Statistical Concepts Covered

### 1. The Spread of the Regressor ($X$) and Estimator Variance
The variance of the slope estimator $\widehat{\beta}_1$ in a simple linear regression is inversely proportional to the "spread" or sum of squared deviations of the independent variable:

$$\text{Var}(\widehat{\beta}_1) = \frac{\sigma^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2}$$

* **Low Spread:** When data points of $X$ are tightly clustered together, $\text{Var}(\widehat{\beta}_1)$ becomes extremely large. This inflates the sampling distribution curve, significantly increasing the probability of getting a negative sample estimate even if the true population parameter $\beta_1$ is positive.
* **Experimental Trade-off:** While widening the range of $X$ reduces estimator variance, doing it too aggressively introduces risks:
  1. It can force you to use much more complex equations if the underlying true response function is non-linear.
    2. It might push your data points out of the experimenter's actual practical region of interest.

    ### 2. Omitted Variable Bias & Sign Reversal (Total vs. Partial Effects)
    A common reason for an unexpected sign is omitting a vital confounding variable ($x_2$). 
    * **Total Coefficient:** In $\hat{y} = \beta_0 + \beta_1 x_1$, $\widehat{\beta}_1$ captures the absolute total relationship while ignoring $x_2$.
    * **Partial Coefficient:** In $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$, $\widehat{\beta}_1$ measures the effect of $x_1$ **given that $x_2$ is held constant**. 

    When a confounder is omitted, the partial coefficient can completely flip sign (e.g., changing from a positive total slope to a negative partial slope), mimicking Simpson's Paradox.

    ### 3. Severe Multicollinearity
    When regressors are highly correlated with one another, the $(X'X)$ cross-product matrix becomes **ill-conditioned**, driving up the condition number drastically. This causes:
    * Severely inflated variance inflation factors (VIF).
    * Extreme sensitivity to rounding and truncation errors in software computation.
    * Unstable coefficient estimates that flip signs or explode by several orders of magnitude with minor data changes.

    ---

    ## 🛠️ Required Python Packages

    To run the notebooks successfully, ensure you have the following libraries installed:
    ```bash
    pip install numpy pandas matplotlib scipy scikit-learn statsmodels
    ```

