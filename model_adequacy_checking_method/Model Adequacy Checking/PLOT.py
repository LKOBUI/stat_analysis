import numpy as np
import numpy.testing as npt
from scipy import stats
from scipy.stats import t, ttest_ind
from scipy.stats import f
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.regression._prediction import get_prediction
from statsmodels.stats.outliers_influence import OLSInfluence, MLEInfluence
from statsmodels.graphics.gofplots import qqplot_2samples
import pandas as pd
from patsy import dmatrices
from numpy.testing import assert_almost_equal, assert_allclose

# some_file.py
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r"C:\Users\TODO\Desktop\Abhi\AI\AI\Math\Hands-On\statemodelsStudy")
from olsRegressionAnalysis import (
    dispAnalysisOfVariance,
    tableDispFormatt,
    getInvOfProductMat,
)

import matplotlib.pyplot as plt
import seaborn as sns

"""
Visit python file for all type of plot
statsmodels\graphics\gofplots.py
"""
"""
Three Type of plot:
ppplot([xlabel, ylabel, line, other, ax])            Plot of the percentiles of x versus the percentiles of a distribution.
probplot([xlabel, ylabel, line, exceed, ax])         Plot of unscaled quantiles of x against the prob of a distribution.
qqplot([xlabel, ylabel, line, other, ax, swap])      Plot of the quantiles of x versus the quantiles/ppf of a distribution.
Reffer wiki link 
https://en.wikipedia.org/wiki/Normal_probability_plot#:~:text=In%20a%20normal%20probability%20plot,line%20suggest%20departures%20from%20normality.
"""

path = r"C:\Users\TODO\Desktop\Abhi\AI\AI\allDataSet\bOOK-DataSet\TABLE_3_2_DeliveryTimeData.csv"
df = pd.read_csv(path)
df.columns = ["Obs", "DlvrTImeY", "NumCaseX1", "DstX2"]

y, X = dmatrices(
    formula_like="DlvrTImeY ~ NumCaseX1 + DstX2", data=df, return_type="dataframe"
)
res = sm.OLS(y, X).fit()
adequacyCheck = OLSInfluence(res)
ti = adequacyCheck.resid_studentized_external
qqplot_2samples(
    data1=res.resid, data2=ti, xlabel="3-5", ylabel="5-6", line="q", ax=None
)
"""
Three type of plot exixt: [For details See wiki]
ppplot: Probability-Probability plot
    Compares the sample and theoretical probabilities (percentiles).
qqplot: Quantile-Quantile plot
    Compares the sample and theoretical quantiles
probplot: Probability plot
    Same as a Q-Q plot

1). qqplot:
    For more details see this: statsmodels\graphics\gofplots.py
    Ex1: qqplot - residuals of OLS fit
    fig = sm.qqplot(res.resid)
    
    Ex2:
    qqplot of the residuals against quantiles of t-distribution with 4 degrees
    of freedom:
    fig = sm.qqplot(res.resid,line="45", stats.t, distargs=(4,))
    
    Ex 3:
    qqplot against same as above, but with mean 3 and std 10:
    fig = sm.qqplot(res.resid,line="45", stats.t, distargs=(4,), loc=3, scale=10)
    
    Ex4:
    Automatically determine parameters for t distribution including the
    loc and scale:
    fig = sm.qqplot(res.resid,line="45", stats.t, fit=True, line="45")
    
    Ex. 5 - qqplot - compare two sample sets
    pp_x = sm.ProbPlot(x, fit=True)
    pp_y = sm.ProbPlot(y, fit=True)
    fig = pp_x.qqplot(line="45", other=pp_y)

2). For rest you can reffer Same method. See method
    as per youre need
"""
plt.show()
