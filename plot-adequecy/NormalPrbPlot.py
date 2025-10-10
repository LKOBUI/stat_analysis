import numpy as np
import numpy.testing as npt
from scipy import stats
from scipy.stats import t,ttest_ind
from scipy.stats import f
from scipy.stats import f_oneway
from scipy.stats import norm, skewnorm
import statsmodels.api as sm
from statsmodels.regression._prediction import get_prediction
from statsmodels.stats.outliers_influence import OLSInfluence,MLEInfluence
from statsmodels.graphics.gofplots import qqplot_2samples,qqplot
import pandas as pd
from patsy import dmatrices
from numpy.testing import assert_almost_equal, assert_allclose

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r'C:\Users\TODO\Desktop\Abhi\AI\AI\Math\Hands-On\statemodelsStudy')
from olsRegressionAnalysis import dispAnalysisOfVariance, tableDispFormatt,getInvOfProductMat

import matplotlib.pyplot as plt
import seaborn as sns

'''
The location (loc) keyword specifies the mean. 
The scale (scale) keyword specifies the standard deviation.
'''
MEAN = 0
DEV  = 1
fig, axes = plt.subplots(2, 2)

sample_size = 10000 

standard_norm = np.random.normal(size=sample_size,loc = MEAN, scale = DEV)
sns.histplot(standard_norm, kde = True,ax=axes[0,0])
qqplot(data = standard_norm, dist=stats.norm,line='s',loc = MEAN, scale = DEV,ax=axes[0,1])

skewed_norm = skewnorm.rvs(a=10, size=sample_size)
sns.histplot(skewed_norm, kde = True,ax=axes[1,0])
qqplot(data = skewed_norm, dist=stats.norm,line='s',loc = MEAN, scale = DEV,ax=axes[1,1])

plt.show()