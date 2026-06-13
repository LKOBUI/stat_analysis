import numpy as np
import numpy.testing as npt
from math import *

from scipy import stats
from scipy.stats import t,ttest_ind
from scipy.stats import f
from scipy.stats import f_oneway
from scipy.stats import pearsonr

import statsmodels.api as sm
from statsmodels.regression._prediction import get_prediction
from statsmodels.stats.outliers_influence import OLSInfluence,MLEInfluence
from statsmodels.graphics.gofplots import qqplot_2samples,ProbPlot,qqplot
import pandas as pd
from patsy import dmatrices
from numpy.testing import assert_almost_equal, assert_allclose
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r'C:\Users\TODO\Desktop\Abhi\AI\AI\Math\Hands-On\statemodelsStudy')
from olsRegressionAnalysis import dispAnalysisOfVariance, tableDispFormatt,getInvOfProductMat,\
                                  getRegressionEqn,\
                                  dispReghressionAnalysis,norm_scalling,getCorrelation,\
                                      get_variance_inflation_factors

x = 2 # initial values
delta = 10**-6
itr = 0
for itr in np.arange(1,100,1):
    Xnew = x - np.divide(((x**3) + (3*x) + 1),((3*x**2) + 3))    
    if abs(Xnew - x) < delta:
        break
    x = Xnew
print('itr: ',itr, ' x:',x)

# f(x) = x**2 + x -6

x = -6 # initial values
delta = 1e-6 #10**-6
itr = 0
for itr in np.arange(1,100,1):
    Xnew = x - (x**2 + x -6)/(2*x + 1)   
    if abs(Xnew - x) < delta:
        break
    x = Xnew
print('itr: ',itr, ' x:',x)

x = 6 # initial values
delta = 1e-6 #10**-6
itr = 0
for itr in np.arange(1,100,1):
    Xnew = x - (x**2 + x -6)/(2*x + 1)   
    if abs(Xnew - x) < delta:
        break
    x = Xnew
print('itr: ',itr, ' x:',x)