import numpy as np
import numpy.testing as npt
from scipy import stats
from scipy.stats import t,ttest_ind
from scipy.stats import f
from scipy.stats import f_oneway
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
from olsRegressionAnalysis import dispAnalysisOfVariance, tableDispFormatt,getInvOfProductMat,getRegressionEqn,\
                                  dispReghressionAnalysis

path  = r"C:\Users\TODO\Desktop\Abhi\AI\AI\allDataSet\bOOK-DataSet\AcetyleneData_10_1.txt"
df = pd.read_csv(path)
print(df.columns)

#Step 1: Matrix Plot of Data
sns.set_theme(style="ticks")
sns.pairplot(df)
plt.show()