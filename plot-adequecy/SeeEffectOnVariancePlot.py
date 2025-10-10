import numpy as np
import numpy.testing as npt
from scipy import stats
from scipy.stats import f
from scipy.stats import f_oneway
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices
from numpy.testing import assert_almost_equal, assert_allclose
import matplotlib.pyplot as plt
import seaborn as sns

sample_size = 1000
sx = np.linspace(start=-2,stop=2,num=sample_size)
sy = np.exp(sx)

df = pd.DataFrame({'SY':sy,'SX':sx})

y, X = dmatrices(
                 formula_like = 'SY ~ SX', 
                 data=df,
                 return_type='dataframe'
                 )
res = sm.OLS(y, X).fit()

fig, axes = plt.subplots(2, 2)
sns.histplot(data = res.resid, kde = True,ax=axes[0,0])
sm.qqplot(data = res.resid, dist=stats.norm,line='s',ax=axes[0,1])

df['SY_t'] = np.log(sy)

y, X = dmatrices(
                 formula_like = 'SY_t ~ SX', 
                 data=df,
                 return_type='dataframe'
                 )
res = sm.OLS(y, X).fit()

sns.histplot(data = res.resid, kde = True,ax=axes[1,0])
sm.qqplot(data = res.resid, dist=stats.norm,line='s',ax=axes[1,1])

plt.show()