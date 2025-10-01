import numpy as np
import numpy.testing as npt
from scipy import stats
from scipy.stats import f,t
from scipy.stats import f_oneway
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices
from olsRegressionAnalysis import dispAnalysisOfVariance, tableDispFormatt

path = r'C:\Users\TODO\Desktop\Abhi\AI\AI\allDataSet\STAT501_Lesson05\STAT501_Lesson05\iqsize.txt'

df = pd.read_csv(path)
#print(df)

def getRegressionEqn(pdSeries):
    eqnStr = str()
    idx = (pdSeries.index).to_list()
    val = (pdSeries.values).tolist()
    for i in np.arange(0,len(idx),1):
        if val[i] >= 0.0:
            if i > 0:
                eqnStr += '+' + ' ' +str(val[i])+' '+ idx[i] + ' '
            else:
                eqnStr += str(val[i])+' '+ idx[i] + ' '
        else:
            if i > 0:
                eqnStr +=  str(val[i])+' '+ idx[i] + ' '
            else:
                eqnStr +=  str(val[i])+' '+ idx[i] + ' '
    print(eqnStr)
    return eqnStr

y,X = dmatrices('PIQ ~ Brain + Height + Weight',data=df, return_type='dataframe')

mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
tableDispFormatt('You are here')

#Get T values
'''
To claculate t-score you can usr t-score formula.
Not includin calculations of t score.
'''
tableDispFormatt('JamboJet')
print(res.tvalues,res.tvalues.values,(res.tvalues.index.values).tolist())
print(type(res.tvalues.index.values))
#Get p values of T score
idx = 'Intercept,Brain,Height,Weight'
ret = res.t_test(idx)
#Get P val Method - 1
print(ret.pvalue)

#Get P val Method - 2 scipy.stat.t
'''
t.sf(np.abs(t_score), df) * 2  # for two-tailed test
t.sf(t_score, df)              # for left-tailed test
t.sf(-t_score, df)             # for right-tailed test
'''
print('p value from t score: ',2*t.sf(x = res.tvalues, df = res.df_resid, loc=0, scale=1))

# get 95% significance level
'''
CI = 95% or(100*(1-α)) i.e. α = 0.05
'''
print('Significance level: ',t.isf(q = 0.05, df = res.df_resid, loc=0, scale=1))

# t-test Intercept
T = [1,0,0,0]
ret = res.t_test(T)
hypotheses = 'Intercept = 0'
ret = res.t_test(hypotheses)
print(ret.tvalue, ret.pvalue)

# t-test Brain
T = [0,1,0,0]
ret = res.t_test(T)
print(ret.tvalue, ret.pvalue)

# t-test Height
T = [0,0,1,0]
ret = res.t_test(T)
print(ret.tvalue, ret.pvalue)

# t-test Weight
T = [0,0,0,1]
ret = res.t_test(T)
print(ret.tvalue, ret.pvalue)



