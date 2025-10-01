import numpy as np
import numpy.testing as npt
from scipy import stats
from scipy.stats import f
from scipy.stats import f_oneway
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices
from olsRegressionAnalysis import dispAnalysisOfVariance, tableDispFormatt

path = r'C:\Users\TODO\Desktop\Abhi\AI\AI\allDataSet\bOOK-DataSet\TABLE_3_2_DeliveryTimeData.csv'
df = pd.read_csv(path)
df.columns = ['Obs', 'DlvrTime_Y',  'NumCase_X1',  'Dst_X2']
#print(df.head())

y,X = dmatrices('DlvrTime_Y ~ NumCase_X1 + Dst_X2',data=df, return_type='dataframe')

mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
tableDispFormatt('Here we are')

# Significance test of regressors:

# 1). Hypothesis test of compleate models
'''
Let your models erqn => 
y = β0 +  β1x1 +  β2x2 + β3x3 +…
H0(null hypothesis): y not dependent on β1 and/or  β2 and/or β3 ....
Hα(alternate hypothesis): y dependent on β1 and/or  β2 and/or β3 
now if res.fvalue > res.f_pvalue -> Null hypothesis rejected
Conclude, y dependent on all/any of one/more then one 
estimator or cofficient. or also we can say
'''
print(res.fvalue,res.f_pvalue)
print(res.df_model, res.df_resid)
print('f value: ',res.mse_model/res.mse_resid)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
print('pvalue of f score:',stats.f.sf(res.fvalue, dfn = res.df_model,dfd = res.df_resid))


'''
Significance level:
Calcualte Fα,k,n-k-1
where α = 0.05 (95%) 0r 0.01 (99%)
k = num of independent variable = res.df_model
n = num of obs = res.nobs
remember n-k-1 = res.df_resid
'''
print('Significance OR CRITICAL levell: ',f.isf(q = 0.05, dfn = res.df_model,dfd = res.df_resid, loc=0, scale=1))
print('Significance OR CRITICAL level: ',f.isf(q = 0.05, dfn = 1,dfd = 22, loc=0, scale=1))
print('Significance OR CRITICAL level: ',f.ppf(1 - 0.05, dfn = 1,dfd = 22, loc=0, scale=1))
'''
f val: 261.2351086605637 
f_p val: 4.687422207749737e-16
pvalue of f score :  4.687422207749737e-16
Significance level:  3.44335677936672
so f val >> f_p val.
So Rejected null hypothesis.
We can conclude DlvrTime_Y dependent on
NumCase_X1 and/or Dst_X2.
Note: To reject null hypopthesis its also importent
f val >> Significance level
'''
# Genral linear hypothesis test i.e. Tβ = 0
# test pairwise equality of some coefficients
'''
H0(Null hypothesis) : Tβ = 0 or Tβ = c
f val >> f_p val rejected null hypothesis 
y = β0 +  β1x1 +  β2x2 + β3x3 + ϵ
Ex1:
H0: β1 = β3 -> β1-β2 = 0 So we can write
T/R = [0,1,0,-1]
Ftest = res.f_test(T/R)
if Ftest.fvalue >> Ftest.pvalue then  rejected Null hypthesis.
That is β1 ≠ β3. 

Ex2:
Let null hypothesis H0 ->
β1 = β3 and β2=0
T = [0 1 0 -1]
    [0 0 1  0]
if Ftest.fvalue >> Ftest.pvalue then  rejected Null hypthesis.
if Ftest.fvalue << Ftest.pvalue then fail to rejected null hypothesis.
In this case we can conclude β1 = β2 and linearly dependent. Need to further
test and β2 = 0 do not play any significance rol in models.

Ex3:

Let null hypothesis H0 ->
β1 = 0 (β1 play no significance rol in models)
Hα : β1 ≠ 0 (play sinificance roal in models)
T = [0 1 0 0]
if Ftest.fvalue >> Ftest.pvalue then  rejected Null hypthesis.
Conclude β1 play sihnificance roal in models

Ex4:

Let null hypothesis H0 ->
β2 = 0 (β2 play no significance rol in models)
Hα : β2 ≠ 0 (play sinificance roal in models)
T = [0 0 1 0]
if Ftest.fvalue >> Ftest.pvalue then  rejected Null hypthesis.
Conclude β2 play sihnificance roal in models
'''
