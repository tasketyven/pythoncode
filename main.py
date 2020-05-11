#%%
import pandas as pd
import os
from classmate import Proces, splitFeat, splitvar, Estimation, Model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima_model import ARMA
from xgboost import XGBRegressor
import warnings 
warnings.filterwarnings('ignore') 

os.chdir('C:/Users/Thomas/Google Drive/py/electricity')

df = pd.read_excel("datdaily.xlsx",sheet_name="Sheet1", index_col='Datetime', parse_dates=['Datetime'])

df = df.resample('D').sum() 

print(df.head(5))

print(df.describe())

#%%
sys = df['SYS']

df['yclean'], df['seas'] = Proces(sys).season()

dftrain, dftest = splitFeat(df,365)

y_true_train, y_train, x_train, seas_train = splitvar(dftrain)
y_true_test, y_test, x_test, seas_test = splitvar(dftest)

estimator = Estimation(y_true_train, y_train, x_train, seas_train,y_true_test, y_test, x_test, seas_test)

rf = Model("Random Forest",RandomForestRegressor(n_estimators=100),False)
gbm = Model("Gradient Tree Boosting",GradientBoostingRegressor(n_estimators=100),False)
xg = Model("XGBoost",XGBRegressor(n_estimators=100),False)
arma = Model("ARMA(7,7)",ARMA(y_train,order=(7,7)),True)

estimator.add_model(rf)
estimator.add_model(gbm)
estimator.add_model(xg)
estimator.add_model(arma)

estimator.estimate()

