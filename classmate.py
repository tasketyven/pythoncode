
#%%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller
import math
from sklearn.linear_model import LinearRegression
import sys
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from mcs import ModelConfidenceSet

class Proces:

    def __init__(self, y):
        self.y = y

    def season(self):
        pi = math.pi
        t = np.arange(1,len(self.y)+1)
        year = (pi*2*t)/(365)
        month = (pi*2*t)/(30)
        week = (pi*2*t)/(7)
    
        sinweek = [math.sin(x) for x in week]
        cosweek = [math.cos(x) for x in week]
        sinmonth = [math.sin(x) for x in month]
        cosmonth = [math.sin(x) for x in month]
        sinyear = [math.sin(x) for x in year]
        cosyear = [math.cos(x) for x in year]

        dummies = np.array([sinweek,cosweek,sinmonth,cosmonth,sinyear,cosyear,t],dtype=object).T

        seas = LinearRegression().fit(dummies, self.y).predict(dummies)

        yclean = self.y - seas

        self.stationarity(yclean)

        return yclean, seas

    def stationarity(self,y):
        adf = adfuller(y,regression='nc')[1]
        pp = PhillipsPerron(y,trend='nc').pvalue

        if adf < 0.05 and pp < 0.05:
            print("Data is stationary")
        else:
            sys.exit("Data not stationary")     

def splitFeat(df,testlength):

    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['lag1'] = df['SYS'].shift(1)
    df['lag2'] = df['SYS'].shift(2)
    df['lag3'] = df['SYS'].shift(3)
    df['lag7'] = df['SYS'].shift(7)

    df.dropna(inplace=True)

    split = int(len(df)-testlength)

    dftest = df[split:].copy()
    dftrain = df[:split].copy()

    return dftrain, dftest


def splitvar(df):

    ytrue = df['SYS']

    seas = df['seas']

    y = df['yclean']
    
    x = df.drop(['SYS','yclean','seas'],axis=1)

    return ytrue, y, x, seas  


class Model:

    def __init__(self,name,estimator,univariate):
        self.name = name
        self.estimator = estimator
        self.univariate = univariate


class Estimation:

    def __init__(self,y_true_train, y_train, x_train, seas_train,y_true_test, y_test, x_test, seas_test):
        self.y_true_train = y_true_train
        self.y_train = y_train
        self.x_train = x_train
        self.seas_train = seas_train
        self.y_true_test = y_true_test
        self.y_test = y_test
        self.x_test = x_test
        self.seas_test = seas_test

        self.models = []
        self.predictions = pd.DataFrame()
        self.error = pd.DataFrame()
        
    def add_model(self,model):
        self.models.append(model)
        print(f"{model.name} added")
    
    def estimate(self):
        for i, model in enumerate(self.models):
            if model.univariate:
                predict = model.estimator.fit(disp=0,start_ar_lags=20).forecast(steps=365)[0]
            else:
                predict = model.estimator.fit(self.x_train,self.y_train).predict(self.x_test)
            self.predictions.insert(i, model.name, predict, True)
            print(f"Estimates based on {model.name} created")

        for model in self.predictions.columns:
            self.error[model] = np.abs(self.predictions[model] - self.y_test.values)
            print(f"{model} AE calculated")

        mcs = ModelConfidenceSet(self.error,0.05,5, 1000).run()

        finalDF = self.predictions[mcs.included]

        #Get final predictor, mean if multiple final models
        predictor = np.add(finalDF.mean(axis=1),self.seas_test.values) 

        mae = mean_absolute_error(self.y_true_test,predictor)

        print(f"Final model(s): {mcs.included} with mae of {mae}")

        self.plot_final(predictor,self.y_true_test)

    def plot_final(self,pred,true):
        plt.plot(pred,linewidth=1,color ='r')
        plt.plot(true.values,linewidth=1)
        plt.title('Prediction and daily System Price')
        plt.legend(['Prediction','System Price'])        
        plt.xlabel('Days forecasted ahead')
        plt.ylabel('System Price')
        plt.show()