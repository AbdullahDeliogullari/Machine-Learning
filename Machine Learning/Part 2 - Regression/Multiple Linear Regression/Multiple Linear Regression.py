import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X=X[:,1:]

#Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=1/3, random_state = 0)

#Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
#B0.X0 --> X0=1 so we should add 1 to X
#Remove the highest p valuse from X_opt
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()

#Backward Elimination with p values only

#import statsmodels.formula.api as sm
#def backwardElimination(x, sl):
#    numVars = len(x[0])
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        if maxVar > sl:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    x = np.delete(x, j, 1)
#    regressor_OLS.summary()
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)

#Backward elimination with p values and adjused R squared

#import statsmodels.formula.api as sm
#def backwardElimination(x, SL):
#    numVars = len(x[0])
#    temp = np.zeros((50,6)).astype(int)
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        adjR_before = regressor_OLS.rsquared_adj.astype(float)
#        if maxVar > SL:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    temp[:,j] = x[:, j]
#                    x = np.delete(x, j, 1)
#                    tmp_regressor = sm.OLS(y, x).fit()
#                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                    if (adjR_before >= adjR_after):
#                        x_rollback = np.hstack((x, temp[:,[0,j]]))
#                        x_rollback = np.delete(x_rollback, j, 1)
#                        print (regressor_OLS.summary())
#                        return x_rollback
#                    else:
#                        continue
#    regressor_OLS.summary()
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)






