import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Linear_Regression():
    def __init__(self):
        self.__m = 0 # y = m*x + n
        self.__n = 0 # n = y - m*n

    # Not using now maybe later ----------------------------------------------
    def Ordinary_Least_Squares(self,x,y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        Cov_xy = np.sum((x - mean_x) * (y - mean_y)) / np.size(x)
        Var_x  = np.sum((x - mean_x) * 2) / np.size(x)
        
        self.__m = Cov_xy / Var_x
        self.__n = np.mean(y) - self.__m * np.mean(x)
    # -------------------------------------------------------------------------
    def SOS_Error_m(self,x,y,m,n):
        m_slope = np.sum((-2 * x) * (y - (m * x + n))) / np.size(x)
        return m_slope
    
    def SOS_Error_n(self,x,y,m,n):
        n_slope = np.sum(-2 * (y - (m * x + n))) / np.size(x)
        return n_slope
    
    def Gradient_Descent(self,x,y,m = 0,n = 0,learn_rate = .001,iter = 100):
        for i in range(iter):
            m = m - (self.SOS_Error_m(x,y,m,n) * learn_rate)
            n = n - (self.SOS_Error_n(x,y,m,n) * learn_rate)
        self.__m = m
        self.__n = n
    
    def fit(self,x,y):
        self.Gradient_Descent(x,y)
    
    def predict(self,x):
        prediction = (self.__m * x) + self.__n
        return prediction


#Importing the dataset
dataset=pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=1/3, random_state = 0)


#Fitting Simple Linear Regression to the Training set
regressor=Linear_Regression()
regressor.fit(X_train,Y_train)

#Predicting the Test set result
#Y_pred --> salary X_set --> work years
Y_pred=regressor.predict(X_test)

#Visualising the Training set
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
        
        
        
        
        