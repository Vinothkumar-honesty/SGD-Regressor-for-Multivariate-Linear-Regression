# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load California housing data and select features for input X and output Y.
2. Split data into train/test sets and scale using StandardScaler.
3. Initialize SGDRegressor with MultiOutputRegressor and fit on training data.
4. Predict on test data, inverse-transform results, and compute mean squared error.



## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:  VINOTHKUMAR R
RegisterNumber:  212224040361
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
```
```
X=df.drop(columns=['AveOccup','HousingPrice'])
Y=df[['AveOccup','HousingPrice']]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train = scaler_X.fit_transform(X_train) 
X_test = scaler_X.transform(X_test) 
Y_train = scaler_Y.fit_transform(Y_train) 
Y_test = scaler_Y.transform(Y_test)
sgd =  SGDRegressor(max_iter=1000, tol=1e-3) 

multi_output_sgd= MultiOutputRegressor(sgd) 
multi_output_sgd.fit(X_train, Y_train) 

Y_pred=multi_output_sgd.predict(X_test) 

Y_test= scaler_Y.inverse_transform(Y_pred)  
 
mse= mean_squared_error (Y_test, Y_pred) 
print("Mean Squared Error:", mse) 

print("\nPredictions: \n",Y_pred[:5])
```

## Output:

![1](https://github.com/user-attachments/assets/0e3369f5-411d-4238-a648-57497aa3f15a)

![2](https://github.com/user-attachments/assets/2395f7db-6269-4219-afe2-32fd983c17ed)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
