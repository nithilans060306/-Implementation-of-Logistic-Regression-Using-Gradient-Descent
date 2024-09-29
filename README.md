# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step 1: Start
### Step 2: Load the Dataset
### Step 3 Preprocess the Data
### Step 4: Split the Data
### Step 5: Build and Train the Model
### Step 6: Evaluate the Model
### Step 7: Predict New Data
### Step 8: Stop

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Nithilan S
RegisterNumber:  212223240108
```
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Placement_Data.csv")
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
theta = np.random.randn(X.shape[1])
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha = 0.01,num_iterations = 1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
Accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy: ",Accuracy)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
##  Output:
#### Dataset:
![image](https://github.com/user-attachments/assets/0f438418-4a04-4c24-beb6-3f344e96f59e)
#### Type of attributes:
![image](https://github.com/user-attachments/assets/a9a93573-c953-4f40-a419-472393f77542)
#### Accuracy
![image](https://github.com/user-attachments/assets/1bc5b7ff-41ba-4ea4-9974-b55bbc6b58ee)
![image](https://github.com/user-attachments/assets/9d221c4c-530d-4a54-8959-3432d4a1de06)
![image](https://github.com/user-attachments/assets/0ade8171-840a-4217-b9fe-88bf60927095)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
