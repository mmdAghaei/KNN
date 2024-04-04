# Import Package
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# Read Dataset
df = pd.read_csv("../DataSet/diabetes.csv")
# Variable value
x = df.drop("Outcome",axis=1)
y = df["Outcome"]
# Convert to array
x = np.array(x)
y = np.array(y)
# Normalize data
scaler = StandardScaler()
x = scaler.fit_transform(x)
# Split Train and Test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.1)
# KNN
KNN = KNeighborsClassifier(n_neighbors=8)
KNN.fit(x_train,y_train)
# Predict
y_predict_train = KNN.predict(x_train)
y_predict_test = KNN.predict(x_test)
# Accuracy
acc_train = accuracy_score(y_train,y_predict_train)
acc_test = accuracy_score(y_test,y_predict_test)
# Confuison
cnm = confusion_matrix(y_test,y_predict_test)
# Precision
p = precision_score(y_test,y_predict_test)
# Recall
r = recall_score(y_test,y_predict_test)
# Print
print("-----------------------------------------")
print(f"Accuracy Train=>{acc_train}\nAccuracy Test=>{acc_test}\nConfuison Test\n{cnm}\nPrecision Test=>{p}\nRecall=>{r}")
print("-----------------------------------------")