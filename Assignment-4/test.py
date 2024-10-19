import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
df = pd.read_csv("abalone.csv")
df
df.corr()['Rings'].sort_values(ascending=False)
from sklearn.model_selection import train_test_split
# I

features = df.drop(['Rings'],axis=1)
target = df['Rings']
train_df, test_df = train_test_split(df,test_size=0.3, random_state=42)
train_df.to_csv("abalone_train.csv",index=False)
test_df.to_csv("abalone_test.csv",index=False)
# II
df.corr(method='pearson')['Rings'].sort_values(ascending=False)[1:]

train_df_hc = pd.read_csv("abalone_train.csv")
X_train_hc = df['Shell weight'].to_numpy().reshape((-1,1))
y_train_hc = df['Rings'].to_numpy().reshape((-1,1))
bias = np.ones(shape=(len(X_train_hc),1))
bias
X_train_hc = np.concatenate((bias,X_train_hc),axis=1)
X_train_hc
weigths_hc = inv(X_train_hc.T @ X_train_hc) @ X_train_hc.T @ y_train_hc
weigths_hc
y_pred = X_train_hc @ weigths_hc
y_pred
plt.style.use('dark_background')
plt.scatter(X_train_hc[:,1], y_train_hc, color='white', marker='o')
plt.plot(X_train_hc[:,1], y_pred, color='r')
plt.grid(alpha=0.5) 
plt.show()
y_train_hc
y_pred
def RMSE_percentage(y_true, y_pred):
    mysum = 0
    for i,j in zip(y_true,y_pred):
        # print(mysum)
        mysum += (i-j)**2
    mysum /= len(y_true)
    mysum = mysum**0.5
    mysum /= np.mean(y_true)
    return mysum
RMSE_percentage(y_train_hc, y_pred)
test_df_hc = pd.read_csv("abalone_test.csv")
