import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
df = pd.read_csv("abalone.csv")
df
df.corr()['Rings'].sort_values(ascending=False)
from sklearn.model_selection import train_test_split
# I
# **Train-test Split**
features = df.drop(['Rings'],axis=1)
target = df['Rings']
train_df, test_df = train_test_split(df,test_size=0.3, random_state=42)
train_df.to_csv("abalone_train.csv",index=False)
test_df.to_csv("abalone_test.csv",index=False)
# II
(train_df.corr(method='pearson')['Rings'].sort_values(ascending=False))
hc_feat = (train_df.corr(method='pearson')['Rings'].sort_values(ascending=False)).index[1]
hc_feat
#  *As we can see from above Series, the attribute **'Shell weight'** is having the highest pearson's correlation value with the target variable **'Rings'**.*
train_df_hc = pd.read_csv("abalone_train.csv")
X_train_hc = train_df[hc_feat].to_numpy().reshape((-1,1))
y_train_hc = train_df['Rings'].to_numpy().reshape((-1,1))
bias = np.ones(shape=(len(X_train_hc),1))
bias
X_train_hc = np.concatenate((bias,X_train_hc),axis=1)
X_train_hc
weigths_hc = inv(X_train_hc.T @ X_train_hc) @ X_train_hc.T @ y_train_hc
weigths_hc
y_train_pred = X_train_hc @ weigths_hc
y_train_pred
plt.style.use('dark_background')
plt.scatter(X_train_hc[:,1], y_train_hc, color='white', marker='o',edgecolors='blue')
plt.plot(X_train_hc[:,1], y_train_pred, color='r')
plt.grid(alpha=0.5) 
plt.show()
y_train_hc
y_train_pred
def RMSE_percentage(y_true, y_pred):
    mysum = 0
    for i,j in zip(y_true,y_pred):
        # print(mysum)
        mysum += (i-j)**2
    mysum /= len(y_true)
    mysum = mysum**0.5
    # mysum /= np.mean(y_true)
    return mysum
RMSE_percentage(y_train_hc, y_train_pred)
X_train_hc
# II (b)
test_df_hc = pd.read_csv("abalone_test.csv")
X_test_hc = test_df['Shell weight'].to_numpy().reshape((-1,1))
y_test_hc = test_df['Rings'].to_numpy().reshape((-1,1))
X_test_hc
test_bias = np.ones(shape=(len(X_test_hc),1))
test_bias
X_test_hc = np.concatenate((test_bias,X_test_hc),axis=1)
X_test_hc
y_test_pred = X_test_hc @ weigths_hc
y_test_pred
plt.style.use('dark_background')
plt.scatter(X_test_hc[:,1], y_test_hc, color='white', marker='o', edgecolors='blue')
plt.plot(X_test_hc[:,1], y_test_pred, color='r')
plt.grid(alpha=0.5) 
plt.show()
RMSE_percentage(y_test_hc, y_test_pred)
# II (c)
# plt.style.use('dark_background')
plt.scatter(y_test_hc, y_test_pred, color='white', marker='o',edgecolors='blue')
plt.grid(alpha=0.3)
plt.show()
# III
X_train_hc
def Poly_regres(p,for_pred, X=X_train_hc , y=y_train_hc):
    for i in range(2,p+1):
        new_col = X[:,1].reshape((-1,1))**i
        new_col_1 = for_pred[:,1].reshape((-1,1))**i
        # print(len(new_col), len(X))
        X = np.concatenate((X,new_col),axis=1)
        for_pred = np.concatenate((for_pred,new_col_1),axis=1)
    weigths = ((inv((X.T) @ X)) @ (X.T)) @ y
    ans = for_pred @ weigths
    return ans
X_train_hc[:,1].reshape((-1,1))**2
Poly_regres(3,X_train_hc)
def plot_all(which, y_true):
    err_lst = []
    degree_lst = []
    for i in range(2,6):
        degree_lst.append(f'{i}')
        pred = Poly_regres(i,which)
        # print((RMSE_percentage(y_true,pred)))
        err_lst.append(float(RMSE_percentage(y_true,pred)[0]))
    # plt.style.use('dark_background')
    plt.bar(degree_lst,err_lst,color='white')
    plt.xlabel("Degrees")
    plt.ylabel("RMSE values")
    plt.title("Bar graph of RMSE values v/s degree of polynomial")
    plt.ylim(min(err_lst)-0.002,max(err_lst)+0.002)
    plt.show()
plot_all(X_train_hc,y_train_hc)
plot_all(X_test_hc,y_test_hc)
# III (c)
# **As we can see, p = 4 corresponds to the best-fit model as the value of RMSE for test set is the lowest for p = 4.**
# plt.style.use('dark_background')
sorted_arr = X_train_hc[X_train_hc[:, 1].argsort()]
curve_pts = Poly_regres(4,sorted_arr)
plt.scatter(X_train_hc[:,1], y_train_hc,color='black',edgecolors='white',alpha=0.8)
plt.plot(sorted_arr[:,1], curve_pts, color='red',linewidth=3)
plt.grid(alpha=0.3)
plt.show()
plt.style.available
# **Auto-regression**
asian_df = pd.read_csv("asianpaint.csv",index_col=0)
asian_df
# **Train-test Split**
asian_train = asian_df[:int(.65*len(asian_df))]
asian_test = asian_df[int(.65*len(asian_df)):]
# **Plotting both datasets**
asian_train.plot()
plt.show()
asian_test.plot()
plt.show()
# **Implement auto-regression**
X_as = asian_train.to_numpy()[:-1]
biases = np.ones(shape=(len(X_as),1))
X_as = np.concatenate((biases,X_as),axis=1)
y_as = asian_train.to_numpy()[1:]
weights = inv(X_as.T @ X_as) @ X_as.T @ y_as
weights
asian_train.index
plt.plot(asian_train.index[:-1],asian_train[1:],color='white',alpha=1,label='Actual')
plt.plot(asian_train.index[:-1],X_as @ weights,color='red',alpha=1,label='Predicted')
plt.legend()
plt.show()
X_test_as = asian_test.to_numpy()
biases = np.ones(shape=(len(X_test_as),1))
X_test_as = np.concatenate((biases,X_test_as),axis=1)
days_lst = asian_test.index.tolist() + ['1-05-2021']
days_lst
plt.plot(days_lst[1:], X_test_as@weights, color='red',label='Predicted')
plt.plot(asian_test.index[1:], asian_test[1:], color='white',label='Actual')
plt.legend()
plt.show()
# **Calculating RMSE & MAPE**/
def RMSE(y_true, y_pred):
    mysum = 0
    for i,j in zip(y_true,y_pred):
        # print(mysum)
        mysum += (i-j)**2
    mysum /= len(y_true)
    mysum = mysum**0.5
    return mysum
(np.mean((asian_test[1:].to_numpy() - (X_test_as@weights)[:-1])**2))**0.5
def MAPE_percent(y_true, y_pred):
    mysum = 0
    for i,j in zip(y_true,y_pred):
        mysum += (abs(i-j) / i)
    mysum /= len(y_true)
    return mysum*100
RMSE(asian_test[1:].to_numpy(), X_test_as@weights)
MAPE_percent(asian_test[1:].to_numpy(), X_test_as@weights)
