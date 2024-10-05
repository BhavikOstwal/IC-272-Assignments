import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Iris.csv")

'''Part (a)'''
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


'''Part (b)'''
plt.style.use('dark_background')
plt.boxplot(X)
# plt.show() #Got outlier in 2nd column, so let's fix that


out_col = X.iloc[:,1]
Q1 = out_col.quantile(0.25)
Q3 = out_col.quantile(0.75)
IQR = Q3-Q1

out_numbers = out_col.loc[(out_col<Q1 - 1.5*IQR) | (out_col>Q3+1.5*IQR)]
# print(out_numbers.info())
replaced_col = np.where((out_col<Q1 - 1.5*IQR) | (out_col>Q3+1.5*IQR), out_col.median(), out_col)

X['SepalWidthCm'] = pd.Series(replaced_col) #updated the original one
plt.boxplot(X)
# plt.show()


'''Part (c)'''
# Appling PCA algo
X_0mean = X - X.mean()
# X.cov()
C = np.matmul(X_0mean.T,X_0mean)
C # but real covvariance matrix is C/149
eign_vec = np.linalg.eig(C).eigenvectors
eign_val = np.linalg.eig(C).eigenvalues
# print(eign_vec)
# print(eign_val)

np.dot(C,eign_vec[:,0].reshape((4,1)))
order_ind = np.argsort(eign_val)[::-1] 
d_eig_vec = eign_vec[:, order_ind][:,:2] #took the max eigenval's eignvectors

X_proj = np.matmul(X, d_eig_vec)
# print(X_proj)


'''Part (d)'''

plt.close()
plt.scatter(X_proj.iloc[:,0], X_proj.iloc[:,1], color = 'white')
plt.title("Scatter plot of 2-dimensional datapoints")
plt.xlabel("C1")
plt.ylabel("C2")

dp_00 = np.dot(d_eig_vec[:,0], d_eig_vec[:,0])
dp_01 = np.dot(d_eig_vec[:,0], d_eig_vec[:,1])
dp_10 = np.dot(d_eig_vec[:,1], d_eig_vec[:,0])
dp_11 = np.dot(d_eig_vec[:,1], d_eig_vec[:,1])
plt.quiver(X_proj.iloc[:,0].mean(), X_proj.iloc[:,1].mean(),  dp_00,dp_01 , scale_units='x', scale=2, color='red',alpha=0.6)
plt.quiver(X_proj.iloc[:,0].mean(), X_proj.iloc[:,1].mean(), dp_10 ,dp_11 , scale_units='x', scale=2, color='red',alpha=0.6)
# Note: Not sure, why to project directions of eigenvectors: on standard x-y axes or on plane that contains both eigenvectors and considering those eigenvectors as standard axes

if __name__ == "__main__":
     plt.savefig("scatter.png")
     # plt.show()
     plt.close()


'''Part (e)'''
X_reverted =np.matmul(X_proj,d_eig_vec.T)
X_reverted.columns = X.columns[:4]
# print(X_reverted)

'''Part (f)'''
RMSE = []
for i in range(len(X.columns)):
    # sum = np.dot(((X_reverted.iloc(axis=1)[i] - X.iloc(axis=1)[i]).to_numpy())**2, np.ones((150,)))
    sum = (X_reverted.iloc(axis=1)[i].to_numpy() - X.iloc(axis=1)[i].to_numpy())**2
    # print(sum)
    sum = np.mean(sum)
    # RMSE.append(sum/150)    
    RMSE.append(sum**0.5)    

print(RMSE)
# plt.bar(X.columns,RMSE, color= 'white')
# plt.title("RMSE values b/w reverted back and original data")
# plt.xlabel("Columns")
# plt.ylabel("RMSE vals")
# plt.savefig('RMSE.png')
# if __name__ == "__main__":
#      # plt.show()
#      plt.close()

