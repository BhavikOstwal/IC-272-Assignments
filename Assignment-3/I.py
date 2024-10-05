import pandas as pd
import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

train_df = pd.read_csv("iris_train.csv", index_col=0)
test_df = pd.read_csv("iris_test.csv", index_col=0)

# Applying PCA to reduce dimension to one
### Removing outliers first
train_df_X = train_df.iloc[:,:-1]
Q1 = train_df_X.quantile(0.25)
Q3 = train_df_X.quantile(0.75)
IQR = Q3 - Q1
newdf = np.where((train_df_X <= (Q1 - 1.5*IQR)) | (train_df_X >= (Q3 +1.5*IQR)), train_df_X.median(),train_df_X)
train_df.iloc(axis=1)[:-1] = newdf
# train_df.boxplot()
# plt.show()

# Applying PCA
def PCA(t_df,dim:int, training_df = train_df):
    X = t_df.iloc[:,:-1]
    C = X.cov()
    eign_val = np.linalg.eig(C).eigenvalues
    eign_vec = np.linalg.eig(C).eigenvectors
    order_ind = np.argsort(eign_val)[::-1]
    d_eign_vec = eign_vec[:,order_ind][:,:dim]
    X_tra_proj = np.matmul(X,d_eign_vec)
    train_df_proj = pd.DataFrame()
    train_df_proj['proj1'] = X_tra_proj
    train_df_proj['Species'] = t_df['Species']
    return train_df_proj

train_proj = PCA(train_df, 1)
test_proj = PCA(test_df,1)
no_of_classes = len(train_df['Species'].value_counts()) # = 3

# Training Bayes Classifier with help of my own classifier
class Bayes_classifier:
    def __init__(self,train_proj:pd.DataFrame,train_df=train_df, univariate:bool = True):
        self.tr_proj = train_proj
        self.variate = univariate
        self.n_col = len(train_df.iloc[:,:-1].columns) # here  = 4
        self.n_c = len(train_df['Species'].value_counts()) # here = 3
        self.class_prob = (train_df['Species'].value_counts(sort=False) / sum(train_df['Species'].value_counts())).to_numpy().reshape((self.n_c,))
    
    def statistics(self):
        if (self.variate==True):
            uni_means = self.tr_proj.groupby('Species',sort=False).mean().to_numpy().reshape((self.n_c,))
            uni_stds = self.tr_proj.groupby('Species',sort=False).std(ddof=0).to_numpy().reshape((self.n_c,))
            return uni_means, uni_stds
        else:
            multi_means = train_df.groupby('Species',sort=False).mean().to_numpy()
            multi_stds = train_df.groupby('Species',sort=False).cov(ddof=0).to_numpy().reshape((self.n_c,self.n_col,self.n_col)) # reshaped to (3,4,4) so that I can access cov matrices by indices
            return multi_means, multi_stds
    
    def L(self,x, i:int):
        mu, sigma = self.statistics()
        mu = mu[i]
        sigma = sigma[i]
        class_prob = self.class_prob[i]
        if self.variate:
            P = np.exp(-(((x-mu)/sigma)**2)/2) / (np.sqrt(2*np.pi) * sigma)            
            return  P * class_prob
        else:
            dist = np.matmul(x-mu, inv(sigma))
            dist = np.matmul(dist,(x-mu).T)
            dist = -dist/2
            P = np.exp(dist) / (pow(2*np.pi,self.n_col/2) * det(sigma)**0.5)
            return P * class_prob

    def Predict(self,x):
        likelihoods = []
        for i in range(self.n_c):
            likelihoods.append(self.L(x,i))
        return np.argsort(likelihoods)[-1]


## Ignore my Bayes clf testing
mybayes = Bayes_classifier(train_proj)
mybayes.L(6.464326,1)

## **Testing Time 😎**
classes_lst = list(train_df.groupby(sort=False, by='Species').groups.keys())
X_test_uni = test_proj['proj1'] #removing class labels
Y_pred_uni = []
myclf = Bayes_classifier(train_proj)
for i in X_test_uni:
    Y_pred_uni.append(classes_lst[myclf.Predict(i)])

## **Evaluation Time**
ConfusionMatrixDisplay.from_predictions(test_df['Species'], Y_pred_uni)
acc_uni = accuracy_score(test_df['Species'], Y_pred_uni)*100
if __name__ == '__main__':
    print(f"Accuracy: {acc_uni:.2f}%")
    plt.savefig("PART1_CM.png")
    plt.show()
    