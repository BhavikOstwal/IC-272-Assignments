import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats
from I import X_proj, y


df_proj = X_proj
# df_proj.columns = X.columns[:2]
df_proj['Species'] = y
# print(df_proj)

X_train, X_test, y_train, y_test = train_test_split(X_proj, y, random_state=104, test_size=0.2, shuffle=True)

def Mode(arr:np.array):
    unique, counts = np.unique(arr,return_counts=True)
    return unique[np.argmax(counts)]


def KNN():
    y_pred = []
    for i in range(len(X_test.index)):
        mypt = X_test.iloc[i].to_numpy()[:2]
        # print(mypt)
        dist_lst = []
        for j in range(len(X_train.index)):
            t_pt = X_train.iloc[j].to_numpy()[:2]
            dist = (mypt - t_pt)**2
            dist = np.sqrt(np.sum(dist))
            dist_lst.append([dist,j])
        dist_lst = (np.array(sorted(dist_lst)))[:5]
        labels_lst = y_train.iloc[dist_lst[:,1]].to_numpy()
        # print(dist_lst)
        y_pred.append(Mode(labels_lst))
        # print(labels_lst)
        # print(dist_lst)
    return y_pred

y_pred = KNN()


y_test.unique()
# print(X_test)

predicted= X_test.iloc[:,:-1]
predicted.insert(2,"species_pred" ,y_pred)
# print(predicted)

'''Part (b)'''
cm = confusion_matrix(y_test, y_pred)
plt.close()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig("cm.png")
plt.show()
