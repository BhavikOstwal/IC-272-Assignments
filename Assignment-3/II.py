import pandas as pd
import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from I import Bayes_classifier, classes_lst, train_df, test_df

# Testing Time 😎
X_test_multi = test_df.iloc[:,:-1].to_numpy()

Y_pred_multi = []
myclf = Bayes_classifier(train_df,univariate=False)
for i in X_test_multi:
    Y_pred_multi.append(classes_lst[myclf.Predict(i)])

plt.close()
ConfusionMatrixDisplay.from_predictions(test_df['Species'], Y_pred_multi)
acc_multi = accuracy_score(test_df['Species'], Y_pred_multi)*100
print(f"Accuracy: {acc_multi:.2f}%")
# plt.savefig("PART2_CM.png")
plt.show()

# Part III is done in last of jupyter notebook file: "All_scrapbook.ipynb"