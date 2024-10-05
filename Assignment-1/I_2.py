import pandas as pd
import numpy as np

landslide = pd.read_csv("landslide_data_original.csv")

landslide_new = landslide.drop(labels=["dates", "stationid"], axis=1)
landslide_new = landslide_new[:len(landslide_new.columns)]
landslide_new = pd.DataFrame(index=landslide_new.columns, columns=landslide_new.columns, dtype='float64')

def Pearson(i,j):
    attr1 = landslide.iloc[:,i].to_numpy()
    attr2 = landslide.iloc[:,j].to_numpy()
    mean1 = np.mean(attr1)
    mean2 = np.mean(attr2)
    if (i==j): return 1

    else:
        numera =  sum((attr1 - mean1)*(attr2-mean2))
        denom1 = (sum((attr1-mean1)**2))**0.5
        denom2 = (sum((attr2-mean2)**2))**0.5
        return numera/(denom1*denom2)

for i in range(len(landslide_new.columns)):
    for j in range(len(landslide_new.columns)):
        landslide_new.iloc[i,j] = Pearson(i+2,j+2)

print(landslide_new)

print()
print("List of redundand attribute with respect to \"lightavg\":", 
      [i for i in landslide_new[landslide_new['lightavg']>0.6].index if i!='lightavg'])

