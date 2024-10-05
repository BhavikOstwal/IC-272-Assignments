import pandas as pd
import matplotlib.pyplot as plt
from I_1 import Statistic

ls_cleaned = pd.read_csv("ls_clean_intp.csv")

Q1 = ls_cleaned.iloc[:,2:].quantile(0.25)
Q3 = ls_cleaned.iloc[:,2:].quantile(0.75)

outliers_df = ls_cleaned[(ls_cleaned.iloc(axis=1)[2:] <= Q1 - 1.5*(Q3-Q1)) | (ls_cleaned.iloc(axis=1)[2:] >= Q3 + 1.5*(Q3-Q1))]


def median_forthis(df:pd.DataFrame):
    copydf = df.copy()
    for i in copydf.columns[2:]:
        med = Statistic("ls_clean_intp.csv", i).med_()
        mycol = ls_cleaned[outliers_df.isna()][i].fillna(med)
        copydf[i] = mycol
    return copydf

new_without_outliers = median_forthis(ls_cleaned)

plt.style.use('dark_background')
plt.figure(figsize=(8,7))
plt.boxplot(new_without_outliers.iloc(axis=1)[2:], vert=True)
plt.grid(alpha=0.4)
plt.show()

# plt.boxplot(new_without_outliers['rain'])
# plt.show()
new_without_outliers.to_csv("without_outliers.csv",index=False)