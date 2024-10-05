import pandas as pd
import matplotlib.pyplot as plt

ls_cleaned = pd.read_csv("ls_clean_intp.csv")

Q1 = ls_cleaned.iloc[:,2:].quantile(0.25)
Q3 = ls_cleaned.iloc[:,2:].quantile(0.75)

outliers_df = ls_cleaned[(ls_cleaned.iloc(axis=1)[2:] <= Q1 - 1.5*(Q3-Q1)) | (ls_cleaned.iloc(axis=1)[2:] >= Q3 + 1.5*(Q3-Q1))]

# print(outliers_df['rain'].dropna())

for i in ls_cleaned.columns:
    print(f"{i}: {outliers_df[i].dropna().to_list()}")
    print()


plt.style.use('dark_background')
plt.figure(figsize=(8,7))
plt.boxplot(ls_cleaned.iloc(axis=1)[2:],labels=ls_cleaned.columns[2:], vert=True)
plt.title("BoxPlots for various attributes")
plt.xlabel("Attributes")
plt.grid(alpha=0.4)
plt.show()