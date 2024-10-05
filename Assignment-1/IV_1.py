import pandas as pd
import matplotlib.pyplot as plt

new_without_outliers = pd.read_csv("without_outliers.csv")

# new_without_outliers.info()
def min_max_norm(df:pd.DataFrame):
    copy_df = df.iloc[:,2:].copy()
    cols = copy_df.columns
    # print(cols)
    for i in cols:
        my_min= min(copy_df[i])
        my_max= max(copy_df[i])

        copy_df[i] = ((copy_df[i]-my_min)/(my_max-my_min)) * (7) + 5
    
    df.iloc(axis=1)[2:] = copy_df
    return df


removed_outl_df = new_without_outliers.copy()
removed_outl_df = min_max_norm(removed_outl_df)

print(removed_outl_df.describe()) #to find min and max valeus

plt.style.use('dark_background')
plt.boxplot(removed_outl_df.iloc[:, 2:])
plt.show()

# plt.boxplot(new_without_outliers.iloc[:, 2:])
# plt.show()