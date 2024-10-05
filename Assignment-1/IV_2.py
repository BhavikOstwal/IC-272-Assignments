import pandas as pd
import numpy as np

last_df = pd.read_csv("without_outliers.csv")
print(last_df.describe())

for i in last_df.columns[2:]:
    last_df[i] = (last_df[i] - np.mean(last_df[i]))/np.std(last_df[i])

print(last_df.describe())