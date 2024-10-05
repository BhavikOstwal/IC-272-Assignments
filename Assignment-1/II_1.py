import pandas as pd

landslide_miss = pd.read_csv('landslide_data_miss.csv')
# landslide_miss.info()
landslide_cleaned = landslide_miss.dropna(subset=['stationid'])
no_of_Attr = landslide_cleaned.shape[1]
landslide_cleaned.dropna(thresh=(no_of_Attr/3)*2, inplace=True)

# landslide_cleaned.to_csv("landslide_cleaned.csv")
print(landslide_cleaned.info())
# print(landslide_cleaned.head())

landslide_cleaned.to_csv("missing_fix_data.csv", index=False)