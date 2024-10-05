import pandas as pd
import numpy as np

landslide = pd.read_csv("landslide_data_original.csv")

humidity = landslide[landslide['stationid'] == "t12"]['humidity']
import matplotlib.pyplot as plt

bins = np.arange(min(humidity)-5, max(humidity)+5, 5)

plt.style.use('dark_background')
plt.hist(humidity, bins=bins,edgecolor='black', color = 'white')
plt.xticks(bins)
plt.title("Histogram of Humidity values for Station ID: t12")
plt.xlabel("Humidity in %")
plt.ylabel("No. of Days at Specific Humidity %")
plt.show()