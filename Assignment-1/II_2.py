import pandas as pd
import matplotlib.pyplot as plt
from I_1 import Statistic

landslide = pd.read_csv("landslide_data_original.csv")
ls_cleaned = pd.read_csv("missing_fix_data.csv")


'''Part (a)'''

def interpolate(ser:pd.Series):
    copy_ser = ser.copy()
    null_indices = copy_ser.index[copy_ser.isna()]
    # print(null_indices)

    for ind in null_indices:
        if ind == 0 or ind == len(copy_ser)-1:
            return
        
        lower = ind-1
        upper = ind+1
        while upper<len(copy_ser)-1 and pd.isna(copy_ser[upper]):
            upper+=1
        
        while lower>0 and pd.isna(copy_ser[lower]):
            lower-=1

        if lower>=0 and upper<=len(copy_ser)-1:
            copy_ser.loc[ind] = ((copy_ser[lower]) + (copy_ser[upper]))/2

    return copy_ser

# interpolate(ls_cleaned['temperature'])

for i in ls_cleaned.columns[2:]:
    ls_cleaned[i] = interpolate(ls_cleaned[i])


print(ls_cleaned.info())

ls_cleaned.to_csv("ls_clean_intp.csv",index=False)

mean_dict = dict({
    "Interploated" : [Statistic("ls_clean_intp.csv", i).mean_() for i in ls_cleaned.columns[2:]],
    "Original" : [Statistic("landslide_data_original.csv", i).mean_() for i in ls_cleaned.columns[2:]]
})

med_dict = dict({
    "Interploated" : [Statistic("ls_clean_intp.csv", i).med_() for i in ls_cleaned.columns[2:]],
    "Original" : [Statistic("landslide_data_original.csv", i).med_() for i in ls_cleaned.columns[2:]]
})

std_dict = dict({
    "Interploated" : [Statistic("ls_clean_intp.csv", i).std_() for i in ls_cleaned.columns[2:]],
    "Original" : [Statistic("landslide_data_original.csv", i).std_() for i in ls_cleaned.columns[2:]]
})


a =pd.DataFrame(mean_dict)
b = pd.DataFrame(med_dict)
c = pd.DataFrame(std_dict)

print(a,"\n")    #mean comparison
print(b, "\n")    #median comparison
print(c, "\n")    #std comparision

# ls_cleaned.info()


'''Part (b)'''

# To filter landslide data to get equal rows
landslide['key'] = landslide['dates'].astype(str) + '_' + landslide['stationid'].astype(str)
ls_cleaned['key'] = ls_cleaned['dates'].astype(str) + '_' + ls_cleaned['stationid'].astype(str)

landslide = landslide[landslide['key'].isin(ls_cleaned['key'])]

landslide = landslide.drop(columns=['key'])
ls_cleaned = ls_cleaned.drop(columns=['key'])
# filteration done


def RSME(ser1:pd.Series, ser2:pd.Series):
    ser1 = ser1.to_numpy()
    ser2 = ser2.to_numpy()
    rmse = (sum((ser1-ser2)**2) / (len(ser1)))**0.5
    return rmse   
     
rmse_vals = []
for i in ls_cleaned.columns[2:]:
    # print(i)
    rmse_vals.append(RSME(ls_cleaned[i], landslide[i]))

# print(rmse_vals)
plt.style.use('dark_background')
plt.figure(figsize=(8,7))
plt.bar(ls_cleaned.columns[2:], rmse_vals, color='white')
plt.title("RMSE b/w original and replaced values for each attribute")
# plt.ylim(0,100)
plt.xticks(rotation=25)
plt.show()