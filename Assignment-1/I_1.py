import pandas as pd

class Statistic():
    def __init__(self, dataname:str, seriesname) -> None:
        self.landslide = pd.read_csv(dataname)
        self.temp_arr = self.landslide[seriesname].to_numpy()
        self.length = len(self.temp_arr)   
    
    def mean_(self):
        mean = 0;
        for i in self.temp_arr:
            mean+=i
        mean /= self.length
        
        return mean

    def min_(self):
        min = self.temp_arr[0]
        for i in self.temp_arr:
            if min > i:
                min = i
        
        return min
        
    def max_(self):
        max = self.temp_arr[0]
        for i in self.temp_arr:
            if max < i:
                max = i
        
        return max
        
    def med_(self):
        self.temp_arr.sort()
        if self.length%2:
            med = self.temp_arr[int(self.length/2)]
        else:
            med = (self.temp_arr[int(self.length/2-1)] + self.temp_arr[int(self.length/2)])/2
        
        return med
    
    def std_(self):
        std = 0
        for i in self.temp_arr:
            std+=(i-self.mean_())**2
        std /= self.length
        std = std**0.5
        
        return std

stats = Statistic("landslide_data_original.csv", seriesname="temperature")


if __name__ == "__main__":
    mean, mininmun, maximum, median, stdv = stats.mean_(), stats.min_(), stats.max_(), stats.med_(), stats.std_()
    print(f"The statistical measures of Temperature attribute are: \nMean = {mean:.2f} \nMaximum = {maximum:.2f} \nMinimum = {mininmun:.2f} \nMedian = {median:.2f} \nSTD = {stdv:.2f}")
