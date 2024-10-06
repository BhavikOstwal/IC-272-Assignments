import numpy as np
import pandas as pd

def custom_mean(group):
    numeric_cols = group.select_dtypes(include='number')
    mean_values = numeric_cols.sum() / len(group)
    return mean_values

def custom_std(group):
    numeric_cols = group.select_dtypes(include='number')
    mean_values = numeric_cols.sum() / len(group)
    variance = ((numeric_cols - mean_values) ** 2).sum() / len(group)
    std_dev = np.sqrt(variance)
    return std_dev

def custom_cov(group):
    numeric_cols = group.select_dtypes(include='number')
    mean_values = custom_mean(group)
    centered = numeric_cols - mean_values
    cov_matrix = (centered.T @ centered) / len(group)
    return cov_matrix