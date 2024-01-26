import pandas as pd 
import numpy as np 

def first4Moments(data):
    n = len(data)

    mean = np.sum(data)/n
    variance = np.sum(np.square(data - mean))/(n-1)

    # calculate the biased sigma2 to form the unbiased skewness and kurtosis
    sim_corrected = data - mean
    sigma2 = np.sum(np.square(data - mean))/n

    # normalized skewness = unnormalized skewness/ sigma3
    skewness = n*np.sum((data-mean)**3)/((n-1)*(n-2))/sigma2**1.5
    kurtosis = np.sum(sim_corrected ** 4) / n / (sigma2 ** 2)

    return mean, variance, skewness, kurtosis

file_path = "problem1.csv"
data = pd.read_csv(file_path)
dataset = data['x'].values

m, s2, sk, k = first4Moments(dataset)

print("calculate_Mean:", m)
print("calculate_Variance:", s2)
print("calculate_Skewness:", sk)
print("calculate_Kurtosis:", k)

pd_m = data['x'].mean()
pd_x = data['x'].var()
pd_sk = data['x'].skew()
pd_k = data['x'].kurtosis()
print ("\n")
print("pandas_Mean:", pd_m)
print("pandas_Variance:", pd_x)
print("pandas_Skewness:", pd_sk)
print("pandas_Kurtosis:", pd_k)