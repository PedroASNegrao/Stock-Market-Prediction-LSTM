import csv
import numpy as np
import pandas as pd
import math

csv_file = './../../Data/ITSA4_SA.csv'

data = pd.read_csv(csv_file)

final_df_columns = data.columns

data_array = data.to_numpy()

final_array = []

for line in data_array:
    if not math.isnan(line[1]):
        final_array.append(line)


#print(inverted_array[129][1])
df = pd.DataFrame(data = final_array, columns=final_df_columns)
df.to_csv(csv_file, sep=',', index=False, encoding='utf-8')

#test later
print("checking if any null values are present\n", df.isna().sum())

