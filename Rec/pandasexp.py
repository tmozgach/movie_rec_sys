import numpy as np
import pandas as pd
data=pd.read_csv("ml-latest-small/ratings.csv")
print(data.keys())
df=data[["userId","movieId","rating","timestamp"]].values
print(df.shape)

