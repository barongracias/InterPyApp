import numpy as np
import pandas as pd

cw_data = pd.read_pickle('coursework_dataset.pkl')
X = cw_data["X"]
y = cw_data["y"]
feature_names = cw_data["metadata"]["feature_names"]
target_name = cw_data["metadata"]["target_name"]

df = pd.DataFrame(X, columns=feature_names)
df[target_name] = y

print(df.shape)