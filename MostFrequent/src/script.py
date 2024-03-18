import pandas as pd

data = pd.read_csv("E:/mostfrequent/task_1_3.txt", header=None, sep=" ")
data = data.sort_values(axis=0, ascending=False, by=1)[:10]
data.to_csv("E:/mostfrequent/task_1_3.txt", header=None, index=None, sep=" ")