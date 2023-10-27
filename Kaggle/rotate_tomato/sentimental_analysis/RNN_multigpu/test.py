import os
import pandas as pd

current_path=os.path.dirname(__file__)
data_path="../data/rotate_tomato"
path=os.path.join(current_path,data_path,"train.tsv")
print(path)

with open (path,"r") as t:
    data=pd.read_csv(t, sep='\t',)
print(data.head)