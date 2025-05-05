import os
import pandas as pd

# Define paths
csv_file = "archive"


Id=[]
for dirname, _, filenames in os.walk('archive/train'):
    for filename in filenames:
        Id.append(os.path.join(dirname, filename))


train=pd.DataFrame()
train=train.assign(filename=Id)
train.head()

train['label']=train['filename']
train['label']=train['label'].str.replace('archive/train','')

train['label'] = train['label'].str.split('\\').str[1]
print(train.head())

train.to_csv("my_data.csv")





