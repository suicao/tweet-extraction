import pandas as pd

df = pd.read_csv("data/train.csv")
df = df.sample(frac=1,random_state=96)
df.fillna("NaN",inplace=True)
df[:-3500].to_csv("data/train_holdout.csv",index=False)
df[-3500:].to_csv("data/test_holdout.csv",index=False)

