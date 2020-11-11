import pandas as pd
from random import shuffle

df = pd.read_csv(r"D:\Model\project\data\adj\speed\5_min_speed_adj_limitation.csv")

df.time = pd.to_datetime(df.time)
print(df)

time_interval = 300
a = []
b = []
for i in range(len(df) - 1):
    if i == 64:
        print('a')
    f = df['time'].iloc[i + 1]
    g = df['time'].iloc[i]
    if (df['time'].iloc[i + 1] - df['time'].iloc[i]).seconds == time_interval:
        if i == len(df) - 1:
            b.append(df.iloc[i])
            b.append(df.iloc[i+1])
        else:
            b.append(df.iloc[i])
    else:
        b.append(df.iloc[i])
        a.append(b)
        b = []

print(a)
c = []
shuffle(a)
for i in range(len(a)):
    for j in range(len(a[i])):
        c.append(a[i][j])


print(c)
c = pd.DataFrame(c)
c.to_csv(r'D:\Model\project\data\adj\speed\datas.csv')
