import pandas as pd
import numpy as np

#df = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/data1.csv')
df_adj = pd.read_excel(r'X:/project/data/adj/adj_tyson.xlsx')
df = pd.read_csv(r'X:/project/data/tmc.csv')
#df = pd.read_csv(r'Z:/project/data/data_tmc.csv')
#tmc = ['TMC_I-66','TMC_I-495','TMC_VA-7','TMC_VA-123','TMC_VA-267']

#a = np.load(r'D:\Model\ASTGCN-master\data\PEMS04\pems04.npz')

print(df)
#df.to_csv(r'Z:\project\data\data_tmc.csv', index = False, header=True)
tmc = list(df.tmc)

for i in range(len(df_adj)):
    a = df_adj['start'].iloc[i]
    b = df_adj['end'].iloc[i]

    a_m= df[df['tmc'] == a]['miles'].values
    b_m= df[df['tmc'] == b]['miles'].values

    df_adj['distance'].iloc[i] = ((a_m[0] + b_m[0]) / 2).round(3)


df_adj.to_excel(r'X:/project/data/adj/adj_tyson.xlsx')




