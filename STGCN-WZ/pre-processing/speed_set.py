import pandas as pd
import numpy as np

df = pd.read_csv(r'Y:\project\data\data_tmc2.csv')
#df = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/data_tmc.csv')
df_tmc = pd.read_csv(r'Y:\project\data\tmc.csv')
#df_tmc = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/tmc.csv')
time = pd.read_csv(r'Y:\project\data\time_random.csv')
#time = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/time_random.csv')
tmc_list = list(df_tmc.tmc)
time_list = list(time.time)
#time = df['measurement_tstamp'].drop_duplicates()
#time = pd.DataFrame(time)
#time.to_csv("X:/project/data/time_random.csv")
print(time)
df.measurement_tstamp = pd.to_datetime(df.measurement_tstamp)
time.time = pd.to_datetime(time.time)
speed_adj = pd.DataFrame(np.zeros(shape = (len(time_list),len(tmc_list))), index = time_list, columns = tmc_list)
construction_adj = pd.DataFrame(np.zeros(shape = (len(time_list),len(tmc_list))), index = time_list, columns = tmc_list)
for i in range(len(time_list)):
    tmc = df[df['measurement_tstamp'] == time['time'].iloc[i]]
    for j in range(len(tmc)):
        b_index = tmc_list.index(tmc['tmc_code'].iloc[j])
        print(tmc['speed'].iloc[j])
        speed_adj.iat[i,b_index ] = tmc['speed'].iloc[j]
        construction_adj.iat[i, b_index] = tmc['Work_zone'].iloc[j]
speed_adj.to_csv(r'Z:\project\data\adj\new_speed_adj_10.csv', index = True, header=True)
construction_adj.to_csv(r'Z:\project\data\adj\new_construction_adj_10.csv', index = True, header=True)
#speed_adj.to_csv('/Users/yuanjielu/Desktop/python/project/data/adj/speed_adj.csv', index = True, header = True)
#construction_adj.to_csv('/Users/yuanjielu/Desktop/python/project/data/adj/construction_adj_10.csv', index = True, header = True)


