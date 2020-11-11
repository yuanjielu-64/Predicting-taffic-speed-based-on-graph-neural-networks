
import pandas as pd
import numpy as np
adj = pd.read_csv(r'X:\project\data\adj\adj_1.csv',index_col = 0)
con = pd.read_csv(r'X:\project\data\adj\construction_adj_10.csv', index_col = 0)

# name = []
col = con.columns.values
adj_col = adj.columns.values
# for i in range(len(con)):
#     print(i)
#     for j in range(len(col)):
#         if con[col[j]].iloc[i] == 1:
#             name.append(col[j])
#     for k in range(len(name)):
#         a = name[k]
#         for j in range(len(adj_col)):
#             dis = adj[adj_col[j]][name[k]]
#             if dis <= 2.5:
#                 con[adj_col[j]].iloc[i] = 1

x = []
for i in range(len(adj_col)):
    y = []
    for j in range(len(adj_col)):
        a = adj_col[i]
        b = adj_col[j]
        dis = adj[adj_col[j]][adj_col[i]]
        if dis <= 2.5 and dis > 0:
            y.append(adj_col[j])
    x.append(y)
print('1')

name = []
for i in range(len(con)):
    for j in range(len(col)):
        if con[col[j]].iloc[i] == 1:
            name.append(j)
    for k in range(len(name)):
        a = x[name[k]]
        for h in range(len(a)):
            con[a[h]].iloc[i] = 1








#con.to_csv(r'X:\project\data\adj\construction_adj_10_1.csv', index = True, header=True)
