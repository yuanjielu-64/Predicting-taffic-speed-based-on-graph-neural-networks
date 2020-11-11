import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.dates as mdates
from matplotlib.pylab import datestr2num
from datetime import datetime

STGCN_MU_6_speed = pd.read_csv('result/label.speed_12.csv')
STGCN_MU_6_pred = pd.read_csv('result/pred_speed_12.csv')

ASTGCN_6 = pd.read_csv('result/ASTGCN_6.csv')
graph = pd.read_csv('result/GraphWaveNet_6.csv')
STGCN_6 = pd.read_csv('result/STGCN_6.csv')
STGCN_MU_3 = pd.read_csv('result/STGCN_MU_3.csv')
STGCN_MU_6 = pd.read_csv('result/STGCN_MU_6.csv')
STGCN_MU_12 = pd.read_csv('result/STGCN_MU_12.csv')
T_GCN = pd.read_csv('result/T-GCN_6.csv')
hm = pd.read_csv('result/aaa.csv', index_col = 'road segment')

# f, ax = plt.subplots(figsize=(6, 8))
# sns.heatmap(hm.T, cmap="YlGnBu")
# plt.xlabel("Time step = 6")
# plt.ylabel("road segment name")
# plt.savefig('picture/heapmap.jpg')
# plt.show()


# STGCN_MU_6_speed['time'] = pd.to_datetime(STGCN_MU_6_speed['time'])
# STGCN_MU_6_pred['time'] = pd.to_datetime(STGCN_MU_6_pred['time'])
# s = STGCN_MU_6_speed.iloc[240:310]
# p = STGCN_MU_6_pred.iloc[240:310]
# x = range(len(s))
# x_date = s.time
# plt.style.use('bmh')
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(1,1,1)
# ax.xaxis.set_major_locator(mdates.HourLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
#
# plt.xlabel("time")
# plt.xticks(rotation=45)
# plt.ylabel("speed")
# a = '110P04611'
# plt.plot_date(x_date,s[a],'-',label="groundTrue")
# plt.plot_date(x_date,p[a],'-',label="Predicted speed")
# plt.ylim(5, 65)
# plt.legend()
# plt.grid(True)
# plt.savefig('picture/3_12.jpg')
# plt.show()
#
# plt.plot( 'epoch', 'training_rmse', label = 'P = 3', data=STGCN_MU_3, color = 'red')
# plt.plot( 'epoch', 'training_rmse', label = 'P = 6', data=STGCN_MU_6, color = 'orange')
# plt.plot( 'epoch', 'training_rmse', label = 'P = 12', data=STGCN_MU_12, color = 'green')
# plt.plot( 'epoch', 'validation_rmse', label = '(Val)P = 3', data=STGCN_MU_3, linestyle='dashed', color = 'red')
# plt.plot( 'epoch', 'validation_rmse', label = '(Val)P = 6', data=STGCN_MU_6, linestyle='dashed', color = 'orange')
# plt.plot( 'epoch', 'validation_rmse', label = '(Val)P = 12', data=STGCN_MU_12, linestyle='dashed', color = 'green')
# plt.xlabel('training epoch')
# plt.ylabel('RMSE')
# plt.ylim(3.5, 12)
# plt.legend()
# plt.savefig('picture/STGCN_WZ.jpg')
# plt.show()
#
# plt.plot( 'epoch', 'validation_rmse', label = 'GraphWaveNet_6', data=graph, linestyle='dashed')
# plt.plot( 'epoch', 'validation_rmse', label = 'ASTGCN_6', data=ASTGCN_6, linestyle='dashed')
# plt.plot( 'epoch', 'validation_rmse', label = 'STGCN_6', data=STGCN_6, linestyle='dashed')
# plt.plot( 'epoch', 'validation_rmse', label = 'STGCN_MU_6', data=STGCN_MU_6, linestyle='dashed')
# plt.plot( 'epoch', 'validation_rmse', label = 'T_GCN', data=T_GCN, linestyle='dashed')
# plt.xlabel('epoch')
# plt.ylabel('validation_rmse')
#
# plt.legend()
# plt.savefig('picture/validation_rmse.jpg')
# plt.show()

# plt.plot( 'epoch', 'running_time', label = 'STGCN_MU_6', data=STGCN_MU_6, linestyle='dashed')
# plt.plot( 'epoch', 'running_time', label = 'ASTGCN_6', data=ASTGCN_6, linestyle='dashed')
# plt.xlabel('training epoch')
# plt.ylabel('training time')
#
# plt.legend()
# plt.savefig('picture/STGCN_MU_training_time.jpg')
# plt.show()

TGCN = {'index':['TGCN'],'3_rmse': [5.3574],'3_mae': [3.3912],'3_mape': [0.0990],'6_rmse': [6.0414],'6_mae':[3.5641],'6_mape':[0.1121],'9_rmse': [ 6.8192],'9_mae': [4.3231],'9_mape': [0.1257]}
ASTGCN = {'index':['ASTGCN'],'3_rmse': [4.1365 ],'3_mae': [2.7490],'3_mape': [0.0865],'6_rmse': [5.1173],'6_mae':[3.1641],'6_mape':[0.1037],'9_rmse': [5.9516],'9_mae':[3.5749],'9_mape': [0.1121]}
STGCN = {'index':['STGCN'],'3_rmse':[5.1046],'3_mae': [3.2957],'3_mape': [0.1001],'6_rmse': [ 5.6391],'6_mae':[3.4207],'6_mape':[0.1107],'9_rmse': [6.5761],'9_mae': [4.2532],'9_mape': [0.1332]}
GWNET = {'index':['GWNET'],'3_rmse': [4.9261],'3_mae': [3.0173],'3_mape':[0.0904],'6_rmse': [5.8157],'6_mae':[3.5207],'6_mape': [0.1168],'9_rmse': [6.9043],'9_mae': [4.3532],'9_mape': [0.1387]}
STGCN_WZ= {'index':['STGCN_WZ'],'3_rmse': [3.9938],'3_mae': [2.6023],'3_mape': [0.0820],'6_rmse': [4.8719],'6_mae':[3.0844],'6_mape':[0.0954],'9_rmse': [5.7287],'9_mae': [3.5138],'9_mape': [0.1065]}
STGCN_CON = {'index':['STGCN_WZ(NO)'],'3_rmse': [4.0584],'3_mae': [2.6805],'3_mape': [0.0837],'6_rmse': [4.9594],'6_mae':[3.0880],'6_mape': [0.1079],'9_rmse': [5.8751],'9_mae': [3.5622],'9_mape': [0.1093]}
df = pd.DataFrame(data=TGCN)
df1 = pd.DataFrame(data=ASTGCN)
df2 = pd.DataFrame(data=STGCN)
df3 = pd.DataFrame(data=GWNET)
df4 = pd.DataFrame(data=STGCN_WZ)
df5 = pd.DataFrame(data=STGCN_CON)
a = pd.concat([df,df1,df2,df3,df4,df5])

x = np.arange(3)  # the label locations
width = 0.1 # the width of the bars

a1 = [a['3_rmse'].iloc[0],a['6_rmse'].iloc[0],a['9_rmse'].iloc[0]] # TGCN
a2 = [a['3_rmse'].iloc[3],a['6_rmse'].iloc[3],a['9_rmse'].iloc[3]] # GWNET
a3 = [a['3_rmse'].iloc[2],a['6_rmse'].iloc[2],a['9_rmse'].iloc[2]] # STGCN
a4 = [a['3_rmse'].iloc[1], a['6_rmse'].iloc[1],a['9_rmse'].iloc[1]] #ASTGCN
a6 = [a['3_rmse'].iloc[5],a['6_rmse'].iloc[5],a['9_rmse'].iloc[5]] # CON
a5 = [a['3_rmse'].iloc[4],a['6_rmse'].iloc[4],a['9_rmse'].iloc[4]] # WZ

fig, ax = plt.subplots(figsize=(6, 6))
rects1 = ax.bar(x - 5 * width/2, a1, width, label='T-GCN', color = 'orchid')
rects2 = ax.bar(x - 3 * width/2, a2, width, label='GWNET')
rects3 = ax.bar(x - 1 * width/2, a3, width, label='STGCN')
rects4 = ax.bar(x + 1 * width/2, a4, width, label='ASTGCN')
rects5 = ax.bar(x + 3 * width/2, a6, width, label='STGCN_WZ(NO)')
rects6 = ax.bar(x + 5 * width/2, a5, width, label='STGCN_WZ')
plt.xticks(range(0, 3), ['15 min(t\' = 3)', '30 min(t\' = 6)', '60 min(t\' = 12)'])
plt.ylim(3.5, 7.5)
plt.ylabel('test RMSE')
plt.legend()
plt.savefig('picture/RMSE_tyson.jpg')
plt.show()


TGCN = {'index':['TGCN'],'3_rmse': [6.1833],'3_mae': [4.1618],'3_mape': [0.1104],'6_rmse': [6.9281],'6_mae':[4.8187],'6_mape':[0.1374],'9_rmse': [8.2604],'9_mae': [5.4124],'9_mape': [0.1504]}
ASTGCN = {'index':['ASTGCN'],'3_rmse': [5.2021],'3_mae': [3.0443],'3_mape': [0.0787],'6_rmse': [6.1271],'6_mae':[3.5061],'6_mape':[0.1011],'9_rmse': [7.5650],'9_mae':[4.4192],'9_mape': [0.1289]}
STGCN = {'index':['STGCN'],'3_rmse':[6.2700],'3_mae': [4.1805],'3_mape': [0.1192],'6_rmse': [6.8814],'6_mae':[4.7357],'6_mape':[0.1253],'9_rmse': [7.9887],'9_mae': [5.3654],'9_mape': [0.1414]}
GWNET = {'index':['GWNET'],'3_rmse': [5.5654],'3_mae': [3.0042],'3_mape':[0.0977],'6_rmse': [6.6461],'6_mae':[3.7182],'6_mape': [0.1235],'9_rmse': [7.8951],'9_mae': [4.7148],'9_mape': [0.1400]}
STGCN_WZ= {'index':['STGCN_WZ(NO)'],'3_rmse': [4.9676],'3_mae': [2.9429],'3_mape': [0.0776],'6_rmse': [5.8438],'6_mae':[3.2782],'6_mape':[0.0907],'9_rmse': [7.2826],'9_mae': [4.3831],'9_mape': [0.1271]}

df = pd.DataFrame(data=TGCN)
df1 = pd.DataFrame(data=ASTGCN)
df2 = pd.DataFrame(data=STGCN)
df3 = pd.DataFrame(data=GWNET)
df4 = pd.DataFrame(data=STGCN_WZ)
#df5 = pd.DataFrame(data=STGCN_CON)
a = pd.concat([df,df1,df2,df3,df4])

x = np.arange(3)  # the label locations
width = 0.1 # the width of the bars

a1 = [a['3_rmse'].iloc[0],a['6_rmse'].iloc[0],a['9_rmse'].iloc[0]]
a2 = [a['3_rmse'].iloc[3],a['6_rmse'].iloc[3],a['9_rmse'].iloc[3]]
a3 = [a['3_rmse'].iloc[2],a['6_rmse'].iloc[2],a['9_rmse'].iloc[2]]
a4 = [a['3_rmse'].iloc[1],a['6_rmse'].iloc[1],a['9_rmse'].iloc[1]]
a5 = [a['3_rmse'].iloc[4],a['6_rmse'].iloc[4],a['9_rmse'].iloc[4]]
#a6 = [a['3_rmse'].iloc[5],a['6_rmse'].iloc[5],a['9_rmse'].iloc[5]]
fig, ax = plt.subplots(figsize=(6, 6))
rects1 = ax.bar(x - 5 * width/2, a1, width, label='T-GCN', color = 'orchid')
rects2 = ax.bar(x - 3 * width/2, a2, width, label='GWNET')
rects3 = ax.bar(x - 1 * width/2, a3, width, label='STGCN')
rects4 = ax.bar(x + 1 * width/2, a4, width, label='ASTGCN')
rects5 = ax.bar(x + 3 * width/2, a5, width, label='STGCN_WZ(NO)')
#rects6 = ax.bar(x + 5 * width/2, a6, width, label='STGCN_WZ(NO)')
plt.xticks(range(0, 3), ['15 min(t\' = 3)', '30 min(t\' = 6)', '60 min(t\' = 12)'])
plt.ylim(4, 9)
plt.ylabel('test RMSE')
plt.legend()
plt.savefig('picture/RMSE_los.jpg')
plt.show()

