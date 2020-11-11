import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

#df = pd.read_csv(r'Y:\Project\data_tmc.csv',index_col = 0)
df = pd.read_csv('/Users/yuanjielu/Desktop/Research/Project/tmc_final.csv')
#df = pd.read_csv(r'Y:\Project\tmc_final.csv')
df_speed = pd.DataFrame(df[df.direction == 'south'])
#df_speed = pd.DataFrame(df[(df.tmc_code == '110-04175') | (df.tmc_code == '110N04175') | (df.tmc_code == '110-04174') | (df.tmc_code == '110N04174')])
#df_speed = pd.DataFrame(df[(df.tmc_code != '110+04176') &(df.tmc_code != '110N04174') & (df.tmc_code != '110N04607') & (df.tmc_code != '110+04609')])
# '10-15' '10-16' '10-17' '10-18' '10-19' '10-20' '11-11' '9-29'

fig,ax=plt.subplots()
fig.set_size_inches(18,6)
order_hour = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12,13,14,15,16,17,18,19,20,21,22,23]
count_by_hour = pd.DataFrame(df_speed[df_speed.Facts == 'AllClean'].groupby(by='hour')['speed'].mean()).reset_index()
count_by_hour1 = pd.DataFrame(df_speed[(df_speed.WorkZone == 1) | (df_speed.R_WorkZone == 1)].groupby(by='hour')['speed'].mean()).reset_index()
#count_by_hour2 = pd.DataFrame(df_speed[df_speed.Facts == 'Collision'].groupby(by='hour')['speed'].mean()).reset_index()
print(count_by_hour1)
bar_width = 0.3
plt.bar(x = np.arange(len(count_by_hour.hour)),height = count_by_hour.speed,width = bar_width, label = 'Allclean')
plt.bar(x = np.arange(len(count_by_hour1.hour))+bar_width, height = count_by_hour1.speed, width = bar_width, label = 'Workzone and many')
#plt.bar(x = np.arange(len(count_by_hour2.hour))+2 * bar_width, height = count_by_hour2.speed, width = bar_width, label = 'Collision')
plt.xticks(np.arange(24)+0.4, order_hour)
plt.legend()
plt.savefig("/Users/yuanjielu/Desktop/Research/Project/picture/bar of hour(south).png")
plt.show()


fig, ax = plt.subplots()
fig.set_size_inches(15, 6)

count_by_weekday = pd.DataFrame(df_speed[df_speed.Facts == 'AllClean'].groupby(by=['weekday', 'hour'])['speed'].mean()).reset_index()
count_by_weekday1 = pd.DataFrame(df_speed[(df_speed.WorkZone == 1) | (df_speed.R_WorkZone == 1)].groupby(by=['weekday', 'hour'])['speed'].mean()).reset_index()
order_weekday = [1,2,3,4,5,6,7]
sns.pointplot(data=count_by_weekday, x='hour', y='speed', hue='weekday', hue_order=order_weekday, scale=0.5, ax=ax)
sns.pointplot(data=count_by_weekday1, x='hour', y='speed', hue='weekday', hue_order=order_weekday, scale=0.5, ax=ax,linestyles = '--')
plt.legend()
ax.set(xlabel='hour', ylabel='speed', title='weekday and speed')
plt.savefig("/Users/yuanjielu/Desktop/Research/Project/picture/speed of weekday(south).png")
plt.show()


speed = df_speed[df_speed.Facts == 'AllClean'].pivot_table(index = 'hour', columns = 'weekday', values = 'total cost', aggfunc = np.mean , margins=1)
plt.figure(figsize=(16,16))
sns.heatmap( data = speed,cmap='Reds', annot=True, fmt=".0f")
plt.xlabel('\nweekday', fontsize=22)
plt.ylabel('hour\n', fontsize=22)
plt.title('\nHeatmap: Median cost by Hour and Weekday(allclean)\n\n', fontsize=14, fontweight='bold');
plt.savefig("/Users/yuanjielu/Desktop/Research/Project/picture/Heatmap_allclean(south).png")
plt.show()



speed = df_speed[(df_speed.WorkZone == 1) | (df_speed.R_WorkZone == 1)].pivot_table(index = 'hour', columns = 'weekday', values = 'total cost', aggfunc = np.mean , margins=1)
plt.figure(figsize=(16,16))
sns.heatmap( data = speed,cmap='Reds', annot=True, fmt=".0f")
plt.xlabel('\nweekday', fontsize=22)
plt.ylabel('hour\n', fontsize=22)
plt.title('\nHeatmap: Median cost  by Hour and Weekday(workzone)\n\n', fontsize=14, fontweight='bold');
plt.savefig("/Users/yuanjielu/Desktop/Research/Project/picture/Heatmap_workzone(south).png")
plt.show()


df['measurement_tstamp'] = pd.to_datetime(df['measurement_tstamp'])
tmc1 = df[(df.tmc_code == '110-04175') & (df.measurement_tstamp >= '2019-5-1 00:00:00') &  (df.measurement_tstamp <= '2019-5-31 23:00:00')]
tmc2 = df[(df.tmc_code == '110-04175') & (df.measurement_tstamp >= '2019-5-1 00:00:00') &  (df.measurement_tstamp <= '2019-5-31 23:00:00')]
record = []

fig, ax1 = plt.subplots()
fig.set_size_inches(300,6)

xs = tmc1.measurement_tstamp
ys = tmc1.speed
ys1= tmc1.average_speed
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d%H'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax1.tick_params(pad=10)
sns.pointplot(x = xs, y = ys, hue = tmc1.Facts, scale=0.5, ax=ax1 )
sns.pointplot(x = xs, y = ys1, scale=0.2, ax=ax1,linestyles = '--', color = 'black')
plt.gcf().autofmt_xdate()
plt.legend()
ax1.set(xlabel='measurement_tstamp', ylabel='speed', title='weekday and speed')
plt.savefig("/Users/yuanjielu/Desktop/Research/Project/picture/measurement_tstamp(110-04175).png")
plt.show()

