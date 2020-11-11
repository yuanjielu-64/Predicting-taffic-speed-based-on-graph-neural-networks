import pandas as pd
import datetime
import holidays

#df = pd.read_csv(r'Z:\project\data\data_tmc.csv')
df = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/data_tmc.csv')
#df_weather = pd.read_csv(r'Z:\project\data\weather.csv')
df_weather = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/weather.csv')
df.measurement_tstamp = pd.to_datetime(df.measurement_tstamp)
print(df.shape)
#_time = []
#for i in range(len(df_weather)):
#    _time.append('2019/' + df_weather.date.iloc[i] + ' ' + df_weather.time.iloc[i])
#df_weather['time'] = _time
df_weather['time'] = pd.to_datetime(df_weather['time'])
#df_weather['time'] = df_weather['time'].dt.round('10min')

#df_weather = df_weather.sort_values('time', ascending=False).drop_duplicates(
#    ['time']).sort_index().reset_index(drop=True)

#df_weather.to_csv('/Users/yuanjielu/Desktop/python/project/data/weather.csv', index = False, header = True)
#print(df_weather['time'])

_weather = []

#print(df_weather['time'].dt.day)
#print(df['measurement_tstamp'].iloc[1])

for i in range(len(df)):
    a = df_weather[(df_weather['time'] - df['measurement_tstamp'].iloc[i]).dt.days == 0 ]
    a = a[(a['time'] - df['measurement_tstamp'].iloc[i]).dt.seconds <= 3600]
    print(i)
    if a.empty != True:
        for k in range(len(a)):
            b = a['weather']
            _weather.append(a['weather'].iloc[k])
            break
    else:
        _weather.append('Cloudy')

_weather = pd.DataFrame(_weather)
#df['weather'] = _weather
_weather.to_csv('/Users/yuanjielu/Desktop/python/project/data/_weather.csv')
#_weather.to_csv(r'Z:\project\data\_weather.csv', index = False, header=True)
day = []
month = []
for date, name in sorted(holidays.US(state='VA', years=2019).items()):
    print(date,name)
    month.append(date.month)
    day.append(date.day)

record = []

for i in range(len(df)):
    flag = 0
    for j in range(len(month)):
        if df['measurement_tstamp'].iloc[i].month == month[j] and df['measurement_tstamp'].iloc[i].day == day[j]:
            flag = 1
    record.append(flag)

df['IsHoliday'] = record

tmc = ['TMC_I-66','TMC_I-495','TMC_VA-7','TMC_VA-123','TMC_VA-267']

_tmc = []
for i in range(len(df)):
    for j in tmc:
        #df_tmc = pd.read_csv(r'Z:/project/data/speed/' + j + '.csv')
        df_tmc = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/speed/' + j + '.csv')
        df_tmc = df_tmc.sort_values('tmc', ascending=False).drop_duplicates(
            ['tmc']).sort_index().reset_index(drop=True)
        a = df_tmc[df_tmc['tmc'] == df['tmc_code'].iloc[i]]
        if a.empty != True:
            for k in range(len(a)):
                _tmc.append(j)

#df['tmc'] = _tmc
_tmc = pd.DataFrame(_tmc)
#df.to_csv(r'Z:\project\data\data_tmc1.csv', index = False, header=True)

#_tmc.to_csv('/Users/yuanjielu/Desktop/python/project/data/_tmc.csv')
#_tmc.to_csv(r'Z:\project\data\data_tmc1.csv', index = False, header=True)
df.to_csv('/Users/yuanjielu/Desktop/python/project/data/data_tmc1.csv', index = False, header = True)