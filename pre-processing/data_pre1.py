import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

tmc = ['TMC_I-66','TMC_I-495','TMC_VA-7','TMC_VA-123','TMC_VA-267']
speed = ['I-66','I-495','VA-7','VA-123','VA-267']
tmc_ = []

#df = pd.read_csv('/Users/yuanjielu/Desktop/Research/Project/data.csv')

# for i in tmc:
#     df = pd.read_csv(r'X:/project/data/speed/' + i +'.csv')
#     #df = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/speed/' + i +'.csv')
#     df.drop(columns=['type', 'country', 'active_start_date', 'active_end_date','timezone_name'], inplace=True)
#     df = df.sort_values('tmc', ascending=False).drop_duplicates(
#         ['tmc']).sort_index().reset_index(drop=True)
#     for j in range(len(df)):
#         tmc_.append(df.iloc[j])
#
# tmc_ = pd.DataFrame(tmc_)
#tmc_.to_csv(r'Z:\project\data\tmc.csv', index = False, header=True)

df = pd.read_csv(r'Y:\project\data\dataaaaaaaa.csv')
#df = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/data1.csv')
df['Start1'] = pd.to_datetime(df['Start1'])
df['Close1'] = pd.to_datetime(df['Close1'])
_array = []
_work_road = []
_Latitude = []
_Longitude = []
_Max_Lane = []
_work_zone = []
for i in range(len(df)):
    ts = df['Close1'].iloc[i] - df['Start1'].iloc[i]
    num_ten_mins = int(ts.seconds / 300)
    _time = []

    if num_ten_mins != 0:
        for j in range(num_ten_mins + 1):
            t = j * 5
            _time.append(df['Start1'].iloc[i] + datetime.timedelta(minutes=t))
    else:
        _time.append(df['Start1'].iloc[i])

    print(_time)

    for s in speed:
        df_s = pd.read_csv(r'Y:/project/data/speed/' + s + '.csv')
        #df_s = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/speed/' + s + '.csv')
        df_s['measurement_tstamp'] = pd.to_datetime(df_s['measurement_tstamp'])
        for t in range(len(_time)):
            a = df_s[df_s['measurement_tstamp'] == _time[t]]
            for k in range(len(a)):
                _array.append(a.iloc[k])
                _work_road.append(df['Road'].iloc[i])
                _Latitude.append(df['Latitude'].iloc[i])
                _Longitude.append(df['Longitude'].iloc[i])
                _Max_Lane.append(df['Max Lanes Closed'].iloc[i])
                _work_zone.append(df['con_tmc'].iloc[i])

_array = pd.DataFrame(_array)
_array['w_road'] = _work_road
_array['w_latitude'] = _Latitude
_array['w_longitude'] = _Longitude
_array['Max_Lanes_Closed'] = _Max_Lane
print(_array)

_array.to_csv(r'Y:\project\data\data_good.csv', index = False, header=True)
#_array.to_csv('/Users/yuanjielu/Desktop/python/project/data/data_tmc.csv', index = False, header = True)















