import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import folium
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

df = pd.read_csv(r'Y:\Project\data.csv')
df['Start time'] = pd.to_datetime(df['Start time'])
df['Closed time'] = pd.to_datetime(df['Closed time'])
records = []

lati = 38.88978
long = -77.21302
distance = []

for i in range(len(df)):
    distance.append(haversine(long,lati,df['Longitude'].iloc[i],df['Latitude'].iloc[i]))
df['distance'] = distance
df.drop(df[df.distance > 14].index,axis=0,inplace=True)
df.drop(columns = ['distance'],inplace = True)

for i in range(len(df)):
    if df['Duration (Incident clearance time)'].iloc[i] != 'Ends before it began' and df["Roadway Clearance Time"].iloc[i] != "Road cleared before the event began":
        ts = df['Closed time'].iloc[i] - df['Start time'].iloc[i]
        days = float(ts.days * 24 * 60)
        ts = (float(ts.seconds) // 60)
        ts = days + ts
        records.append(ts)
    else:
        records.append("error")

df['Duration'] = records
df.drop(df[df.Duration == 'error'].index, axis = 0, inplace = True)
df = df.sort_values('Duration', ascending=False).drop_duplicates(['Start time','Standardized Type','Road']).sort_index().reset_index(drop=True)

records = []
for i in range(len(df)):
    if "Disabled Vehicle" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Traffic" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Delays" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Fire" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Overgrown Plants" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Weather" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Flood" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Special" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Fallen Tree" in df['Standardized Type'].iloc[i]:
        records.append('OtherFacts')
    elif "Collision" in df['Standardized Type'].iloc[i]:
        records.append('Collision')
    elif "Incident" in df['Standardized Type'].iloc[i]:
        records.append('Collision')
    else:
        records.append('WorkZone')

df['Standardized_Type'] = records
df_other = df.drop(df[df.Standardized_Type == 'WorkZone'].index, axis = 0)
df.drop(df[df.Standardized_Type != 'WorkZone'].index, axis = 0)

for i in range(len(df)):
    start = df['Start time'].iloc[i]
    end = df['Closed time'].iloc[i]
    print(i)
    for j in range(len(df_other)):
        other_start = df_other['Start time'].iloc[j]
        other_end =  df_other['Closed time'].iloc[j]
        a = (start - other_end).total_seconds()
        b = (end - other_start).total_seconds()

        if a > 0 or b < 0:
            pass
        else:
            df.drop(index=i, inplace=True)
            break

print(df)


a = df["Roadway Clearance Time"].str.split(" ")
a = np.array(a)
records = []
for i in range(len(a)):
    mins = 0
    day = 0
    hour = 0
    second = 0
    for j in range(len(a[i])):
        if a[i][j] == "minutes":
            mins = float(a[i][j-1])
        elif a[i][j] == "hours":
            hour = float(a[i][j-1])
        elif a[i][j] == "day":
            day = float(a[i][j-1])
        elif a[i][j] == "seconds":
            second = float(a[i][j-1])
    records.append(day * 24 * 60 + hour * 60 + mins + second / 60)

df['Roadway_Clearance_Time'] = records
df['Roadway_Clearance_Time'] = df['Roadway_Clearance_Time'].round(1)
df["Latitude"] = df["Latitude"].round(6)
df["Longitude"] = df["Longitude"].round(6)
df.drop(columns = ['Roadway Clearance Time','Duration (Incident clearance time)'],inplace = True)

world_map = folium.Map()
latitude = 38.88978
longitude = -77.21302
collision = folium.map.FeatureGroup()
for lat,lng in zip(df.Latitude, df.Longitude):
    collision.add_child(
        folium.CircleMarker(
            [lat, lng],
            radius=2,
            color='yellow',
            fill=True,
            fill_color='red',
            fill_opacity=0.4
        )
    )

san_map = folium.Map(location = [latitude, longitude], zoom_start= 12,tiles='Stamen Toner')
san_map.add_child(collision)
san_map.save('X:/project/data/2019_tyson.html')

df.to_csv(r'X:\project\data\data.csv', index = False, header=True)
df = pd.read_csv(r'X:\project\data\data.csv')

start = []
close = []

for i in range(len(df)):
    start.append(df['Start time'].iloc[i][0:19])
    close.append(df['Closed time'].iloc[i][0:19])
df['Start'] = start
df['R_Close'] = close

record = []

df['Start'] = pd.to_datetime(df['Start'])

for i in range(len(df)):
    R = df.Roadway_Clearance_Time.iloc[i]
    day = 0
    hour = 0
    min = 0
    if R < 60:
       min = R
    elif R >=60 and R < 1440:
        hour = R // 60
        min = R % 60
    else:
        day = R // 1440
        hour = R % 1440 // 60
        min = R % 1440 % 60
    record.append(df.Start.iloc[i] + pd.Timedelta(days = day,hours = hour, minutes = min))
df['Close'] = record
df.drop(columns = ['EDC Incident Type','Start time','Closed time','R_Close'],inplace = True)
df['Start1'] = df['Start'].dt.round('10min')
df['Close1'] = df['Close'].dt.round('10min')

index = []
for i in range(len(df)):
    index.append(i)
df['index'] = index
df = pd.read_csv(r'X:\project\data\data1.csv')
tmc = pd.read_csv(r'X:\project\data\tmc.csv')
_tmc = []
for i in range(len(df)):
    min = haversine(df['Longitude'].iloc[i],df['Latitude'].iloc[i],tmc['longtitude'].iloc[0],tmc['latitude'].iloc[0])
    tmc_name = tmc.tmc.iloc[0]
    for j in range(1,len(tmc)):
        dis = haversine(df['Longitude'].iloc[i],df['Latitude'].iloc[i],tmc['longtitude'].iloc[j],tmc['latitude'].iloc[j])
        if dis < min:
            min = dis
            tmc_name = tmc.tmc.iloc[j]
    _tmc.append(tmc_name)

df['con_tmc'] = _tmc

df.to_csv(r'X:\project\data\data1.csv', index = False, header=True)



