
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn import metrics
from sklearn.metrics import mean_squared_error

from math import sqrt
from sklearn.metrics.scorer import make_scorer
def rmse(true, predicted):
  return sqrt(mean_squared_error(np.expm1(true), np.expm1(predicted)))

df = pd.read_csv(r'Y:\Project\tmc_final.csv')

target = df.speed
target = np.array(target)
useless = ['average_speed','reference_speed','travel_time_minutes','Date','measurement_tstamp',
            'total cost','person_hrs_of_delay','Vehicle_hrs_of_delay', 'speed','total VMT'
           ]
dataset = df.drop(columns = useless)

enc = preprocessing.LabelEncoder()
dataset.tmc_code = enc.fit_transform(dataset.tmc_code)
dataset.direction = enc.fit_transform(dataset.direction)
dataset.Facts = enc.fit_transform(dataset.Facts)
dataset.to_csv("day_.csv")
y = np.log1p(target)

x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=0.25, random_state = 1234)
my_scorer = make_scorer(rmse, greater_is_better=True)

y_0 = []
y_2 = []
y_4 = []
for i in range(len(x_test)):
    if x_test['Facts'].iloc[i] == 0:
        y_0.append(y_test[i])
    elif x_test['Facts'].iloc[i] == 2:
        y_2.append(y_test[i])
    elif x_test['Facts'].iloc[i] == 4:
        y_4.append(y_test[i])

knn_model = KNeighborsRegressor(weights='distance', n_neighbors = 12)
Rf = RandomForestRegressor(n_estimators = 100 , max_depth = 19,bootstrap = True, max_features = "auto")
SVR_model = SVR()
knn_model.fit(x_train,y_train)
Rf.fit(x_train,y_train)
SVR_model.fit(x_train,y_train)

print("knn")
knn_grid_workzone = knn_model.predict(x_test[x_test.Facts == 4])
print(rmse(y_4, knn_grid_workzone ))
# all_clean 0
knn_grid_allclean = knn_model.predict(x_test[x_test.Facts == 0])
print(rmse(y_0, knn_grid_allclean ))
# Many     2
knn_grid_Many = knn_model.predict(x_test[x_test.Facts == 2])
print(rmse(y_2, knn_grid_Many ))

print("Rf")
knn_grid_workzone = Rf.predict(x_test[x_test.Facts == 4])
print(rmse(y_4, knn_grid_workzone ))
# all_clean 0
knn_grid_allclean = Rf.predict(x_test[x_test.Facts == 0])
print(rmse(y_0, knn_grid_allclean ))
# Many     2
knn_grid_Many = Rf.predict(x_test[x_test.Facts == 2])
print(rmse(y_2, knn_grid_Many ))

print("SVR")
knn_grid_workzone = SVR_model.predict(x_test[x_test.Facts == 4])
print(rmse(y_4, knn_grid_workzone ))
# all_clean 0
knn_grid_allclean = SVR_model.predict(x_test[x_test.Facts == 0])
print(rmse(y_0, knn_grid_allclean ))
# Many     2
knn_grid_Many = SVR_model.predict(x_test[x_test.Facts == 2])
print(rmse(y_2, knn_grid_Many ))
