import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.scorer import make_scorer
from joblib import dump, load
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from math import sqrt
from sklearn.metrics.scorer import make_scorer
def rmse(true, predicted):
  return sqrt(mean_squared_error(np.expm1(true), np.expm1(predicted)))

df = pd.read_csv(r'Y:\Project\tmc_final.csv')
#df = pd.read_csv('/Users/yuanjielu/Desktop/Research/Project/tmc_final.csv')
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
#y = np.log1p(target)

x_train, x_test, y_train, y_test = model_selection. train_test_split(dataset, target, test_size=0.25, random_state = 1234)
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

parameters = {'weights': ['distance'],'n_neighbors': [12]}
knn_model = model_selection.GridSearchCV(estimator= KNeighborsRegressor(), param_grid = parameters, scoring = my_scorer,cv = 5, verbose = 2, n_jobs = 2, refit = True)
knn_model.fit(x_train,y_train)
knn_model1 = model_selection.GridSearchCV(estimator= KNeighborsRegressor(), param_grid = parameters, scoring = my_scorer,cv = 5, verbose = 2, n_jobs = 2, refit = True)
knn_model1.fit(x_train,y_train)
knn_model2 = model_selection.GridSearchCV(estimator= KNeighborsRegressor(), param_grid = parameters, scoring = my_scorer,cv = 5, verbose = 2, n_jobs = 2, refit = True)
knn_model2.fit(x_train,y_train)
knn_model3 = model_selection.GridSearchCV(estimator= KNeighborsRegressor(), param_grid = parameters, scoring = my_scorer,cv = 5, verbose = 2, n_jobs = 2, refit = True)
knn_model3.fit(x_train,y_train)

# workzone   4
knn_grid_workzone = knn_model1.predict(x_test[x_test.Facts == 4])
print(rmse(y_4, knn_grid_workzone ))
# all_clean 0
knn_grid_allclean = knn_model2.predict(x_test[x_test.Facts == 0])
print(rmse(y_0, knn_grid_allclean ))
# Many     2
knn_grid_Many = knn_model3.predict(x_test[x_test.Facts == 2])
print(rmse(y_2, knn_grid_Many ))

knn_grid_model = knn_model.predict(x_test)
dt_report = metrics.r2_score(y_test, knn_grid_model )

a = pd.DataFrame(x_test)
b = pd.DataFrame(y_test)
c = pd.DataFrame(knn_grid_model)

# a.to_csv("x_test.csv")
# b.to_csv("y_test.csv")
# c.to_csv("knn_grid.csv")

means = knn_model.cv_results_['mean_test_score']
stds = knn_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, knn_model.cv_results_['params']):
        print("Accuracy %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print(means)

