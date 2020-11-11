import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from joblib import dump, load
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

reg = RandomForestRegressor(random_state=1234)
# parameters={'n_estimators':[10,30,50,100],'max_depth':range(10,20),
#             'bootstrap': [True, False], "max_features" : ["auto", "log2", "sqrt"] }
parameters={'n_estimators':[100],'max_depth':[19],
            'bootstrap': [True], "max_features" : ["auto"] }
clf = GridSearchCV(reg,parameters, cv =5, n_jobs = -1, verbose = 1, refit = True, scoring = my_scorer)
clf.fit(X=x_train,y=y_train.ravel())

# workzone 4
knn_grid_workzone = clf.predict(x_test[x_test.Facts == 4])
print(rmse(y_4, knn_grid_workzone ))
# all_clean 0
knn_grid_allclean = clf.predict(x_test[x_test.Facts == 0])
print(rmse(y_0, knn_grid_allclean ))
# Many     2
knn_grid_Many = clf.predict(x_test[x_test.Facts == 2])
print(rmse(y_2, knn_grid_Many ))

predictions = clf.predict(x_test)
dt_report = metrics.r2_score(y_test, predictions)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("RMSE %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print(means)

plt.plot(means)
plt.title("RMSE in different parameter settings")
plt.xlabel("Different parameter combination")
plt.ylabel("RMSE")
plt.show()