import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

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
y = np.log1p(target)

x_train, x_test, y_train, y_test = model_selection. train_test_split(dataset, y, test_size=0.25, random_state = 1234)
my_scorer = make_scorer(rmse, greater_is_better=True)

sizes = np.logspace(2, 5, 7).astype(np.int)

for name, estimator in {"KNN": KNeighborsRegressor(weights='distance', n_neighbors= 10),
                        "SVR": SVR(kernel='rbf', C=1, gamma=0.01 ),
                        "RandomForest": RandomForestRegressor(random_state=1234,n_estimators = 50, max_depth= 16, bootstrap = True, max_features= 'auto')
                        }.items():

    train_time = []
    test_time = []
    for train_test_size in sizes:
        t0 = time.time()
        estimator.fit(x_train[:train_test_size], y_train[:train_test_size])
        train_time.append(time.time() - t0)
        print(t0)
        t0 = time.time()
        estimator.predict(x_test[:1000])
        test_time.append(time.time() - t0)

    if name == 'SVR':
        plt.plot(sizes, train_time, 'o-', color="r" , label="%s (train)" % name)
        plt.plot(sizes, test_time, 'o--', color="r", label="%s (test)" % name)
    elif name == 'KNN':
        plt.plot(sizes, train_time, 'o-', color="g", label="%s (train)" % name)
        plt.plot(sizes, test_time, 'o--', color="g", label="%s (test)" % name)
    else:
        plt.plot(sizes, train_time, 'o-', color="steelblue", label="%s (train)" % name)
        plt.plot(sizes, test_time, 'o--', color="steelblue", label="%s (test)" % name)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")
plt.title('Execution Time')
plt.legend(loc="best")
plt.show()