import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn import svm
from scipy.stats import norm
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import metrics
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.scorer import make_scorer
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.metrics.scorer import make_scorer

df = pd.read_csv(r'Y:\Project\tmc_final.csv', index_col = 0)

def rmse(true, predicted):
  return sqrt(mean_squared_error(np.expm1(true), np.expm1(predicted)))

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

# Find suitable parameter
epsilon = [0.1,1.5,0.2]
C = np.array([0.1,1,10])
gamma =np.array([0.01,0.1,1,10])
parameters = {'svr__C': C,'svr__gamma': gamma, 'svr__epsilon':epsilon}

clf = Pipeline([('ss',preprocessing.StandardScaler()), ('svr', svm.SVR())])
grid_svr = model_selection.GridSearchCV(clf,parameters, scoring = my_scorer,cv = 5, verbose = 2, n_jobs = 4, refit = True)
grid_svr.fit(x_train,y_train)
print(grid_svr.best_params_,grid_svr.best_score_)
pred_grid_svr = grid_svr.predict(x_test)

dt_report = metrics.r2_score(y_test, pred_grid_svr)

means = grid_svr.cv_results_['mean_test_score']
stds = grid_svr.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_svr.cv_results_['params']):
        print("Accuracy %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print(means)
#
plt.plot(means)
plt.title(" in different parameter settings")
plt.xlabel("parameter(in order to check)")
plt.ylabel("RMSE")
plt.show()
