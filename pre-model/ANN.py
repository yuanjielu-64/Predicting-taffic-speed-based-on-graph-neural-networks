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
from sklearn.neural_network import MLPRegressor
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

x_train, x_test, y_train, y_test = model_selection. train_test_split(dataset, y, test_size=0.25, random_state = 1234)
my_scorer = make_scorer(rmse, greater_is_better=True)


parameters = {'activation': ['logistic','relu'],'hidden_layer_sizes': [12, 13, 14, 15, 16],'solver':['adam'],
              'batch_size': [24],'learning_rate_init':[0.001],'max_iter':[400],'learning_rate':['adaptive']}
mlp_model = model_selection.GridSearchCV(estimator= MLPRegressor(),param_grid = parameters, scoring = my_scorer,cv = 5, verbose = 2, n_jobs = 2, refit = True)
mlp_model.fit(x_train,y_train)
print(mlp_model.best_params_,mlp_model.best_score_)

mlp_grid_model = mlp_model.predict(x_test)
dt_report = metrics.r2_score(y_test, mlp_grid_model )

means = mlp_model.cv_results_['mean_test_score']
stds = mlp_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, mlp_model.cv_results_['params']):
        print("Accuracy %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print(means)
#
plt.plot(means)
plt.title(" in different parameter settings")
plt.xlabel("parameter(in order to check")
plt.ylabel("RMSE")
plt.show()