import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from math import sqrt
from sklearn.metrics.scorer import make_scorer
def rmse(true, predicted):
  return sqrt(mean_squared_error(np.expm1(true), np.expm1(predicted)))


#load dataframe
df = pd.read_csv('/Users/yuanjielu/Desktop/Research/Project/tmc_final.csv', index_col=0)

print(df)

target = df[['speed']]
target = np.array(target)
dataset = df.drop(columns=['speed'])

original_target = target
target = np.log1p(target)

x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=0.25)
reg = RandomForestRegressor(random_state=1234)
parameters={'n_estimators':[10,30,50,100],'max_depth':range(10,20),
            'bootstrap': [True, False], "max_features" : ["auto", "log2", "sqrt"] }

my_scorer = make_scorer(rmse, greater_is_better=True)

clf = GridSearchCV(reg, parameters, cv= 5, n_jobs = -1,verbose= 1, refit = True, scoring= my_scorer)
clf.fit(X=x_train, y=y_train.ravel())
predictions = clf.predict(x_test)
dt_report = metrics.r2_score(y_test, predictions)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("RMSE %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print(means)
#
plt.plot(means)
plt.title("RMSE in different parameter settings")
plt.xlabel("Different parameter combination")
plt.ylabel("RMSE")
plt.show()


